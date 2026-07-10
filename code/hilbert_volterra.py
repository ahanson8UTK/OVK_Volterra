"""Kernelized Hilbert/Fock-space Volterra targets.

The nonlinear state object here is never materialized as finite tensor
features.  The implementation computes only the finite-sample Gram matrix of
the infinite-level geometric Fock signature and then uses that Gram matrix to
define state-similarity weights.  The final OVK target remains a p-by-p moment
operator on the original LP score coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from time_series_targets import (
    TargetResult,
    make_bartlett_filtered_scores,
    make_time_weights,
    soft_reference_geometry,
)


Array = np.ndarray


@dataclass(frozen=True)
class HilbertVolterraKernelConfig:
    memory_half_lives: tuple[float, ...] = (3.0, 12.0, 36.0)
    memory_weights: str | tuple[float, ...] = "equal"
    gamma: float = 0.05
    base_inner: str = "reference_soft"
    rho: float | None = None
    strict_past: bool = True
    normalize_kernel: bool = True
    feature_bandwidth: str | float = "median"
    min_effective_sample_size: float | None = None


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _sym_last(A: Array) -> Array:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _row_normalize(W: Array) -> Array:
    W = np.asarray(W, dtype=float)
    W = np.where((W >= 0.0) & np.isfinite(W), W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        bad = np.flatnonzero(row_sum[:, 0] <= 0.0)
        for i in bad:
            W[i, i if i < W.shape[1] else 0] = 1.0
        row_sum = W.sum(axis=1, keepdims=True)
    return W / row_sum


def _outer_stack(scores: Array) -> Array:
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("scores must be a T-by-p array")
    return np.einsum("ti,tj->tij", S, S, optimize=True)


def _weighted_moment(outer: Array, weights: Array) -> Array:
    return _sym_last(np.einsum("st,tij->sij", np.asarray(weights, float), outer, optimize=True))


def _reference_from_weights(outer: Array, weights: Array, reference: str) -> tuple[Array, Array]:
    T = outer.shape[0]
    rule = reference.lower()
    if rule in {"empirical", "mean"}:
        a = np.full(T, 1.0 / T)
    elif rule in {"grid", "grid_induced", "grid_or_empirical", "induced"}:
        q = np.full(weights.shape[0], 1.0 / max(weights.shape[0], 1))
        a = q @ np.asarray(weights, dtype=float)
        total = float(a.sum())
        if total <= 0.0:
            raise ValueError("grid-induced reference weights have zero mass")
        a = a / total
    else:
        raise ValueError(f"unknown reference rule: {reference}")
    C = _sym(np.einsum("t,tij->ij", a, outer, optimize=True))
    return C, a


def _memory_arrays(config: HilbertVolterraKernelConfig) -> tuple[Array, Array, dict[str, Any]]:
    half_lives = tuple(float(h) for h in config.memory_half_lives)
    if not half_lives or any(h <= 0.0 or not np.isfinite(h) for h in half_lives):
        raise ValueError("memory_half_lives must be positive finite values")
    decays = np.asarray([math.exp(-math.log(2.0) / h) for h in half_lives], dtype=float)
    if isinstance(config.memory_weights, str):
        if config.memory_weights.lower() != "equal":
            raise ValueError("memory_weights string must be 'equal'")
        weights = np.full(len(half_lives), 1.0 / len(half_lives), dtype=float)
        rule = "equal"
    else:
        weights = np.asarray(config.memory_weights, dtype=float)
        if weights.shape != (len(half_lives),):
            raise ValueError("memory_weights must match memory_half_lives")
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("memory_weights must be nonnegative with positive mass")
        weights = weights / weights.sum()
        rule = "custom_normalized"
    meta = {
        "memory_half_lives": list(half_lives),
        "memory_decays": decays.tolist(),
        "memory_weights": weights.tolist(),
        "memory_weights_rule": rule,
    }
    return decays, weights, meta


def check_kernel_psd(K: Array, tol: float = 1e-8) -> dict[str, Any]:
    """Symmetry, finite, eigenvalue, and condition diagnostics for a kernel."""
    A = np.asarray(K, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("K must be square")
    finite = bool(np.isfinite(A).all())
    sym_err = float(np.max(np.abs(A - A.T))) if A.size else 0.0
    S = _sym(A)
    eig = np.linalg.eigvalsh(S) if finite else np.asarray([np.nan])
    pos = eig[eig > tol]
    cond = float(pos.max() / pos.min()) if len(pos) else float("inf")
    return {
        "n": int(A.shape[0]),
        "finite": finite,
        "max_symmetry_error": sym_err,
        "min_eigenvalue": float(np.nanmin(eig)),
        "median_eigenvalue": float(np.nanmedian(eig)),
        "max_eigenvalue": float(np.nanmax(eig)),
        "condition_positive_spectrum": cond,
        "tol": float(tol),
        "ok": bool(finite and sym_err <= tol and float(np.nanmin(eig)) >= -tol),
    }


def check_moment_psd(K_by_state: Array, C_ref: Array, tol: float = 1e-8) -> dict[str, Any]:
    """Symmetry/PSD diagnostics for the p-by-p moment target path."""
    K = np.asarray(K_by_state, dtype=float)
    if K.ndim != 3 or K.shape[1] != K.shape[2]:
        raise ValueError("K_by_state must have shape S-by-p-by-p")
    C = np.asarray(C_ref, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C_ref must be square")
    sym_err = np.max(np.abs(K - np.swapaxes(K, 1, 2)), axis=(1, 2))
    eig_min = np.asarray([float(np.linalg.eigvalsh(_sym(A)).min()) for A in K])
    C_sym = float(np.max(np.abs(C - C.T)))
    C_eig = np.linalg.eigvalsh(_sym(C))
    return {
        "K_by_state": {
            "n_matrices": int(K.shape[0]),
            "max_symmetry_error": float(np.max(sym_err)),
            "min_eigenvalue": float(np.min(eig_min)),
            "median_min_eigenvalue": float(np.median(eig_min)),
            "ok": bool(np.max(sym_err) <= tol and np.min(eig_min) >= -tol),
        },
        "C_ref": {
            "max_symmetry_error": C_sym,
            "min_eigenvalue": float(np.min(C_eig)),
            "ok": bool(C_sym <= tol and np.min(C_eig) >= -tol),
        },
        "tol": float(tol),
    }


def compute_base_score_gram(
    psi: Array,
    C_base: Array | None = None,
    rho: float | None = None,
    method: str = "reference_soft",
) -> tuple[Array, dict[str, Any]]:
    """
    Return c[i,j] = <psi_i, psi_j>_base and scaling metadata.

    The default inner product is the old ridge-soft reference geometry:
    u' (C_base + rho I)^(-1) v / d_rho.  The linear solve uses the ridge
    matrix directly and never forms an explicit inverse.
    """
    scores = np.asarray(psi, dtype=float)
    if scores.ndim != 2:
        raise ValueError("psi must be a T-by-p array")
    if not np.isfinite(scores).all():
        raise ValueError("psi contains nonfinite values")
    method_key = method.lower()
    meta: dict[str, Any] = {"base_inner": method_key, "score_shape": list(scores.shape)}
    if method_key == "euclidean":
        raw = scores @ scores.T
        C_used = _sym((scores.T @ scores) / max(len(scores), 1))
        ref = soft_reference_geometry(C_used, rho=rho)
        meta.update(
            {
                "C_base_trace": float(np.trace(C_used)),
                "rho": float(ref["rho"]),
                "d_rho": float(ref["d_rho"]),
                "reference_soft_diagnostics_available": True,
            }
        )
    elif method_key == "reference_soft":
        C_used = _sym(np.asarray(C_base, dtype=float)) if C_base is not None else _sym((scores.T @ scores) / max(len(scores), 1))
        ref = soft_reference_geometry(C_used, rho=rho)
        D = np.asarray(ref["D_rho"], dtype=float)
        solved = np.linalg.solve(D, scores.T).T
        raw = (scores @ solved.T) / float(ref["d_rho"])
        meta.update(
            {
                "C_base_trace": float(np.trace(C_used)),
                "C_base_min_eigenvalue": float(np.linalg.eigvalsh(C_used).min()),
                "rho": float(ref["rho"]),
                "d_rho": float(ref["d_rho"]),
            }
        )
    else:
        raise ValueError("method must be 'reference_soft' or 'euclidean'")

    raw = _sym(np.asarray(raw, dtype=float))
    diag = np.diag(raw)
    positive_diag = diag[np.isfinite(diag) & (diag > 1e-14)]
    scale = float(np.median(positive_diag)) if len(positive_diag) else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    c = raw / scale
    meta.update(
        {
            "base_gram_scale_divisor": scale,
            "base_gram_diag_min_before_scale": float(np.nanmin(diag)),
            "base_gram_diag_median_positive_before_scale": float(np.median(positive_diag)) if len(positive_diag) else None,
            "base_gram_diag_median_positive_after_scale": float(np.median(np.diag(c)[np.diag(c) > 1e-14]))
            if np.any(np.diag(c) > 1e-14)
            else None,
            "base_gram_psd": check_kernel_psd(c),
        }
    )
    if not np.isfinite(c).all():
        raise FloatingPointError("base score Gram contains nonfinite entries")
    return c, meta


def compute_hilbert_volterra_gram(
    c: Array,
    dates: pd.Series | Array | list[Any] | None = None,
    config: HilbertVolterraKernelConfig | None = None,
) -> tuple[Array, Array, Array, dict[str, Any]]:
    """
    Compute the infinite-level discrete Volterra/Fock Gram matrix.

    The default strict-past endpoint t uses increments u < t.  When
    ``strict_past=False`` the endpoint's current increment is included by a
    scalar-resolved convention for the current-current recursion term; this is
    for sensitivity only and is not used by the main empirical target.
    """
    del dates
    cfg = config or HilbertVolterraKernelConfig()
    gram = _sym(np.asarray(c, dtype=float))
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("c must be a square base Gram matrix")
    if cfg.gamma <= 0.0 or not np.isfinite(cfg.gamma):
        raise ValueError("gamma must be positive and finite")
    if not np.isfinite(gram).all():
        raise ValueError("base Gram contains nonfinite entries")

    decays, weights, memory_meta = _memory_arrays(cfg)
    T = gram.shape[0]
    J = len(decays)
    S = np.zeros((J, J, T, T), dtype=float)
    raw = np.zeros((T, T), dtype=float)
    R = np.zeros((T, T), dtype=float)
    current_weight_sum = float(np.sum(weights[:, None] * weights[None, :]))

    for t in range(T):
        for u in range(T):
            if cfg.strict_past:
                prefix = 0.0
                if t > 0 and u > 0:
                    for q in range(J):
                        for r in range(J):
                            prefix += weights[q] * weights[r] * decays[q] * decays[r] * S[q, r, t - 1, u - 1]
                kij = 1.0 + cfg.gamma * prefix
                pre_s = None
            else:
                weighted_prefix = 0.0
                pre_s = np.zeros((J, J), dtype=float)
                for q in range(J):
                    for r in range(J):
                        val = 0.0
                        if t > 0:
                            val += decays[q] * S[q, r, t - 1, u]
                        if u > 0:
                            val += decays[r] * S[q, r, t, u - 1]
                        if t > 0 and u > 0:
                            val -= decays[q] * decays[r] * S[q, r, t - 1, u - 1]
                        pre_s[q, r] = val
                        weighted_prefix += weights[q] * weights[r] * val
                denom = 1.0 - cfg.gamma * gram[t, u] * current_weight_sum
                if denom <= 1e-12 or not np.isfinite(denom):
                    raise FloatingPointError(
                        "include-current Hilbert-Volterra recursion is unstable; "
                        "use strict_past=True or a smaller gamma"
                    )
                kij = (1.0 + cfg.gamma * weighted_prefix) / denom
            raw[t, u] = kij
            R[t, u] = gram[t, u] * kij
            if not np.isfinite(raw[t, u]) or not np.isfinite(R[t, u]):
                raise FloatingPointError(
                    f"nonfinite Fock kernel entry at {(t, u)} with gamma={cfg.gamma}, "
                    f"base_gram_range=({float(np.min(gram))}, {float(np.max(gram))}), "
                    f"memory_half_lives={cfg.memory_half_lives}"
                )
            for q in range(J):
                for r in range(J):
                    if pre_s is None:
                        val = R[t, u]
                        if t > 0:
                            val += decays[q] * S[q, r, t - 1, u]
                        if u > 0:
                            val += decays[r] * S[q, r, t, u - 1]
                        if t > 0 and u > 0:
                            val -= decays[q] * decays[r] * S[q, r, t - 1, u - 1]
                    else:
                        val = pre_s[q, r] + R[t, u]
                    S[q, r, t, u] = val

    raw_symmetry_error_before = float(np.max(np.abs(raw - raw.T)))
    raw = _sym(raw)
    raw_diag = np.diag(raw)
    if np.any(~np.isfinite(raw_diag)) or np.any(raw_diag <= 0.0):
        raise FloatingPointError(
            f"Fock kernel diagonal must be positive and finite; "
            f"diag_range=({float(np.nanmin(raw_diag))}, {float(np.nanmax(raw_diag))})"
        )
    denom = np.sqrt(np.outer(raw_diag, raw_diag))
    kappa_norm = raw / denom if cfg.normalize_kernel else raw.copy()
    kappa_norm = _sym(kappa_norm)
    d2 = np.diag(kappa_norm)[:, None] + np.diag(kappa_norm)[None, :] - 2.0 * kappa_norm
    roundoff_negative = d2 < 0.0
    tiny_negative = roundoff_negative & (d2 >= -1e-10)
    d2[tiny_negative] = 0.0
    if np.any(d2 < -1e-10):
        raise FloatingPointError(f"Fock distance has large negative squared entries; min={float(np.min(d2)):.6g}")
    distance = np.sqrt(np.maximum(d2, 0.0))
    positive_dist = distance[np.isfinite(distance) & (distance > 0.0)]
    meta = {
        "target_kernel": "infinite_level_hilbert_fock_volterra",
        "strict_past": bool(cfg.strict_past),
        "normalize_kernel": bool(cfg.normalize_kernel),
        "signature_gamma": float(cfg.gamma),
        "raw_symmetry_error_before_symmetrization": raw_symmetry_error_before,
        "raw_kernel_min": float(np.min(raw)),
        "raw_kernel_max": float(np.max(raw)),
        "raw_kernel_diag_min": float(np.min(raw_diag)),
        "raw_kernel_diag_max": float(np.max(raw_diag)),
        "normalized_kernel_min": float(np.min(kappa_norm)),
        "normalized_kernel_max": float(np.max(kappa_norm)),
        "normalized_kernel_diag_min": float(np.min(np.diag(kappa_norm))),
        "normalized_kernel_diag_max": float(np.max(np.diag(kappa_norm))),
        "distance_min": float(np.min(distance)),
        "distance_median_positive": float(np.median(positive_dist)) if len(positive_dist) else 0.0,
        "distance_max": float(np.max(distance)),
        "distance_roundoff_clipped_count": int(np.sum(tiny_negative)),
        "raw_kernel_psd": check_kernel_psd(raw),
        "normalized_kernel_psd": check_kernel_psd(kappa_norm),
        **memory_meta,
    }
    return raw, kappa_norm, distance, meta


def make_hilbert_volterra_weights(
    dates: pd.Series | Array | list[Any],
    kappa_norm: Array,
    distance: Array,
    time_bandwidth: float | None,
    feature_bandwidth: str | float = "median",
    valid_index: Array | None = None,
    base_time_kernel: str = "graph_resolvent",
) -> tuple[Array, dict[str, Any]]:
    """Return W_HV proportional to calendar weights times Fock RBF weights."""
    del kappa_norm
    dist = np.asarray(distance, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distance must be square")
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if valid_index is not None and len(date_series) != dist.shape[0]:
        idx = np.asarray(valid_index, dtype=int)
        date_series = date_series.iloc[idx].reset_index(drop=True)
    if len(date_series) != dist.shape[0]:
        raise ValueError("dates length must match distance rows after valid_index handling")
    positive = dist[np.isfinite(dist) & (dist > 0.0)]
    if isinstance(feature_bandwidth, str):
        if feature_bandwidth.lower() != "median":
            raise ValueError("feature_bandwidth string must be 'median'")
        h_fock = float(np.median(positive)) if len(positive) else float("nan")
        if not np.isfinite(h_fock) or h_fock <= 0.0:
            h_fock = 1.0
        bandwidth_rule = "median_positive_distance"
    else:
        h_fock = float(feature_bandwidth)
        if not np.isfinite(h_fock) or h_fock <= 0.0:
            raise ValueError("numeric feature_bandwidth must be positive and finite")
        bandwidth_rule = "fixed"
    W_time = make_time_weights(date_series, bandwidth=time_bandwidth, kernel=base_time_kernel)
    W_feature = np.exp(-0.5 * (dist**2) / max(h_fock**2, 1e-300))
    W = _row_normalize(W_time * W_feature)
    ess = 1.0 / np.maximum(np.sum(W**2, axis=1), 1e-300)
    meta = {
        "base_time_kernel": base_time_kernel,
        "time_bandwidth": time_bandwidth,
        "feature_bandwidth": h_fock,
        "feature_bandwidth_rule": bandwidth_rule,
        "ESS_min": float(np.min(ess)),
        "ESS_median": float(np.median(ess)),
        "ESS_max": float(np.max(ess)),
        "normalized_ESS_min": float(np.min(ess / max(W.shape[1], 1))),
        "normalized_ESS_median": float(np.median(ess / max(W.shape[1], 1))),
        "normalized_ESS_max": float(np.max(ess / max(W.shape[1], 1))),
        "row_sum_error": float(np.max(np.abs(W.sum(axis=1) - 1.0))),
        "min_weight": float(np.min(W)),
        "max_weight": float(np.max(W)),
    }
    return W, meta


def make_hilbert_volterra_target(
    psi: Array,
    dates: pd.Series | Array | list[Any],
    Z: Array | None = None,
    hac_lags: int = 12,
    config: HilbertVolterraKernelConfig | None = None,
    time_bandwidth: float | None = None,
    reference: str = "grid_induced",
) -> TargetResult:
    """
    Build K_HV(s) = sum_t W_HV[s,t] Z_t Z_t' on the original p-space.

    The Fock kernel is computed from all score coordinates and then subset to
    the valid HAC-filtered endpoint dates.  With strict-past default, endpoint
    t's nonlinear history similarity uses score increments u < t.
    """
    cfg = config or HilbertVolterraKernelConfig()
    scores = np.asarray(psi, dtype=float)
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if scores.ndim != 2:
        raise ValueError("psi must be a T-by-p array")
    if len(date_series) != scores.shape[0]:
        raise ValueError("dates length must match psi rows")
    if Z is None:
        filtered, valid_index, filter_meta = make_bartlett_filtered_scores(scores, L=hac_lags, drop_initial=True)
    else:
        filtered = np.asarray(Z, dtype=float)
        if filtered.ndim != 2 or filtered.shape[1] != scores.shape[1]:
            raise ValueError("Z must have shape S-by-p")
        valid_index = np.arange(filtered.shape[0], dtype=int)
        filter_meta = {
            "L": int(hac_lags),
            "filter": "provided_Z",
            "filter_weights": None,
            "drop_initial": False,
            "boundary_renormalize": False,
            "valid_count": int(len(valid_index)),
        }
    state_dates = date_series.iloc[valid_index].reset_index(drop=True)

    base_gram, base_meta = compute_base_score_gram(scores, rho=cfg.rho, method=cfg.base_inner)
    kappa_raw_full, kappa_norm_full, distance_full, kernel_meta = compute_hilbert_volterra_gram(
        base_gram,
        dates=date_series,
        config=cfg,
    )
    ix = np.asarray(valid_index, dtype=int)
    kappa_raw = kappa_raw_full[np.ix_(ix, ix)]
    kappa_norm = kappa_norm_full[np.ix_(ix, ix)]
    distance = distance_full[np.ix_(ix, ix)]
    W, weight_meta = make_hilbert_volterra_weights(
        state_dates,
        kappa_norm,
        distance,
        time_bandwidth=time_bandwidth,
        feature_bandwidth=cfg.feature_bandwidth,
        base_time_kernel="graph_resolvent",
    )
    outer = _outer_stack(filtered)
    K = _weighted_moment(outer, W)
    C, ref_weights = _reference_from_weights(outer, W, reference)
    moment_diag = check_moment_psd(K, C)
    meta = {
        "target_type": "hilbert_volterra",
        "score_shape": list(scores.shape),
        "filtered_score_shape": list(filtered.shape),
        "reference_rule": reference,
        "reference_weights": ref_weights.tolist(),
        "base_score_gram": base_meta,
        "hilbert_volterra_kernel": kernel_meta,
        "hilbert_volterra_weights": weight_meta,
        "kappa_raw_shape": list(kappa_raw.shape),
        "kappa_norm_shape": list(kappa_norm.shape),
        "moment_psd_diagnostics": moment_diag,
        **filter_meta,
    }
    result = TargetResult(K, C, state_dates, W, filtered, meta, valid_index=valid_index)
    result.metadata["kappa_raw"] = kappa_raw
    result.metadata["kappa_norm"] = kappa_norm
    result.metadata["fock_distance"] = distance
    return result

