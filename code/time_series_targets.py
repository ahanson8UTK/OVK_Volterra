"""Time-series OVK targets for monetary-policy LP response scores.

The constructors in this module keep the target operators on the original
LP coefficient/influence grid.  Memory enters either by filtering scores before
forming outer products or by changing the state weights with low-dimensional
Volterra history features.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class TargetResult:
    """Container returned by target constructors."""

    K_by_state: Array
    C_ref: Array
    state_dates: pd.Series
    weights: Array
    scores: Array
    metadata: dict[str, Any]
    valid_index: Array | None = None
    features: Array | None = None


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _sym_last(A: Array) -> Array:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _month_index(dates: pd.Series | Array | list[Any]) -> Array:
    dt = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if dt.isna().any():
        raise ValueError("dates contain non-parsable values")
    return dt.dt.year.to_numpy(int) * 12 + dt.dt.month.to_numpy(int)


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


def _doubly_stochastic_symmetric(W: Array, tol: float = 1e-12, max_iter: int = 1000) -> Array:
    out = _sym(np.asarray(W, dtype=float))
    tiny_negative = (out < 0.0) & (out >= -tol)
    out[tiny_negative] = 0.0
    if float(np.min(out)) < -tol:
        raise ValueError(f"weight matrix must be nonnegative; min={float(np.min(out)):.6g}")
    for _ in range(max_iter):
        row_sum = out.sum(axis=1)
        if np.all(np.abs(row_sum - 1.0) <= tol):
            return _sym(out)
        if np.any(row_sum <= 0.0):
            raise ValueError("weight matrix contains a zero-mass row")
        scale = 1.0 / np.sqrt(row_sum)
        out = scale[:, None] * out * scale[None, :]
    row_sum = out.sum(axis=1)
    if np.max(np.abs(row_sum - 1.0)) > 1e-8:
        raise RuntimeError("symmetric stochastic normalization did not converge")
    return _sym(out)


def _path_graph_laplacian(dates: pd.Series | Array | list[Any]) -> Array:
    mi = _month_index(dates)
    gaps = np.diff(mi)
    if np.any(gaps <= 0):
        raise ValueError("dates must be strictly increasing for graph-resolvent weights")
    T = len(mi)
    Lmat = np.zeros((T, T), dtype=float)
    edge_weights = 1.0 / np.maximum(gaps.astype(float), 1.0) ** 2
    for i, w in enumerate(edge_weights):
        Lmat[i, i] += w
        Lmat[i + 1, i + 1] += w
        Lmat[i, i + 1] -= w
        Lmat[i + 1, i] -= w
    return _sym(Lmat)


def make_time_weights(
    dates: pd.Series | Array | list[Any],
    bandwidth: float | None,
    kernel: str = "gaussian",
    valid_mask: Array | None = None,
    **kwargs: Any,
) -> Array:
    """
    Return nonnegative row-normalized calendar-time weights.

    For ``kernel="graph_resolvent"`` this reproduces the old full-coordinate
    temporal smoother: eta * (eta I + graph_laplacian)^(-1), followed by the
    same symmetric doubly-stochastic normalization.
    """
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape[0] != len(date_series):
            raise ValueError("valid_mask length must match dates")
        date_series = date_series.loc[mask].reset_index(drop=True)
    T = len(date_series)
    if T == 0:
        raise ValueError("cannot build weights for an empty date vector")
    kernel_key = kernel.lower().replace("-", "_")
    if kernel_key in {"graph", "graph_resolvent", "old", "old_temporal"}:
        eta = 0.08 if bandwidth is None else float(bandwidth)
        if eta <= 0.0:
            return np.eye(T)
        Lmat = _path_graph_laplacian(date_series)
        W = eta * np.linalg.solve(eta * np.eye(T) + Lmat, np.eye(T))
        return _doubly_stochastic_symmetric(W)
    if kernel_key in {"identity", "none"}:
        return np.eye(T)

    mi = _month_index(date_series).astype(float)
    dist = np.abs(mi[:, None] - mi[None, :])
    bw = float(bandwidth) if bandwidth is not None else float(kwargs.get("default_bandwidth", 12.0))
    if bw <= 0.0:
        return np.eye(T)
    if kernel_key == "gaussian":
        W = np.exp(-0.5 * (dist / bw) ** 2)
    elif kernel_key == "bartlett":
        W = np.maximum(1.0 - dist / (bw + 1.0), 0.0)
    elif kernel_key in {"uniform", "boxcar"}:
        W = (dist <= bw).astype(float)
    else:
        raise ValueError(f"unknown time kernel: {kernel}")
    return _row_normalize(W)


def _outer_stack(scores: Array) -> Array:
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("scores must be a two-dimensional T-by-p array")
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


def make_diagonal_old_target(
    psi: Array,
    dates: pd.Series | Array | list[Any],
    time_bandwidth: float | None = None,
    kernel: str = "graph_resolvent",
    reference: str = "empirical",
    **kwargs: Any,
) -> TargetResult:
    """Old diagonal observation-level target K(s)=sum_t w(s,t) psi_t psi_t'."""
    scores = np.asarray(psi, dtype=float)
    state_dates = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if len(state_dates) != scores.shape[0]:
        raise ValueError("dates length must match psi rows")
    W = make_time_weights(state_dates, bandwidth=time_bandwidth, kernel=kernel, **kwargs)
    outer = _outer_stack(scores)
    K = _weighted_moment(outer, W)
    C, ref_weights = _reference_from_weights(outer, W, reference)
    meta = {
        "target_type": "diagonal_old",
        "time_kernel": kernel,
        "time_bandwidth": time_bandwidth,
        "reference_rule": reference,
        "reference_weights": ref_weights.tolist(),
    }
    return TargetResult(K, C, state_dates, W, scores, meta, valid_index=np.arange(scores.shape[0]))


def make_bartlett_filtered_scores(
    psi: Array,
    L: int,
    drop_initial: bool = True,
    boundary_renormalize: bool = False,
) -> tuple[Array, Array, dict[str, Any]]:
    """
    Build Z_t = (1/sqrt(L+1)) sum_{ell=0}^L psi_{t-ell} on the valid sample.

    With the default ``drop_initial=True`` the first L observations are dropped,
    so every retained filtered score uses exactly the same Bartlett factor
    filter.  Boundary renormalization is available for sensitivity checks but is
    not the empirical default.
    """
    scores = np.asarray(psi, dtype=float)
    if scores.ndim != 2:
        raise ValueError("psi must be a T-by-p array")
    lag = int(L)
    if lag < 0:
        raise ValueError("L must be nonnegative")
    T, p = scores.shape
    if drop_initial and T <= lag:
        raise ValueError("not enough observations for requested L with drop_initial=True")

    rows: list[Array] = []
    valid: list[int] = []
    if drop_initial:
        for t in range(lag, T):
            rows.append(scores[t - lag : t + 1].sum(axis=0) / math.sqrt(lag + 1.0))
            valid.append(t)
    else:
        for t in range(T):
            available = min(lag, t) + 1
            denom = math.sqrt(float(available if boundary_renormalize else lag + 1))
            rows.append(scores[t - available + 1 : t + 1].sum(axis=0) / denom)
            valid.append(t)
    Z = np.vstack(rows).reshape(len(rows), p)
    base_weights = np.full(lag + 1, 1.0 / math.sqrt(lag + 1.0))
    meta = {
        "L": lag,
        "filter": "bartlett_factor",
        "filter_weights": base_weights.tolist(),
        "drop_initial": bool(drop_initial),
        "boundary_renormalize": bool(boundary_renormalize),
        "valid_count": int(len(valid)),
    }
    return Z, np.asarray(valid, dtype=int), meta


def make_exponential_filtered_scores(
    psi: Array,
    half_lives: tuple[float, ...] | list[float],
) -> tuple[Array, dict[str, Any]]:
    """Return exponentially filtered score states for optional robustness work."""
    scores = np.asarray(psi, dtype=float)
    if scores.ndim != 2:
        raise ValueError("psi must be a T-by-p array")
    filters = []
    alphas = []
    for half_life in half_lives:
        H = float(half_life)
        if H <= 0.0:
            raise ValueError("half-lives must be positive")
        a = math.exp(-math.log(2.0) / H)
        z = np.zeros(scores.shape[1], dtype=float)
        path = np.empty_like(scores)
        for t, row in enumerate(scores):
            z = a * z + row
            path[t] = z
        filters.append(path)
        alphas.append(a)
    stacked = np.stack(filters, axis=0)
    return stacked, {"half_lives": [float(x) for x in half_lives], "decay_factors": alphas}


def make_hac_filtered_target(
    psi: Array,
    dates: pd.Series | Array | list[Any],
    L: int = 12,
    time_bandwidth: float | None = None,
    reference: str = "empirical",
    kernel: str = "graph_resolvent",
    drop_initial: bool = True,
    boundary_renormalize: bool = False,
    **kwargs: Any,
) -> TargetResult:
    """HAC-aware first-level Volterra / filtered-score target."""
    scores = np.asarray(psi, dtype=float)
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if len(date_series) != scores.shape[0]:
        raise ValueError("dates length must match psi rows")
    Z, valid_index, filter_meta = make_bartlett_filtered_scores(
        scores,
        L=L,
        drop_initial=drop_initial,
        boundary_renormalize=boundary_renormalize,
    )
    state_dates = date_series.iloc[valid_index].reset_index(drop=True)
    W = make_time_weights(state_dates, bandwidth=time_bandwidth, kernel=kernel, **kwargs)
    outer = _outer_stack(Z)
    K = _weighted_moment(outer, W)
    C, ref_weights = _reference_from_weights(outer, W, reference)
    meta = {
        "target_type": "hac_filtered",
        "score_shape": list(scores.shape),
        "filtered_score_shape": list(Z.shape),
        "time_kernel": kernel,
        "time_bandwidth": time_bandwidth,
        "reference_rule": reference,
        "reference_weights": ref_weights.tolist(),
        **filter_meta,
    }
    return TargetResult(K, C, state_dates, W, Z, meta, valid_index=valid_index)


def make_projected_score_path(
    psi: Array,
    basis: str = "reference_eig",
    r: int = 5,
    C_ref: Array | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Project scores to a standardized low-dimensional history path."""
    warnings.warn(
        "Finite Volterra rank/level features are legacy. The main Hilbert-Volterra target "
        "uses a kernelized infinite-level Fock Gram matrix and does not construct Phi_t.",
        DeprecationWarning,
        stacklevel=2,
    )
    scores = np.asarray(psi, dtype=float)
    if scores.ndim != 2:
        raise ValueError("psi must be a T-by-p array")
    rank = max(1, min(int(r), scores.shape[1]))
    if basis != "reference_eig":
        raise ValueError("only basis='reference_eig' is currently implemented")
    C = _sym(np.asarray(C_ref, dtype=float)) if C_ref is not None else _sym((scores.T @ scores) / max(len(scores), 1))
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    loadings = eigvecs[:, :rank]
    raw = scores @ loadings
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    x = (raw - mean) / scale
    total = float(np.sum(np.maximum(eigvals, 0.0)))
    explained = np.maximum(eigvals[:rank], 0.0) / total if total > 0.0 else np.zeros(rank)
    meta = {
        "basis": basis,
        "rank": rank,
        "loadings": loadings.tolist(),
        "eigenvalues": eigvals[:rank].tolist(),
        "explained_reference_moment_share": explained.tolist(),
        "cumulative_explained_reference_moment_share": float(np.sum(explained)),
        "projected_mean": mean.tolist(),
        "projected_scale": scale.tolist(),
    }
    return x, meta


def _standardize_features(Phi: Array) -> tuple[Array, Array, Array]:
    mean = Phi.mean(axis=0)
    scale = Phi.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return (Phi - mean) / scale, mean, scale


def _pca_reduce(Phi: Array, pca_dim: int) -> tuple[Array, dict[str, Any]]:
    dim = max(1, min(int(pca_dim), Phi.shape[1], Phi.shape[0]))
    centered = Phi - Phi.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    reduced = U[:, :dim] * s[:dim]
    variance = s**2
    total = float(np.sum(variance))
    shares = variance[:dim] / total if total > 0.0 else np.zeros(dim)
    meta = {
        "pca_applied": True,
        "pca_dim": dim,
        "pca_components": Vt[:dim].tolist(),
        "pca_feature_mean": Phi.mean(axis=0).tolist(),
        "pca_variance_shares": shares.tolist(),
        "pca_cumulative_variance_share": float(np.sum(shares)),
    }
    return reduced, meta


def make_volterra_signature_features(
    x: Array,
    half_lives: tuple[float, ...] | list[float] = (3, 12, 36),
    level: int = 2,
    standardize: bool = True,
    pca_dim: int | None = 10,
) -> tuple[Array, dict[str, Any]]:
    """Compute truncated exponential Volterra signature-style features."""
    warnings.warn(
        "Finite Volterra rank/level features are legacy. The main Hilbert-Volterra target "
        "uses a kernelized infinite-level Fock Gram matrix and does not construct Phi_t.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = np.asarray(x, dtype=float)
    if path.ndim != 2:
        raise ValueError("x must be a T-by-r array")
    lev = int(level)
    if lev < 1 or lev > 3:
        raise ValueError("level must be 1, 2, or 3")
    T, r = path.shape
    blocks: list[Array] = []
    decay_factors: list[float] = []
    for half_life in half_lives:
        H = float(half_life)
        if H <= 0.0:
            raise ValueError("half-lives must be positive")
        a = math.exp(-math.log(2.0) / H)
        decay_factors.append(a)
        h1 = np.zeros(r, dtype=float)
        h2 = np.zeros((r, r), dtype=float)
        h3 = np.zeros((r, r, r), dtype=float)
        rows = []
        for xt in path:
            prev_h1 = h1.copy()
            prev_h2 = h2.copy()
            if lev >= 3:
                h3 = a * h3 + np.einsum("ij,k->ijk", a * prev_h2, xt, optimize=True)
            if lev >= 2:
                h2 = a * h2 + np.outer(a * prev_h1, xt)
            h1 = a * h1 + xt
            parts = [h1.ravel()]
            if lev >= 2:
                parts.append(h2.ravel())
            if lev >= 3:
                parts.append(h3.ravel())
            rows.append(np.concatenate(parts))
        blocks.append(np.vstack(rows))
    raw = np.hstack(blocks) if blocks else np.empty((T, 0))
    Phi = raw
    feature_meta: dict[str, Any] = {
        "half_lives": [float(h) for h in half_lives],
        "decay_factors": decay_factors,
        "level": lev,
        "input_rank": int(r),
        "raw_feature_dim": int(raw.shape[1]),
        "standardized": bool(standardize),
    }
    if standardize and raw.shape[1]:
        Phi, mean, scale = _standardize_features(raw)
        feature_meta["feature_mean"] = mean.tolist()
        feature_meta["feature_scale"] = scale.tolist()
    if pca_dim is not None and Phi.shape[1] > 100:
        Phi, pca_meta = _pca_reduce(Phi, int(pca_dim))
        feature_meta.update(pca_meta)
    else:
        feature_meta.update({"pca_applied": False, "pca_dim": None})
    if not np.isfinite(Phi).all():
        raise FloatingPointError("Volterra features contain non-finite values")
    feature_meta["feature_dim"] = int(Phi.shape[1])
    return Phi, feature_meta


def make_volterra_nonlinear_weights(
    dates: pd.Series | Array | list[Any],
    Phi: Array,
    time_bandwidth: float | None,
    feature_bandwidth: str | float = "median",
    valid_index: Array | None = None,
    kernel: str = "graph_resolvent",
) -> tuple[Array, dict[str, Any]]:
    """Blend old calendar weights with nonlinear Volterra-state similarity."""
    features = np.asarray(Phi, dtype=float)
    if features.ndim != 2:
        raise ValueError("Phi must be a T-by-d array")
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if valid_index is not None and len(date_series) != features.shape[0]:
        idx = np.asarray(valid_index, dtype=int)
        date_series = date_series.iloc[idx].reset_index(drop=True)
    if len(date_series) != features.shape[0]:
        raise ValueError("dates length must match Phi rows after valid_index handling")
    W_time = make_time_weights(date_series, bandwidth=time_bandwidth, kernel=kernel)
    diff = features[:, None, :] - features[None, :, :]
    d2 = np.einsum("std,std->st", diff, diff, optimize=True)
    distances = np.sqrt(np.maximum(d2, 0.0))
    if isinstance(feature_bandwidth, str):
        key = feature_bandwidth.lower()
        if key != "median":
            raise ValueError("feature_bandwidth string must be 'median'")
        positive = distances[np.isfinite(distances) & (distances > 0.0)]
        h_phi = float(np.median(positive)) if positive.size else float("nan")
        if not np.isfinite(h_phi) or h_phi <= 0.0:
            h_phi = math.sqrt(max(features.shape[1], 1))
        bandwidth_rule = "median_positive_distance"
    else:
        h_phi = float(feature_bandwidth)
        if not np.isfinite(h_phi) or h_phi <= 0.0:
            h_phi = math.sqrt(max(features.shape[1], 1))
        bandwidth_rule = "fixed"
    W_feature = np.exp(-0.5 * d2 / max(h_phi**2, 1e-12))
    W = _row_normalize(W_time * W_feature)
    meta = {
        "time_kernel": kernel,
        "time_bandwidth": time_bandwidth,
        "feature_bandwidth": h_phi,
        "feature_bandwidth_rule": bandwidth_rule,
        "feature_dim": int(features.shape[1]),
    }
    return W, meta


def make_volterra_nonlinear_target(
    psi: Array,
    dates: pd.Series | Array | list[Any],
    L: int = 12,
    r: int = 5,
    level: int = 2,
    half_lives: tuple[float, ...] | list[float] = (3, 12, 36),
    time_bandwidth: float | None = None,
    feature_bandwidth: str | float = "median",
    reference: str = "grid_or_empirical",
    kernel: str = "graph_resolvent",
    pca_dim: int | None = 10,
    **kwargs: Any,
) -> TargetResult:
    """Nonlinear Volterra-featured HAC OVK target."""
    warnings.warn(
        "Finite Volterra rank/level features are legacy. The main Hilbert-Volterra target "
        "uses a kernelized infinite-level Fock Gram matrix and does not construct Phi_t.",
        DeprecationWarning,
        stacklevel=2,
    )
    scores = np.asarray(psi, dtype=float)
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce").reset_index(drop=True)
    if len(date_series) != scores.shape[0]:
        raise ValueError("dates length must match psi rows")
    Z, valid_index, filter_meta = make_bartlett_filtered_scores(scores, L=L, drop_initial=True)
    state_dates = date_series.iloc[valid_index].reset_index(drop=True)
    psi_valid = scores[valid_index]
    C_projection = _sym((psi_valid.T @ psi_valid) / max(len(psi_valid), 1))
    x, projection_meta = make_projected_score_path(psi_valid, basis="reference_eig", r=r, C_ref=C_projection)
    Phi, feature_meta = make_volterra_signature_features(
        x,
        half_lives=half_lives,
        level=level,
        standardize=True,
        pca_dim=pca_dim,
    )
    W, weight_meta = make_volterra_nonlinear_weights(
        state_dates,
        Phi,
        time_bandwidth=time_bandwidth,
        feature_bandwidth=feature_bandwidth,
        kernel=kernel,
    )
    outer = _outer_stack(Z)
    K = _weighted_moment(outer, W)
    C, ref_weights = _reference_from_weights(outer, W, reference)
    meta = {
        "target_type": "volterra_nonlinear",
        "score_shape": list(scores.shape),
        "filtered_score_shape": list(Z.shape),
        "reference_rule": reference,
        "reference_weights": ref_weights.tolist(),
        **filter_meta,
        "projection": projection_meta,
        "volterra_features": feature_meta,
        **weight_meta,
    }
    meta.update(kwargs)
    return TargetResult(K, C, state_dates, W, Z, meta, valid_index=valid_index, features=Phi)


def check_psd_symmetric(K: Array, tol: float = 1e-8) -> dict[str, Any]:
    """Return symmetry and PSD diagnostics for one matrix or a matrix path."""
    mats = np.asarray(K, dtype=float)
    if mats.ndim == 2:
        mats = mats[None, :, :]
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError("K must have shape (p,p) or (T,p,p)")
    sym_err = np.max(np.abs(mats - np.swapaxes(mats, 1, 2)), axis=(1, 2))
    eig_min = np.array([float(np.linalg.eigvalsh(_sym(A)).min()) for A in mats])
    return {
        "n_matrices": int(mats.shape[0]),
        "max_symmetry_error": float(np.max(sym_err)),
        "min_eigenvalue": float(np.min(eig_min)),
        "median_min_eigenvalue": float(np.median(eig_min)),
        "max_min_eigenvalue": float(np.max(eig_min)),
        "tol": float(tol),
        "ok": bool(np.max(sym_err) <= tol and np.min(eig_min) >= -tol),
    }


def soft_reference_geometry(C_ref: Array, rho: float | None = None, ridge_scale: float = 1e-8) -> dict[str, Any]:
    """Build ridge-soft whitening objects for a reference moment."""
    C = _sym(np.asarray(C_ref, dtype=float))
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C_ref must be square")
    p = C.shape[0]
    if rho is None:
        avg_variance = max(float(np.trace(C)) / max(p, 1), 1e-12)
        rho_val = float(max(ridge_scale, 0.0) * avg_variance + 1e-12)
    else:
        rho_val = float(rho)
    eigvals, eigvecs = np.linalg.eigh(C)
    d_vals = eigvals + rho_val
    if float(np.min(d_vals)) <= 0.0:
        raise ValueError("C_ref + rho I must be positive definite")
    D_rho = _sym(C + rho_val * np.eye(p))
    D_invsqrt = _sym((eigvecs * (1.0 / np.sqrt(d_vals))[None, :]) @ eigvecs.T)
    d_rho = float(np.sum(eigvals / d_vals))
    if not np.isfinite(d_rho) or d_rho <= 0.0:
        raise ValueError("soft effective dimension must be positive")
    return {
        "C_ref": C,
        "rho": rho_val,
        "D_rho": D_rho,
        "D_invsqrt": D_invsqrt,
        "d_rho": d_rho,
        "eigenvalues": eigvals,
    }


def relative_geometry_from_target(
    K_by_state: Array,
    C_ref: Array,
    rho: float | None = None,
    probes: Any = None,
    lambdas: Any = None,
) -> dict[str, Any]:
    """Compute ridge-soft tau and whitened operators from a target path."""
    del probes, lambdas
    K = _sym_last(np.asarray(K_by_state, dtype=float))
    ref = soft_reference_geometry(C_ref, rho=rho)
    D = np.asarray(ref["D_rho"], dtype=float)
    Dm = np.asarray(ref["D_invsqrt"], dtype=float)
    tau = np.empty(K.shape[0], dtype=float)
    for s, Ks in enumerate(K):
        solved = np.linalg.solve(D, _sym(Ks))
        tau[s] = float(np.trace(solved) / float(ref["d_rho"]))
    A = _sym_last(np.einsum("ij,tjk,kl->til", Dm, K, Dm, optimize=True))
    return {
        **ref,
        "K_by_state": K,
        "A_by_state": A,
        "tau_soft": tau,
    }


def effective_support(weights: Array) -> Array:
    """Normalized effective support for each row of a nonnegative weight matrix."""
    W = _row_normalize(np.asarray(weights, dtype=float))
    denom = np.sum(W**2, axis=1)
    ess = 1.0 / np.maximum(denom, 1e-300)
    return ess / max(W.shape[1], 1)
