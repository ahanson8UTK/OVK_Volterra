"""Route-comparison rotation diagnostics for monetary-policy memory targets."""
from __future__ import annotations

import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_series_targets import soft_reference_geometry


ROUTE_KEYS = {
    "diagonal_old": "D",
    "diagonal": "D",
    "hac_filtered": "H",
    "hac": "H",
    "hilbert_volterra": "V",
    "volterra": "V",
}
ROUTE_LABELS = {"D": "Diagonal", "H": "HAC", "V": "Hilbert-Volterra"}
PAIR_DEFS = [
    ("D_H", "Diagonal vs HAC", "D", "H"),
    ("H_V", "HAC vs Hilbert-Volterra", "H", "V"),
    ("D_V", "Diagonal vs Hilbert-Volterra", "D", "V"),
]
PROBE_DISPLAY = {
    "full_soft": "Full soft",
    "macro": "Macro",
    "financial": "Financial",
    "short_horizons": "Short horizons",
    "medium_horizons": "Medium horizons",
    "long_horizons": "Long horizons",
}


@dataclass(frozen=True)
class RouteRotationConfig:
    targets_dir: Path
    comparison_dir: Path
    routes: tuple[str, str, str] = (
        "diagonal_old",
        "hac_filtered_L12",
        "hilbert_volterra_L12_gamma005_memory_3_12_36",
    )
    rotation_reference: str = "pooled"
    lambda_min: float = 1e-2
    lambda_max: float = 1e2
    lambda_count: int = 41
    min_anisotropy: float = 0.05
    alignment_tol: float = 1e-8
    denominator_tol: float = 1e-12


@dataclass
class RouteData:
    key: str
    label: str
    folder: Path
    K_by_state: np.ndarray
    C_ref: np.ndarray
    dates: pd.Series
    coordinate_map: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class Probe:
    name: str
    display: str
    Q: np.ndarray
    eigvecs: np.ndarray
    sqrt_eigvals: np.ndarray
    support_coordinates: list[int]
    available: bool
    diagnostics: dict[str, Any]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _sym_last(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _safe_eigh(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(_sym(np.asarray(A, dtype=float)))
    return vals, vecs


def _psd_sqrt(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, dict[str, Any]]:
    vals, vecs = _safe_eigh(A)
    clipped = np.where(vals < 0.0, np.maximum(vals, 0.0), vals)
    sqrt = _sym((vecs * np.sqrt(clipped)[None, :]) @ vecs.T)
    return sqrt, {
        "min_eigenvalue": float(np.min(vals)),
        "max_eigenvalue": float(np.max(vals)),
        "negative_eigenvalues_below_tolerance": int(np.sum(vals < -tol)),
    }


def _route_key_from_folder(folder: str | Path) -> str:
    name = Path(folder).name.lower()
    for token, key in ROUTE_KEYS.items():
        if name == token or name.startswith(token):
            return key
    raise ValueError(f"cannot infer route key from folder name: {folder}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_coordinate_map(metadata: dict[str, Any], p: int) -> pd.DataFrame:
    rows = metadata.get("coordinate_map", [])
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "coordinate": np.arange(p, dtype=int),
            "outcome": [f"coord_{j}" for j in range(p)],
            "horizon_months": np.zeros(p, dtype=int),
        }
    )


def load_route(folder: Path) -> RouteData:
    key = _route_key_from_folder(folder)
    metadata = _read_json(folder / "metadata.json")
    k_path = folder / "K_by_state.npy"
    c_path = folder / "C_ref.npy"
    if not k_path.exists() or not c_path.exists():
        raise FileNotFoundError(
            f"{folder} is missing K_by_state.npy or C_ref.npy. "
            "Rerun scripts/run_monetary_lp_memory_targets.py so route operators are serialized."
        )
    K = _sym_last(np.load(k_path))
    C = _sym(np.load(c_path))
    dates_path = folder / "state_dates.csv"
    if dates_path.exists():
        dates_df = pd.read_csv(dates_path)
        date_col = "date" if "date" in dates_df.columns else dates_df.columns[0]
        dates = pd.to_datetime(dates_df[date_col], errors="coerce").reset_index(drop=True)
    else:
        tau_path = folder / "tau_soft.csv"
        if not tau_path.exists():
            raise FileNotFoundError(f"{folder} is missing both state_dates.csv and tau_soft.csv")
        dates = pd.to_datetime(pd.read_csv(tau_path)["date"], errors="coerce").reset_index(drop=True)
    map_path = folder / "coordinate_map.csv"
    coordinate_map = pd.read_csv(map_path) if map_path.exists() else _metadata_coordinate_map(metadata, C.shape[0])
    return RouteData(
        key=key,
        label=ROUTE_LABELS[key],
        folder=folder,
        K_by_state=K,
        C_ref=C,
        dates=dates,
        coordinate_map=coordinate_map,
        metadata=metadata,
    )


def load_routes(targets_dir: Path, routes: tuple[str, ...]) -> dict[str, RouteData]:
    out: dict[str, RouteData] = {}
    for route in routes:
        folder = targets_dir / route
        data = load_route(folder)
        out[data.key] = data
    required = {"D", "H", "V"}
    missing = sorted(required - set(out))
    if missing:
        raise ValueError(f"missing required routes: {missing}")
    return out


def align_common_dates(routes: dict[str, RouteData]) -> tuple[pd.Series, dict[str, np.ndarray]]:
    date_sets: list[set[str]] = []
    route_date_strings: dict[str, list[str]] = {}
    for key, route in routes.items():
        strings = pd.to_datetime(route.dates).dt.strftime("%Y-%m-%d").tolist()
        route_date_strings[key] = strings
        date_sets.append(set(strings))
    common = sorted(set.intersection(*date_sets))
    if not common:
        raise ValueError("routes do not share any state dates")
    indices: dict[str, np.ndarray] = {}
    for key, strings in route_date_strings.items():
        lookup = {date: i for i, date in enumerate(strings)}
        indices[key] = np.asarray([lookup[date] for date in common], dtype=int)
    return pd.Series(pd.to_datetime(common)), indices


def _rho_from_route(route: RouteData) -> float | None:
    for key in ["rho", "score_gram_rho"]:
        value = route.metadata.get(key)
        if value is not None:
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_float) and value_float > 0.0:
                return value_float
    return None


def common_reference(
    routes: dict[str, RouteData],
    mode: str = "pooled",
) -> tuple[np.ndarray, float, dict[str, Any]]:
    mode_key = mode.lower().replace("-", "_")
    if mode_key == "pooled":
        C_star = _sym(sum(route.C_ref for route in routes.values()) / len(routes))
    elif mode_key in {"diagonal", "diagonal_old"}:
        C_star = routes["D"].C_ref
    elif mode_key in {"hac", "hac_filtered"}:
        C_star = routes["H"].C_ref
    elif mode_key in {"hilbert_volterra", "hilbert", "volterra"}:
        C_star = routes["V"].C_ref
    else:
        raise ValueError(f"unknown rotation reference: {mode}")
    rho_values = [rho for rho in (_rho_from_route(route) for route in routes.values()) if rho is not None]
    rho_star = float(np.median(rho_values)) if rho_values else float(soft_reference_geometry(C_star)["rho"])
    c_eig = np.linalg.eigvalsh(_sym(C_star))
    d_eig = np.linalg.eigvalsh(_sym(C_star + rho_star * np.eye(C_star.shape[0])))
    meta = {
        "common_reference_mode": mode_key,
        "rho_values_from_routes": rho_values,
        "rho_star": rho_star,
        "C_star_min_eigenvalue": float(np.min(c_eig)),
        "C_star_max_eigenvalue": float(np.max(c_eig)),
        "D_star_min_eigenvalue": float(np.min(d_eig)),
        "D_star_max_eigenvalue": float(np.max(d_eig)),
    }
    return _sym(C_star), rho_star, meta


def compute_common_geometry(
    routes: dict[str, RouteData],
    date_indices: dict[str, np.ndarray],
    reference_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, float, dict[str, Any]]:
    C_star, rho_star, ref_meta = common_reference(routes, reference_mode)
    ref = soft_reference_geometry(C_star, rho=rho_star)
    W_star = np.asarray(ref["D_invsqrt"], dtype=float)
    S_star = _sym(W_star @ C_star @ W_star)
    d_star = float(np.trace(S_star))
    if not np.isfinite(d_star) or d_star <= 0.0:
        raise ValueError("common soft reference has nonpositive effective trace")
    shapes: dict[str, np.ndarray] = {}
    tau_common: dict[str, np.ndarray] = {}
    invalid_counts: dict[str, int] = {}
    for key, route in routes.items():
        K = route.K_by_state[date_indices[key]]
        A = _sym_last(np.einsum("ij,sjk,kl->sil", W_star, K, W_star, optimize=True))
        tau = np.trace(A, axis1=1, axis2=2) / d_star
        valid = np.isfinite(tau) & (tau > 0.0)
        Abar = np.full_like(A, np.nan)
        Abar[valid] = A[valid] / tau[valid, None, None]
        shapes[key] = _sym_last(Abar)
        tau_common[key] = tau
        invalid_counts[key] = int(np.sum(~valid))
    s_eig = np.linalg.eigvalsh(S_star)
    ref_meta.update(
        {
            "d_star": d_star,
            "S_star_min_eigenvalue": float(np.min(s_eig)),
            "S_star_max_eigenvalue": float(np.max(s_eig)),
            "invalid_tau_counts": invalid_counts,
        }
    )
    return shapes, tau_common, W_star, S_star, d_star, ref_meta


def _normalise_probe(Q_raw: np.ndarray, support: list[int], name: str, display: str, tol: float) -> Probe:
    Q_raw = _sym(np.asarray(Q_raw, dtype=float))
    trace_raw = float(np.trace(Q_raw))
    if not np.isfinite(trace_raw) or trace_raw <= tol:
        diag = {
            "name": name,
            "display": display,
            "available": False,
            "trace_raw": trace_raw,
            "reason": "nonpositive raw trace",
            "support_coordinates": support,
        }
        p = Q_raw.shape[0]
        return Probe(name, display, np.zeros((p, p)), np.zeros((p, 0)), np.zeros(0), support, False, diag)
    Q = _sym(Q_raw / trace_raw)
    vals, vecs = _safe_eigh(Q)
    pos = vals > max(tol, tol * max(float(np.max(vals)), 1.0))
    rank = int(np.sum(pos))
    eigvecs = vecs[:, pos]
    sqrt_eigvals = np.sqrt(np.maximum(vals[pos], 0.0))
    diagnostics = {
        "name": name,
        "display": display,
        "available": True,
        "rank_estimate": rank,
        "trace": float(np.trace(Q)),
        "trace_raw": trace_raw,
        "min_eigenvalue": float(np.min(vals)),
        "max_eigenvalue": float(np.max(vals)),
        "support_coordinates": support,
        "n_support_coordinates": int(len(support)),
        "trace_normalization_error": float(abs(np.trace(Q) - 1.0)),
    }
    return Probe(name, display, Q, eigvecs, sqrt_eigvals, support, True, diagnostics)


def _support_from_map(coordinate_map: pd.DataFrame, p: int, outcomes: set[str] | None = None, horizons: set[int] | None = None) -> list[int]:
    df = coordinate_map.copy()
    if "coordinate" not in df.columns:
        df["coordinate"] = np.arange(len(df), dtype=int)
    if "horizon_months" not in df.columns and "horizon" in df.columns:
        df["horizon_months"] = df["horizon"]
    mask = np.ones(len(df), dtype=bool)
    if outcomes is not None and "outcome" in df.columns:
        mask &= df["outcome"].astype(str).isin(outcomes).to_numpy()
    if horizons is not None and "horizon_months" in df.columns:
        mask &= pd.to_numeric(df["horizon_months"], errors="coerce").isin(horizons).to_numpy()
    coords = pd.to_numeric(df.loc[mask, "coordinate"], errors="coerce").dropna().astype(int).tolist()
    return sorted({coord for coord in coords if 0 <= coord < p})


def build_trace_class_probes(S_star: np.ndarray, coordinate_map: pd.DataFrame, tol: float = 1e-12) -> dict[str, Probe]:
    p = S_star.shape[0]
    S_sqrt, sqrt_diag = _psd_sqrt(S_star, tol=tol)
    probes: dict[str, Probe] = {}
    probes["full_soft"] = _normalise_probe(S_star, list(range(p)), "full_soft", PROBE_DISPLAY["full_soft"], tol)
    definitions = [
        ("macro", PROBE_DISPLAY["macro"], {"IP", "CPI", "Unemployment"}, None),
        ("financial", PROBE_DISPLAY["financial"], {"2Y yield", "BAA-10Y spread"}, None),
        ("short_horizons", PROBE_DISPLAY["short_horizons"], None, set(range(0, 4))),
        ("medium_horizons", PROBE_DISPLAY["medium_horizons"], None, set(range(4, 13))),
        ("long_horizons", PROBE_DISPLAY["long_horizons"], None, set(range(13, 25))),
    ]
    for name, display, outcomes, horizons in definitions:
        support = _support_from_map(coordinate_map, p, outcomes=outcomes, horizons=horizons)
        if support:
            Q_raw = _sym(S_sqrt[:, support] @ S_sqrt[:, support].T)
        else:
            Q_raw = np.zeros((p, p), dtype=float)
        probes[name] = _normalise_probe(Q_raw, support, name, display, tol)
    probes["full_soft"].diagnostics["S_star_sqrt_diagnostics"] = sqrt_diag
    return probes


def _eigen_y_grid(A: np.ndarray, lambdas: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vals, vecs = _safe_eigh(A)
    large_negative = vals < -tol
    vals_clip = np.maximum(vals, 0.0)
    y = vals_clip[None, :] / (lambdas[:, None] + vals_clip[None, :])
    return vals_clip, vecs, {
        "min_raw_eigenvalue": float(np.min(vals)),
        "max_raw_eigenvalue": float(np.max(vals)),
        "large_negative_eigenvalues": int(np.sum(large_negative)),
        "y_grid": y,
    }


def yosida_alignment_for_pair(
    A_left: np.ndarray,
    A_right: np.ndarray,
    Q: np.ndarray,
    lambdas: np.ndarray,
    denominator_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Omega(lambda), denominators, and validity for one route pair."""
    q_vals, q_vecs = _safe_eigh(Q)
    pos = q_vals > max(denominator_tol, denominator_tol * max(float(np.max(q_vals)), 1.0))
    if not np.any(pos):
        nan = np.full(len(lambdas), np.nan)
        return nan, np.zeros(len(lambdas)), np.zeros(len(lambdas), dtype=bool)
    sqrt_q = np.sqrt(np.maximum(q_vals[pos], 0.0))
    Uq = q_vecs[:, pos]
    _, Vl, left_meta = _eigen_y_grid(A_left, lambdas)
    _, Vr, right_meta = _eigen_y_grid(A_right, lambdas)
    Yl = np.asarray(left_meta["y_grid"], dtype=float)
    Yr = np.asarray(right_meta["y_grid"], dtype=float)
    Fl = sqrt_q[:, None] * (Uq.T @ Vl)
    Fr = sqrt_q[:, None] * (Uq.T @ Vr)
    E_lr = (Fl.T @ Fr) ** 2
    E_ll = (Fl.T @ Fl) ** 2
    E_rr = (Fr.T @ Fr) ** 2
    numerator = np.einsum("li,ij,lj->l", Yl, E_lr, Yr, optimize=True)
    left_norm = np.einsum("li,ij,lj->l", Yl, E_ll, Yl, optimize=True)
    right_norm = np.einsum("li,ij,lj->l", Yr, E_rr, Yr, optimize=True)
    denom = np.sqrt(np.maximum(left_norm * right_norm, 0.0))
    valid = np.isfinite(numerator) & np.isfinite(denom) & (denom > denominator_tol)
    omega = np.full(len(lambdas), np.nan)
    omega[valid] = numerator[valid] / denom[valid]
    return omega, denom, valid


def _clamp_alignment(omega: float, tol: float) -> tuple[float, bool, bool]:
    if not np.isfinite(omega):
        return float("nan"), False, False
    if omega < 0.0 and omega >= -tol:
        return 0.0, True, False
    if omega > 1.0 and omega <= 1.0 + tol:
        return 1.0, True, False
    if omega < -tol or omega > 1.0 + tol:
        return float(omega), False, True
    return float(omega), False, False


def compute_yosida_alignment(
    shapes: dict[str, np.ndarray],
    dates: pd.Series,
    probes: dict[str, Probe],
    lambdas: np.ndarray,
    alignment_tol: float = 1e-8,
    denominator_tol: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    long_rows: list[dict[str, Any]] = []
    by_date_rows: list[dict[str, Any]] = []
    route_keys = ["D", "H", "V"]
    n_dates = len(dates)
    n_large_violations = 0
    n_tiny_clamps = 0
    large_negative_eigs = 0
    for s in range(n_dates):
        date_string = pd.to_datetime(dates.iloc[s]).strftime("%Y-%m-%d")
        eig_meta: dict[str, dict[str, Any]] = {}
        eigvecs: dict[str, np.ndarray] = {}
        y_grids: dict[str, np.ndarray] = {}
        route_valid: dict[str, bool] = {}
        for key in route_keys:
            A = shapes[key][s]
            route_valid[key] = np.isfinite(A).all()
            if route_valid[key]:
                _, eigvec, meta = _eigen_y_grid(A, lambdas)
                eigvecs[key] = eigvec
                y_grids[key] = np.asarray(meta["y_grid"], dtype=float)
                eig_meta[key] = meta
                large_negative_eigs += int(meta["large_negative_eigenvalues"])
        for probe in probes.values():
            if not probe.available:
                for pair_key, pair_label, left, right in PAIR_DEFS:
                    for lam in lambdas:
                        long_rows.append(
                            {
                                "date": date_string,
                                "pair_key": pair_key,
                                "pair_label": pair_label,
                                "route_left": left,
                                "route_right": right,
                                "probe": probe.display,
                                "lambda": float(lam),
                                "omega": np.nan,
                                "denominator": 0.0,
                                "valid": False,
                            }
                        )
                    by_date_rows.append(
                        {
                            "date": date_string,
                            "pair_key": pair_key,
                            "pair_label": pair_label,
                            "probe": probe.display,
                            "omega_bar": np.nan,
                            "rotation_distance": np.nan,
                            "angle_degrees": np.nan,
                            "n_lambda_valid": 0,
                            "valid": False,
                        }
                    )
                continue
            F: dict[str, np.ndarray] = {}
            for key in route_keys:
                if route_valid[key]:
                    F[key] = probe.sqrt_eigvals[:, None] * (probe.eigvecs.T @ eigvecs[key])
            self_E: dict[str, np.ndarray] = {}
            for key in F:
                self_E[key] = (F[key].T @ F[key]) ** 2
            for pair_key, pair_label, left, right in PAIR_DEFS:
                if not (route_valid.get(left, False) and route_valid.get(right, False)):
                    omega = np.full(len(lambdas), np.nan)
                    denom = np.zeros(len(lambdas), dtype=float)
                    valid = np.zeros(len(lambdas), dtype=bool)
                else:
                    E_lr = (F[left].T @ F[right]) ** 2
                    Yl = y_grids[left]
                    Yr = y_grids[right]
                    numerator = np.einsum("li,ij,lj->l", Yl, E_lr, Yr, optimize=True)
                    left_norm = np.einsum("li,ij,lj->l", Yl, self_E[left], Yl, optimize=True)
                    right_norm = np.einsum("li,ij,lj->l", Yr, self_E[right], Yr, optimize=True)
                    denom = np.sqrt(np.maximum(left_norm * right_norm, 0.0))
                    valid = np.isfinite(numerator) & np.isfinite(denom) & (denom > denominator_tol)
                    omega = np.full(len(lambdas), np.nan)
                    omega[valid] = numerator[valid] / denom[valid]
                clamped = []
                for lam, omg, den, ok in zip(lambdas, omega, denom, valid):
                    omega_value, tiny_clamp, large_violation = _clamp_alignment(float(omg), alignment_tol)
                    n_tiny_clamps += int(tiny_clamp)
                    n_large_violations += int(large_violation)
                    clamped.append(omega_value)
                    long_rows.append(
                        {
                            "date": date_string,
                            "pair_key": pair_key,
                            "pair_label": pair_label,
                            "route_left": left,
                            "route_right": right,
                            "probe": probe.display,
                            "lambda": float(lam),
                            "omega": omega_value,
                            "denominator": float(den),
                            "valid": bool(ok and not large_violation),
                        }
                    )
                clamped_arr = np.asarray(clamped, dtype=float)
                lambda_valid = np.asarray([row["valid"] for row in long_rows[-len(lambdas) :]], dtype=bool)
                n_valid = int(np.sum(lambda_valid & np.isfinite(clamped_arr)))
                omega_bar = float(np.nanmean(clamped_arr[lambda_valid])) if n_valid else np.nan
                omega_for_angle = float(np.clip(omega_bar, -1.0, 1.0)) if np.isfinite(omega_bar) else np.nan
                by_date_rows.append(
                    {
                        "date": date_string,
                        "pair_key": pair_key,
                        "pair_label": pair_label,
                        "probe": probe.display,
                        "omega_bar": omega_bar,
                        "rotation_distance": float(1.0 - omega_bar) if np.isfinite(omega_bar) else np.nan,
                        "angle_degrees": float(math.degrees(math.acos(omega_for_angle))) if np.isfinite(omega_for_angle) else np.nan,
                        "n_lambda_valid": n_valid,
                        "valid": bool(n_valid > 0),
                    }
                )
    metadata = {
        "large_alignment_violations": n_large_violations,
        "tiny_alignment_clamps": n_tiny_clamps,
        "large_negative_shape_eigenvalues": large_negative_eigs,
    }
    return pd.DataFrame(long_rows), pd.DataFrame(by_date_rows), metadata


def summarize_yosida_alignment(by_date: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    valid = by_date[by_date["valid"].astype(bool)].copy()
    for (pair_key, pair_label, probe), sub in valid.groupby(["pair_key", "pair_label", "probe"], dropna=False):
        rotation = pd.to_numeric(sub["rotation_distance"], errors="coerce")
        angle = pd.to_numeric(sub["angle_degrees"], errors="coerce")
        if rotation.dropna().empty:
            continue
        max_idx = rotation.idxmax()
        rows.append(
            {
                "pair_key": pair_key,
                "pair_label": pair_label,
                "probe": probe,
                "mean_rotation_distance": float(rotation.mean()),
                "median_rotation_distance": float(rotation.median()),
                "p90_rotation_distance": float(rotation.quantile(0.90)),
                "max_rotation_distance": float(rotation.max()),
                "date_max_rotation": str(sub.loc[max_idx, "date"]),
                "mean_angle_degrees": float(angle.mean()),
                "median_angle_degrees": float(angle.median()),
                "p90_angle_degrees": float(angle.quantile(0.90)),
                "max_angle_degrees": float(angle.max()),
                "n_dates_valid": int(rotation.notna().sum()),
            }
        )
    summary = pd.DataFrame(rows)
    mean_compact = _compact_table(summary, "mean_rotation_distance")
    p90_compact = _compact_table(summary, "p90_rotation_distance")
    return summary, mean_compact, p90_compact


def _compact_table(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ordered_pairs = [label for _, label, _, _ in PAIR_DEFS]
    ordered_probes = list(PROBE_DISPLAY.values())
    table = pd.DataFrame(index=ordered_pairs, columns=ordered_probes, dtype=float)
    for row in summary.itertuples(index=False):
        table.loc[row.pair_label, row.probe] = float(getattr(row, value_col))
    table.index.name = "pair_label"
    return table


def compute_commutator_rotation(
    shapes: dict[str, np.ndarray],
    dates: pd.Series,
    S_star: np.ndarray,
    min_anisotropy: float = 0.05,
    denominator_tol: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    I_bar = _sym(S_star)
    norm_ref = max(float(np.linalg.norm(I_bar, ord="fro")), denominator_tol)
    aniso_rows = []
    anisotropy: dict[str, np.ndarray] = {}
    for key in ["D", "H", "V"]:
        vals = np.empty(len(dates), dtype=float)
        for s in range(len(dates)):
            A = shapes[key][s]
            vals[s] = float(np.linalg.norm(A - I_bar, ord="fro") / norm_ref) if np.isfinite(A).all() else np.nan
            aniso_rows.append(
                {
                    "date": pd.to_datetime(dates.iloc[s]).strftime("%Y-%m-%d"),
                    "route": key,
                    "route_label": ROUTE_LABELS[key],
                    "anisotropy": vals[s],
                }
            )
        anisotropy[key] = vals

    rows = []
    for s in range(len(dates)):
        date_string = pd.to_datetime(dates.iloc[s]).strftime("%Y-%m-%d")
        for pair_key, pair_label, left, right in PAIR_DEFS:
            A = shapes[left][s]
            B = shapes[right][s]
            if not (np.isfinite(A).all() and np.isfinite(B).all()):
                comm_norm = denom = index = np.nan
                valid = False
            else:
                comm = A @ B - B @ A
                comm_norm = float(np.linalg.norm(comm, ord="fro"))
                denom = float(np.linalg.norm(A - I_bar, ord="fro") * np.linalg.norm(B - I_bar, ord="fro"))
                valid = bool(np.isfinite(denom) and denom > denominator_tol)
                index = float(comm_norm / denom) if valid else np.nan
            min_pair = float(np.nanmin([anisotropy[left][s], anisotropy[right][s]]))
            interpretable = bool(
                valid
                and np.isfinite(index)
                and anisotropy[left][s] >= min_anisotropy
                and anisotropy[right][s] >= min_anisotropy
            )
            rows.append(
                {
                    "date": date_string,
                    "pair_key": pair_key,
                    "pair_label": pair_label,
                    "route_left": left,
                    "route_right": right,
                    "commutator_hs_norm": comm_norm,
                    "denom_hs": denom,
                    "commutator_index": index,
                    "anisotropy_left": float(anisotropy[left][s]),
                    "anisotropy_right": float(anisotropy[right][s]),
                    "min_pair_anisotropy": min_pair,
                    "interpretable": interpretable,
                    "valid": valid,
                }
            )
    by_date = pd.DataFrame(rows)
    summary_rows = []
    top_rows = []
    for pair_key, pair_label, left, right in [(p[0], p[1], p[2], p[3]) for p in PAIR_DEFS]:
        sub = by_date[by_date["pair_key"].eq(pair_key)].copy()
        valid = sub[sub["valid"].astype(bool)]
        interpretable = sub[sub["interpretable"].astype(bool)]
        metric = pd.to_numeric(interpretable["commutator_index"], errors="coerce")
        source = interpretable if not metric.dropna().empty else valid
        metric = pd.to_numeric(source["commutator_index"], errors="coerce")
        if metric.dropna().empty:
            max_val = np.nan
            date_max = ""
        else:
            max_idx = metric.idxmax()
            max_val = float(metric.max())
            date_max = str(source.loc[max_idx, "date"])
        summary_rows.append(
            {
                "pair_key": pair_key,
                "pair_label": pair_label,
                "mean_commutator_index": float(metric.mean()) if not metric.dropna().empty else np.nan,
                "median_commutator_index": float(metric.median()) if not metric.dropna().empty else np.nan,
                "p90_commutator_index": float(metric.quantile(0.90)) if not metric.dropna().empty else np.nan,
                "max_commutator_index": max_val,
                "date_max_commutator": date_max,
                "mean_anisotropy_left": float(pd.to_numeric(sub["anisotropy_left"], errors="coerce").mean()),
                "mean_anisotropy_right": float(pd.to_numeric(sub["anisotropy_right"], errors="coerce").mean()),
                "n_dates_valid": int(valid.shape[0]),
                "n_dates_interpretable": int(interpretable.shape[0]),
                "share_dates_interpretable": float(interpretable.shape[0] / max(sub.shape[0], 1)),
            }
        )
        for row in interpretable.sort_values("commutator_index", ascending=False).head(10).itertuples(index=False):
            top_rows.append(row._asdict())
    return by_date, pd.DataFrame(summary_rows), pd.DataFrame(top_rows), pd.DataFrame(aniso_rows)


def _plot_full_soft_timeseries(by_date: pd.DataFrame, path: Path) -> None:
    sub = by_date[by_date["probe"].eq(PROBE_DISPLAY["full_soft"]) & by_date["valid"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    for pair_key, pair_label, _, _ in PAIR_DEFS:
        part = sub[sub["pair_key"].eq(pair_key)]
        ax.plot(pd.to_datetime(part["date"]), part["rotation_distance"], label=pair_label, linewidth=1.25)
    ax.set_title("Probe-weighted Yosida route rotation: full soft probe")
    ax.set_ylabel("1 - mean Yosida alignment")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmap(table: pd.DataFrame, path: Path, title: str) -> None:
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 4.1), constrained_layout=True)
    image = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]), table.columns, rotation=28, ha="right")
    ax.set_yticks(np.arange(table.shape[0]), table.index)
    ax.set_title(title)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                scale = val / max(float(np.nanmax(values)), 1e-12)
                color = "white" if scale < 0.35 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("1 - mean alignment")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _selected_dates(
    targets_dir: Path,
    routes: dict[str, RouteData],
    by_date: pd.DataFrame,
    common_dates: pd.Series,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    common_set = set(pd.to_datetime(common_dates).dt.strftime("%Y-%m-%d"))
    for key in ["D", "H", "V"]:
        tau_path = routes[key].folder / "tau_soft.csv"
        if tau_path.exists():
            tau = pd.read_csv(tau_path)
            if "tau_soft" in tau and "date" in tau and len(tau):
                top = tau.sort_values("tau_soft", ascending=False).iloc[0]
                date = pd.to_datetime(top["date"]).strftime("%Y-%m-%d")
                if date in common_set:
                    out.append((date, f"max tau {ROUTE_LABELS[key]}"))
    full = by_date[by_date["probe"].eq(PROBE_DISPLAY["full_soft"]) & by_date["valid"].astype(bool)]
    for pair_key, pair_label, _, _ in PAIR_DEFS:
        sub = full[full["pair_key"].eq(pair_key)]
        if not sub.empty:
            row = sub.sort_values("rotation_distance", ascending=False).iloc[0]
            out.append((str(row["date"]), f"max rotation {pair_label}"))
    dedup: dict[str, str] = {}
    for date, reason in out:
        dedup.setdefault(date, reason)
        if reason not in dedup[date]:
            dedup[date] += f"; {reason}"
    return sorted(dedup.items())


def _plot_selected_alignment(
    selected: list[tuple[str, str]],
    by_date: pd.DataFrame,
    path: Path,
    csv_path: Path,
) -> None:
    rows = []
    n = max(1, len(selected))
    ncols = n if n <= 3 else 2 if n <= 4 else 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.35 * ncols + 0.5, 3.15 * nrows), squeeze=False, constrained_layout=True)
    for ax, (date, reason) in zip(axes.ravel(), selected):
        mat = np.eye(3)
        labels = ["D", "H", "V"]
        full = by_date[
            by_date["date"].eq(date)
            & by_date["probe"].eq(PROBE_DISPLAY["full_soft"])
            & by_date["valid"].astype(bool)
        ]
        for pair_key, pair_label, left, right in PAIR_DEFS:
            sub = full[full["pair_key"].eq(pair_key)]
            val = float(sub["omega_bar"].iloc[0]) if len(sub) else np.nan
            i = labels.index(left)
            j = labels.index(right)
            mat[i, j] = val
            mat[j, i] = val
        for i, row_key in enumerate(labels):
            for j, col_key in enumerate(labels):
                rows.append(
                    {
                        "date": date,
                        "selection_reason": reason,
                        "route_row": row_key,
                        "route_column": col_key,
                        "omega_bar": float(mat[i, j]),
                    }
                )
        image = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(3), [ROUTE_LABELS[k] for k in labels], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(np.arange(3), [ROUTE_LABELS[k] for k in labels], fontsize=8)
        short_reason = (
            reason.replace("max tau ", "tau ")
            .replace("max rotation ", "rot ")
            .replace("Diagonal vs Hilbert-Volterra", "D-V")
            .replace("Diagonal vs HAC", "D-H")
            .replace("HAC vs Hilbert-Volterra", "H-V")
            .replace("Diagonal", "D")
            .replace("Hilbert-Volterra", "V")
        )
        ax.set_title(pd.to_datetime(date).strftime("%Y-%m") + "\n" + short_reason, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] < 0.45 else "black", fontsize=8)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label("mean alignment", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _plot_commutator_timeseries(by_date: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    for pair_key, pair_label, _, _ in PAIR_DEFS:
        sub = by_date[by_date["pair_key"].eq(pair_key)].copy()
        dates = pd.to_datetime(sub["date"])
        interp = sub["interpretable"].astype(bool)
        ax.plot(dates[interp], sub.loc[interp, "commutator_index"], label=pair_label, linewidth=1.2)
        ax.scatter(dates[~interp], sub.loc[~interp, "commutator_index"], s=10, alpha=0.18)
    ax.set_title("Commutator route rotation after removing soft scale")
    ax.set_ylabel("normalized commutator index")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_anisotropy(anisotropy: pd.DataFrame, min_anisotropy: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    for key, label in ROUTE_LABELS.items():
        sub = anisotropy[anisotropy["route"].eq(key)]
        ax.plot(pd.to_datetime(sub["date"]), sub["anisotropy"], label=label, linewidth=1.25)
    ax.axhline(min_anisotropy, color="black", linewidth=0.9, linestyle="--", label="interpretability threshold")
    ax.set_title("Route anisotropy relative to common soft-reference shape")
    ax.set_ylabel("anisotropy")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_commutator_scatter(by_date: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.1))
    for pair_key, pair_label, _, _ in PAIR_DEFS:
        sub = by_date[by_date["pair_key"].eq(pair_key)]
        ax.scatter(sub["min_pair_anisotropy"], sub["commutator_index"], s=18, alpha=0.65, label=pair_label)
    ax.set_xlabel("minimum pair anisotropy")
    ax.set_ylabel("normalized commutator index")
    ax.set_title("Commutator rotation versus route anisotropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_commutator_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(summary["pair_label"], summary["p90_commutator_index"], color="#4c78a8")
    ax.set_ylabel("p90 commutator index")
    ax.set_title("Commutator route rotation summary")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    work = df.copy()
    if work.index.name is not None:
        work = work.reset_index()
    rows = [[str(c) for c in work.columns], ["---"] * len(work.columns)]
    for item in work.itertuples(index=False, name=None):
        row = []
        for value in item:
            if isinstance(value, float):
                row.append("" if not np.isfinite(value) else f"{value:.3f}")
            else:
                row.append(str(value))
        rows.append(row)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def append_rotation_summary(
    summary_path: Path,
    compact: pd.DataFrame,
    compact_p90: pd.DataFrame,
    comm_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "# Monetary LP Memory-Target OVK Summary\n"
    marker = "\n## Route rotation diagnostics\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    section = f"""
## Route rotation diagnostics

The `tau_soft` comparison reports route scale. These diagnostics compare route-specific relative-moment shapes after placing all three routes in a common ridge-soft geometry and removing each route's common soft scale.

The probed Yosida alignment uses

`Omega = <Q^(1/2) Y_r Q^(1/2), Q^(1/2) Y_r' Q^(1/2)>_HS / (norms)`

and reports `1 - average_lambda Omega`. The full soft probe is the full-coordinate soft comparison; the macro, financial, and horizon probes show whether route rotation is concentrated in economically meaningful blocks.

Mean Yosida rotation:

{_markdown_table(compact)}

P90 Yosida rotation:

{_markdown_table(compact_p90)}

The commutator diagnostic uses `[Abar_r, Abar_r'] = Abar_r Abar_r' - Abar_r' Abar_r` after scale removal. It separates directional noncommutativity from pure eigenvalue/shape differences and should be interpreted only when both routes have nontrivial anisotropy relative to the common soft-reference shape.

Commutator summary:

{_markdown_table(comm_summary[["pair_label", "p90_commutator_index", "max_commutator_index", "date_max_commutator", "share_dates_interpretable"]])}

Reference mode: `{metadata["common_reference_mode"]}`; common dates: {metadata["number_of_common_dates"]}; `rho_star={metadata["rho_star"]:.3e}`.

These are diagnostic shape/rotation comparisons, not time-varying causal effects and not conventional HAC standard errors.
"""
    summary_path.write_text(text.rstrip() + "\n" + section.lstrip(), encoding="utf-8")


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ["numpy", "pandas", "matplotlib"]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run_rotation_diagnostics(config: RouteRotationConfig) -> dict[str, Any]:
    config.comparison_dir.mkdir(parents=True, exist_ok=True)
    routes = load_routes(config.targets_dir, config.routes)
    common_dates, date_indices = align_common_dates(routes)
    shapes, tau_common, _, S_star, d_star, ref_meta = compute_common_geometry(routes, date_indices, config.rotation_reference)
    coordinate_map = routes["D"].coordinate_map
    probes = build_trace_class_probes(S_star, coordinate_map, tol=config.denominator_tol)
    lambdas = np.logspace(math.log10(config.lambda_min), math.log10(config.lambda_max), int(config.lambda_count))

    long_df, by_date_df, align_meta = compute_yosida_alignment(
        shapes,
        common_dates,
        probes,
        lambdas,
        alignment_tol=config.alignment_tol,
        denominator_tol=config.denominator_tol,
    )
    align_summary, compact, compact_p90 = summarize_yosida_alignment(by_date_df)

    comm_by_date, comm_summary, comm_top, anisotropy = compute_commutator_rotation(
        shapes,
        common_dates,
        S_star,
        min_anisotropy=config.min_anisotropy,
        denominator_tol=config.denominator_tol,
    )

    comp = config.comparison_dir
    long_df.to_csv(comp / "probed_yosida_alignment_long.csv", index=False)
    by_date_df.to_csv(comp / "probed_yosida_alignment_by_date.csv", index=False)
    align_summary.to_csv(comp / "probed_yosida_alignment_summary.csv", index=False)
    compact.to_csv(comp / "probed_yosida_rotation_compact_table.csv")
    compact_p90.to_csv(comp / "probed_yosida_rotation_compact_table_p90.csv")
    comm_by_date.to_csv(comp / "commutator_rotation_by_date.csv", index=False)
    comm_summary.to_csv(comp / "commutator_rotation_summary.csv", index=False)
    comm_top.to_csv(comp / "commutator_rotation_top_dates.csv", index=False)
    anisotropy.to_csv(comp / "route_anisotropy_by_date.csv", index=False)

    _plot_full_soft_timeseries(by_date_df, comp / "probed_yosida_rotation_full_soft_timeseries.png")
    _plot_heatmap(compact, comp / "probed_yosida_rotation_compact_heatmap.png", "Average probe-weighted Yosida rotation across routes")
    _plot_heatmap(compact_p90, comp / "probed_yosida_rotation_p90_heatmap.png", "P90 probe-weighted Yosida rotation across routes")
    selected = _selected_dates(config.targets_dir, routes, by_date_df, common_dates)
    _plot_selected_alignment(
        selected,
        by_date_df,
        comp / "probed_yosida_alignment_selected_dates.png",
        comp / "probed_yosida_alignment_selected_dates.csv",
    )
    _plot_commutator_timeseries(comm_by_date, comp / "commutator_rotation_timeseries.png")
    _plot_anisotropy(anisotropy, config.min_anisotropy, comp / "route_anisotropy_paths.png")
    _plot_commutator_scatter(comm_by_date, comp / "commutator_vs_anisotropy.png")
    _plot_commutator_summary(comm_summary, comp / "commutator_rotation_summary.png")

    metadata = {
        "route_folders_used": {key: str(route.folder) for key, route in routes.items()},
        "route_labels": ROUTE_LABELS,
        "date_range": f"{common_dates.min().strftime('%Y-%m-%d')} to {common_dates.max().strftime('%Y-%m-%d')}",
        "number_of_common_dates": int(len(common_dates)),
        "p": int(S_star.shape[0]),
        "common_reference_mode": ref_meta["common_reference_mode"],
        "rho_star": float(ref_meta["rho_star"]),
        "d_star": float(d_star),
        "C_star_min_eigenvalue": ref_meta["C_star_min_eigenvalue"],
        "C_star_max_eigenvalue": ref_meta["C_star_max_eigenvalue"],
        "D_star_min_eigenvalue": ref_meta["D_star_min_eigenvalue"],
        "D_star_max_eigenvalue": ref_meta["D_star_max_eigenvalue"],
        "S_star_min_eigenvalue": ref_meta["S_star_min_eigenvalue"],
        "S_star_max_eigenvalue": ref_meta["S_star_max_eigenvalue"],
        "lambda_grid": [float(x) for x in lambdas],
        "probe_names": [probe.display for probe in probes.values()],
        "probe_diagnostics": {name: probe.diagnostics for name, probe in probes.items()},
        "min_anisotropy": float(config.min_anisotropy),
        "invalid_date_counts": ref_meta["invalid_tau_counts"],
        "NaN_counts": {
            "alignment_long_omega": int(long_df["omega"].isna().sum()),
            "alignment_by_date_omega_bar": int(by_date_df["omega_bar"].isna().sum()),
            "commutator_index": int(comm_by_date["commutator_index"].isna().sum()),
        },
        "alignment_denominator_tolerance": float(config.denominator_tol),
        "commutator_denominator_tolerance": float(config.denominator_tol),
        "whether_K_by_state_had_to_be_recomputed": False,
        "commutator_noninterpretable_plot_choice": "faint markers",
        "alignment_metadata": align_meta,
        "package_versions": _package_versions(),
    }
    (comp / "rotation_diagnostics_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_rotation_summary(comp / "summary.md", compact, compact_p90, comm_summary, metadata)
    return metadata
