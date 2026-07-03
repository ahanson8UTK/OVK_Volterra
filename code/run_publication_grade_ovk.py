#!/usr/bin/env python3
"""
Publication-grade upgrade for the monthly monetary-policy OVK estimator.

This script uses the full 125-coordinate ridge-soft covariance path for the
Section 3.1 headline figures, while retaining rank-five state-space results for
explicit comparison and appendix diagnostics.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import html
import json
import math
import os
for _thread_var in [
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")
import pickle
import shutil
import time
import zipfile
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, eigh as scipy_eigh, solve_discrete_lyapunov
from scipy.interpolate import UnivariateSpline
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ovk_data import (
    BASE_OUTCOME_COLUMNS,
    DEFAULT_OUTCOME_LABELS,
    build_outcome_frame,
    merge_extra_outcome_data,
    outcome_group_indices,
    outcome_labels_for_panel,
    outcome_signature_for_panel,
)


# -----------------------------
# Paths and configuration
# -----------------------------
SRC_ZIP = Path(os.environ.get("OVK_DATA_ZIP", "/mnt/data/data.zip"))
ROOT = Path(os.environ.get("OVK_PUBLICATION_ROOT", "/mnt/data/publication_grade_ovk"))
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
CHARTS = OUT / "charts"
CODE = OUT / "code"
REPORTS = Path(os.environ.get("OVK_REPORTS_DIR", str(ROOT / "reports")))
FINAL_PDF = Path(os.environ.get("OVK_PUBLICATION_FINAL_PDF", str(REPORTS / "publication_grade_ovk_report.pdf")))
FINAL_HTML = Path(os.environ.get("OVK_PUBLICATION_FINAL_HTML", str(REPORTS / "publication_grade_ovk_report.html")))
FINAL_ZIP = Path(os.environ.get("OVK_PUBLICATION_FINAL_ZIP", str(REPORTS / "publication_grade_ovk_bundle.zip")))
TOP5_COMPAT_OUT = Path(os.environ.get("OVK_TOP5_OUT", str(ROOT / "top5_compatible_outputs")))
TOP5_COMPAT_PDF = Path(os.environ.get("OVK_TOP5_FINAL_PDF", str(REPORTS / "top5_headline_report.pdf")))
TOP5_COMPAT_HTML = Path(os.environ.get("OVK_TOP5_FINAL_HTML", str(REPORTS / "top5_headline_report.html")))
TOP5_COMPAT_ZIP = Path(os.environ.get("OVK_TOP5_FINAL_ZIP", str(REPORTS / "top5_headline_bundle.zip")))
ROBUST_COMPAT_OUT = Path(os.environ.get("OVK_ROBUST_OUT", str(ROOT / "shock_robustness_outputs")))
for d in [OUT, TABLES, CHARTS, CODE, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)
for p in [FINAL_PDF, FINAL_HTML, FINAL_ZIP, TOP5_COMPAT_PDF, TOP5_COMPAT_HTML, TOP5_COMPAT_ZIP]:
    p.parent.mkdir(parents=True, exist_ok=True)

H = int(os.environ.get("OVK_PUBLICATION_H", "24"))
L = int(os.environ.get("OVK_PUBLICATION_L", "12"))
RANKS = [int(x) for x in os.environ.get("OVK_PUBLICATION_RANKS", "3,5,7").split(",")]
HEADLINE_R = int(os.environ.get("OVK_PUBLICATION_HEADLINE_R", "5"))
B_STATE = int(os.environ.get("OVK_PUBLICATION_STATE_DRAWS", "1000"))
B_BOOT = int(os.environ.get("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "1000"))
BOOT_BLOCK_LEN = int(os.environ.get("OVK_PUBLICATION_BOOT_BLOCK_LEN", "18"))
ROBUST_NU = float(os.environ.get("OVK_PUBLICATION_STUDENT_T_DF", "7"))
MIN_STUDENT_WEIGHT = float(os.environ.get("OVK_PUBLICATION_MIN_STUDENT_WEIGHT", "0.25"))
EM_ITERS = int(os.environ.get("OVK_PUBLICATION_EM_ITERS", "5"))
BOOT_EM_ITERS = int(os.environ.get("OVK_PUBLICATION_BOOT_EM_ITERS", "4"))
STATE_SEED = int(os.environ.get("OVK_PUBLICATION_STATE_SEED", "9127"))
BOOT_SEED = int(os.environ.get("OVK_PUBLICATION_BOOT_SEED", "7031"))
ALPHA_GRID = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35], dtype=float)
ESTIMATOR_MODE = os.environ.get("OVK_COVARIANCE_ESTIMATOR_MODE", "arithmetic_outer_product").strip().lower()
VALID_ESTIMATOR_MODES = {"arithmetic_outer_product", "log_spd_legacy"}
if ESTIMATOR_MODE not in VALID_ESTIMATOR_MODES:
    raise ValueError(f"OVK_COVARIANCE_ESTIMATOR_MODE must be one of {sorted(VALID_ESTIMATOR_MODES)}")
ARITHMETIC_REFERENCE_RIDGE_SCALE = float(os.environ.get("OVK_ARITHMETIC_REFERENCE_RIDGE_SCALE", "1e-8"))
FULL_COORDINATE_KERNEL_ETA = float(os.environ.get("OVK_FULL_COORDINATE_KERNEL_ETA", "0.08"))
FULL_COORDINATE_CELL_VARIANCE_TOL = float(os.environ.get("OVK_FULL_COORDINATE_CELL_VARIANCE_TOL", "1e-12"))
WRITE_LEGACY_TOP5_COMPAT = os.environ.get("OVK_WRITE_LEGACY_TOP5_COMPAT", "0").lower() in {"1", "true", "yes", "on"}
HEADLINE_OUTCOMES = os.environ.get("OVK_HEADLINE_OUTCOMES", "base5").strip().lower()
SF_FED_SURPRISES = Path(os.environ.get("OVK_SF_FED_SURPRISES", "data_raw/external/sf_fed_monetary_policy_surprises.xlsx"))
PLACEBO_SEED = int(os.environ.get("OVK_PUBLICATION_PLACEBO_SEED", "20260603"))
PLACEBO_SHIFT_MONTHS = int(os.environ.get("OVK_PUBLICATION_PLACEBO_SHIFT_MONTHS", "84"))
BOOTSTRAP_VARIANTS = os.environ.get("OVK_PUBLICATION_BOOTSTRAP_VARIANTS", "all").strip().lower()
SMOOTH_SPLINE_S = float(os.environ.get("OVK_PUBLICATION_SMOOTH_SPLINE_S", "1.0"))
OUTCOME_LABELS = list(DEFAULT_OUTCOME_LABELS)
M_DIM = (H + 1) * len(OUTCOME_LABELS)
CACHE_VERSION = "publication_speed_v4_calendar_safe"
WORKER_ARRAY_VERSION = "worker_arrays_v1"

_workers_env = os.environ.get("OVK_PUBLICATION_WORKERS", "").strip()
_max_auto_workers = max(1, min((os.cpu_count() or 2) - 1, 8))
if _workers_env:
    PUBLICATION_WORKERS = max(1, int(_workers_env))
else:
    PUBLICATION_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 4))
BENCHMARK_WORKERS = os.environ.get("OVK_PUBLICATION_BENCHMARK_WORKERS", "0").lower() in {"1", "true", "yes", "on"}
CACHE_DIR = Path(os.environ.get("OVK_CACHE_DIR", str(ROOT.parent / ".ovk_cache")))
DISABLE_CACHE = os.environ.get("OVK_DISABLE_CACHE", "0").lower() in {"1", "true", "yes", "on"}
if not DISABLE_CACHE:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RANK_CONTEXT: dict[str, Any] = {}
_BOOTSTRAP_CONTEXT: dict[str, Any] = {}


def set_outcome_labels(labels: list[str]) -> None:
    global OUTCOME_LABELS, M_DIM
    OUTCOME_LABELS = list(labels)
    M_DIM = (H + 1) * len(OUTCOME_LABELS)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(*parts: Any) -> str:
    payload = json.dumps([CACHE_VERSION, *parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"


def npz_cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.npz"


def cache_get(key: str) -> Any | None:
    if DISABLE_CACHE:
        return None
    path = cache_path(key)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def cache_set(key: str, value: Any) -> None:
    if DISABLE_CACHE:
        return
    path = cache_path(key)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def cache_get_npz(key: str) -> dict[str, np.ndarray] | None:
    if DISABLE_CACHE:
        return None
    path = npz_cache_path(key)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as fh:
        return {name: fh[name] for name in fh.files}


def cache_set_npz(key: str, arrays: dict[str, np.ndarray]) -> None:
    if DISABLE_CACHE:
        return
    path = npz_cache_path(key)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_worker_array(namespace: str, name: str, arr: np.ndarray) -> Path:
    arr = np.asarray(arr)
    if DISABLE_CACHE:
        root = OUT / "_worker_arrays" / namespace
    else:
        root = CACHE_DIR / "worker_arrays" / namespace
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.npy"
    if path.exists():
        return path
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
    with tmp.open("wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)
    return path


@dataclass
class StateFit:
    alpha: float
    ylog: np.ndarray
    mu: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    Rmat: np.ndarray
    xs: np.ndarray
    Ps: np.ndarray
    xf: np.ndarray
    Pf: np.ndarray
    xp: np.ndarray
    Pp: np.ndarray
    weights: np.ndarray
    loglik: float
    robust_loglik: float
    factor_log_score: float
    spectral_radius: float
    transition_shrinkage: dict[str, float]
    estimator_mode: str = "log_spd_legacy"
    observation_kind: str = "log_spd"
    objective_weights: np.ndarray | None = None


@dataclass
class RankResult:
    variant: str
    label: str
    rank: int
    dates: pd.Series
    outcome_labels: list[str]
    Q_scores: np.ndarray
    beta: np.ndarray
    eigvals: np.ndarray
    shares: np.ndarray
    V: np.ndarray
    Z: np.ndarray
    fit: StateFit
    A: np.ndarray
    tau: np.ndarray
    scale_log_tau: np.ndarray
    shape_distance: np.ndarray
    estimator_mode: str = "log_spd_legacy"
    reference_weights: np.ndarray | None = None
    reference_covariance: np.ndarray | None = None
    whitening_ridge: float = 0.0
    whitening_map: np.ndarray | None = None
    unwhitening_map: np.ndarray | None = None
    total_second_moment_whitened: np.ndarray | None = None
    mean_component_whitened: np.ndarray | None = None
    within_covariance_whitened: np.ndarray | None = None
    total_second_moment_original: np.ndarray | None = None
    mean_component_original: np.ndarray | None = None
    within_covariance_original: np.ndarray | None = None
    smoothed_mean_whitened: np.ndarray | None = None
    smoothed_mean_original: np.ndarray | None = None
    mean_fit: StateFit | None = None
    reference_weighted_mean: np.ndarray | None = None


@dataclass
class FullCoordinateResult:
    variant: str
    label: str
    dates: pd.Series
    outcome_labels: list[str]
    chi: np.ndarray
    C_hat: np.ndarray
    D_rho: np.ndarray
    D_invsqrt: np.ndarray
    rho: float
    d_rho: float
    temporal_weights: np.ndarray
    K_hat: np.ndarray
    A_hat: np.ndarray
    tau_soft: np.ndarray
    cell_amp: np.ndarray
    cell_shape: np.ndarray
    low_variance_cell_mask: np.ndarray
    backend: str = "temporal_kernel"
    kernel_eta: float = FULL_COORDINATE_KERNEL_ETA


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    outcome_columns: tuple[str, ...]
    shock_col: str
    cbi_col: str | None
    sample_dates_key: str | None = None
    transform: str = "raw"
    group: str = "headline"
    run_ranks: tuple[int, ...] = (HEADLINE_R,)
    source_note: str = ""
    available: bool = True
    skip_reason: str = ""

    @property
    def transform_signature(self) -> str:
        if self.transform == "smooth":
            return f"smooth_bspline_s{SMOOTH_SPLINE_S:g}"
        return self.transform


# -----------------------------
# Numeric helpers
# -----------------------------
_TRI_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def tri_indices(R: int) -> tuple[np.ndarray, np.ndarray]:
    if R not in _TRI_CACHE:
        rows = list(range(R))
        cols = list(range(R))
        for i in range(R):
            for j in range(i + 1, R):
                rows.append(i)
                cols.append(j)
        _TRI_CACHE[R] = (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))
    return _TRI_CACHE[R]


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def sym_last(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def stabilize_cov(C: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    C = sym(np.asarray(C, float))
    w, U = np.linalg.eigh(C)
    w = np.maximum(w, floor)
    return U @ np.diag(w) @ U.T


def spd_eigh(A: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    w, U = np.linalg.eigh(sym(A))
    return np.maximum(w, eps), U


def spd_invsqrt(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, U = spd_eigh(A, eps)
    return U @ np.diag(1.0 / np.sqrt(w)) @ U.T


def spd_sqrt(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, U = spd_eigh(A, eps)
    return U @ np.diag(np.sqrt(w)) @ U.T


def project_psd(A: np.ndarray, floor: float = 0.0) -> np.ndarray:
    S = sym_last(np.asarray(A, float))
    w, U = np.linalg.eigh(S)
    w = np.maximum(w, floor)
    return (U * w[..., None, :]) @ np.swapaxes(U, -1, -2)


def normalize_reference_weights(n_obs: int, reference_weights: np.ndarray | None = None) -> np.ndarray:
    if reference_weights is None:
        return np.full(n_obs, 1.0 / max(n_obs, 1), dtype=float)
    w = np.asarray(reference_weights, dtype=float).reshape(-1)
    if len(w) != n_obs:
        raise ValueError(f"reference_weights length {len(w)} does not match n_obs={n_obs}")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("reference_weights must be finite and nonnegative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("reference_weights must have positive total mass")
    return w / total


def objective_weight_scale(reference_weights: np.ndarray | None, n_obs: int) -> np.ndarray:
    w = normalize_reference_weights(n_obs, reference_weights)
    return np.maximum(w * n_obs, 1e-12)


def weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return np.average(np.asarray(X, dtype=float), axis=0, weights=w)


def weighted_covariance(X: np.ndarray, weights: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float)
    mu = weighted_mean(X, w)
    Xc = X - mu
    C = (Xc.T * w) @ Xc / max(float(w.sum()), 1e-12)
    return stabilize_cov(C, floor)


def batched_spd_invsqrt(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    w, U = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (U * (1.0 / np.sqrt(w))[..., None, :]) @ np.swapaxes(U, -1, -2)


def spd_log(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, U = spd_eigh(A, eps)
    return U @ np.diag(np.log(w)) @ U.T


def spd_exp(S: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(sym(S))
    return U @ np.diag(np.exp(w)) @ U.T


def batched_spd_log(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    w, U = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (U * np.log(w)[..., None, :]) @ np.swapaxes(U, -1, -2)


def batched_spd_exp(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    w, U = np.linalg.eigh(S)
    return (U * np.exp(w)[..., None, :]) @ np.swapaxes(U, -1, -2)


def svec(S: np.ndarray) -> np.ndarray:
    rows, cols = tri_indices(S.shape[0])
    return np.asarray(S)[rows, cols]


def svec_batch(S: np.ndarray) -> np.ndarray:
    rows, cols = tri_indices(S.shape[-1])
    return np.asarray(S)[..., rows, cols]


def smat(v: np.ndarray, R: int) -> np.ndarray:
    v = np.asarray(v)
    if v.ndim > 1:
        out = np.zeros(v.shape[:-1] + (R, R))
        rows, cols = tri_indices(R)
        out[..., rows, cols] = v
        out[..., cols, rows] = v
        return out
    S = np.zeros((R, R))
    rows, cols = tri_indices(R)
    S[rows, cols] = v
    S[cols, rows] = v
    return S


def circular_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
    blocks = [((s + np.arange(block_len)) % n) for s in starts]
    return np.concatenate(blocks)[:n]


def safe_name(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "").replace("/", "_").replace("(", "").replace(")", "")


def arithmetic_matrix_series_from_state(Xstate: np.ndarray, R: int) -> np.ndarray:
    """Decode arithmetic second-moment states without matrix log/exp maps."""
    return project_psd(smat(np.asarray(Xstate), R), floor=0.0)


def matrix_series_from_state(Xstate: np.ndarray, R: int, estimator_mode: str = "log_spd_legacy") -> np.ndarray:
    if estimator_mode == "arithmetic_outer_product":
        return arithmetic_matrix_series_from_state(Xstate, R)
    if estimator_mode != "log_spd_legacy":
        raise ValueError(f"Unknown estimator mode: {estimator_mode}")
    mats = batched_spd_exp(smat(np.asarray(Xstate), R))
    C = mats.mean(axis=0)
    Cinv = spd_invsqrt(C, eps=1e-10)
    return Cinv @ mats @ Cinv


def state_draw_scale_shape(
    Xdraws: np.ndarray,
    R: int,
    chunk_draws: int = 64,
    estimator_mode: str = "log_spd_legacy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xdraws = np.asarray(Xdraws)
    B, T, d = Xdraws.shape
    tau_draws = np.empty((B, T))
    scale_draws = np.empty((B, T))
    shape_draws = np.empty((B, T))
    diag_draws = np.empty((B, R, T))
    for start in range(0, B, chunk_draws):
        stop = min(start + chunk_draws, B)
        Xc = Xdraws[start:stop]
        n = stop - start
        if estimator_mode == "arithmetic_outer_product":
            A = arithmetic_matrix_series_from_state(Xc.reshape(n * T, d), R).reshape(n, T, R, R)
        elif estimator_mode == "log_spd_legacy":
            mats = batched_spd_exp(smat(Xc.reshape(n * T, d), R)).reshape(n, T, R, R)
            Cinv = batched_spd_invsqrt(mats.mean(axis=1), eps=1e-10)
            A = np.einsum("bij,btjk,bkl->btil", Cinv, mats, Cinv, optimize=True)
        else:
            raise ValueError(f"Unknown estimator mode: {estimator_mode}")
        tau = np.trace(A, axis1=2, axis2=3) / R
        tau_safe = np.maximum(tau, 1e-12)
        shape = A / tau_safe[:, :, None, None]
        if estimator_mode == "arithmetic_outer_product":
            shape_distance = np.linalg.norm(shape - np.eye(R)[None, None, :, :], axis=(2, 3))
        else:
            shape_logs = batched_spd_log(shape.reshape(n * T, R, R), eps=1e-10).reshape(n, T, R, R)
            shape_distance = np.linalg.norm(shape_logs, axis=(2, 3))
        tau_draws[start:stop] = tau
        scale_draws[start:stop] = np.log(tau_safe)
        shape_draws[start:stop] = shape_distance
        diag_draws[start:stop] = np.moveaxis(np.diagonal(A, axis1=2, axis2=3), 2, 1)
    return tau_draws, scale_draws, shape_draws, diag_draws


def scale_shape_from_A(A: np.ndarray, estimator_mode: str = "log_spd_legacy") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = A.shape[1]
    tau = np.trace(A, axis1=1, axis2=2) / R
    tau_safe = np.maximum(tau, 1e-12)
    shape = A / tau_safe[:, None, None]
    if estimator_mode == "arithmetic_outer_product":
        shape_distance = np.linalg.norm(shape - np.eye(R)[None, :, :], axis=(1, 2))
    elif estimator_mode == "log_spd_legacy":
        shape_logs = batched_spd_log(shape, eps=1e-10)
        shape_distance = np.linalg.norm(shape_logs, axis=(1, 2))
    else:
        raise ValueError(f"Unknown estimator mode: {estimator_mode}")
    return tau, np.log(tau_safe), shape_distance


def normalized_effective_support(p: np.ndarray, axis: int) -> np.ndarray:
    p = np.asarray(p, float)
    p = np.maximum(p, 0.0)
    total = np.maximum(p.sum(axis=axis, keepdims=True), 1e-12)
    prob = p / total
    entropy = -np.sum(np.where(prob > 0, prob * np.log(np.maximum(prob, 1e-12)), 0.0), axis=axis)
    return np.exp(entropy) / max(p.shape[axis], 1)


def surface_shape_metrics_from_diag(surface_diag: np.ndarray, H: int, pvars: int, labels: list[str]) -> dict[str, np.ndarray]:
    surface_diag = np.maximum(np.asarray(surface_diag, float), 0.0)
    total = np.maximum(surface_diag.sum(axis=1), 1e-12)
    cube = surface_diag.reshape(surface_diag.shape[0], H + 1, pvars)
    cell_share = surface_diag / total[:, None]
    variable_share = cube.sum(axis=1) / total[:, None]
    horizon_share = cube.sum(axis=2) / total[:, None]
    macro_idx, financial_idx = outcome_group_indices(labels)
    horizons = np.arange(H + 1)
    short_mask = horizons <= min(3, H)
    medium_mask = (horizons >= 4) & (horizons <= min(12, H))
    long_mask = horizons >= 13
    metrics = {
        "macro_variable_share": variable_share[:, macro_idx].sum(axis=1) if macro_idx else np.zeros(surface_diag.shape[0]),
        "financial_variable_share": variable_share[:, financial_idx].sum(axis=1) if financial_idx else np.zeros(surface_diag.shape[0]),
        "short_horizon_share": horizon_share[:, short_mask].sum(axis=1) if short_mask.any() else np.zeros(surface_diag.shape[0]),
        "medium_horizon_share": horizon_share[:, medium_mask].sum(axis=1) if medium_mask.any() else np.zeros(surface_diag.shape[0]),
        "long_horizon_share": horizon_share[:, long_mask].sum(axis=1) if long_mask.any() else np.zeros(surface_diag.shape[0]),
        "cell_effective_support": normalized_effective_support(cell_share, axis=1),
        "variable_effective_support": normalized_effective_support(variable_share, axis=1),
        "horizon_effective_support": normalized_effective_support(horizon_share, axis=1),
    }
    for j, label in enumerate(labels):
        metrics[f"variable_share_{safe_name(label)}"] = variable_share[:, j]
    for hh in range(H + 1):
        metrics[f"horizon_{hh:02d}_share"] = horizon_share[:, hh]
    return metrics


def surface_shape_from_A(A: np.ndarray, V: np.ndarray, H: int, pvars: int, labels: list[str]) -> dict[str, Any]:
    R = A.shape[1]
    tau = np.trace(A, axis1=1, axis2=2) / R
    tau_safe = np.maximum(tau, 1e-12)
    shape = A / tau_safe[:, None, None]
    baseline = np.einsum("mr,mr->m", V, V, optimize=True)
    baseline_safe = np.maximum(baseline, 1e-14)
    surface_diag = np.einsum("mr,trs,ms->tm", V, shape, V, optimize=True)
    surface_diag = np.maximum(surface_diag, 1e-14)
    relative = surface_diag / baseline_safe[None, :]
    log_relative = np.log(np.maximum(relative, 1e-14))
    leverage_weights = baseline / np.maximum(baseline.sum(), 1e-12)
    surface_rms = np.sqrt(np.sum(leverage_weights[None, :] * log_relative**2, axis=1))
    metrics = surface_shape_metrics_from_diag(surface_diag, H, pvars, labels)
    metrics["surface_shape_rms_log_relative"] = surface_rms
    return {
        "shape": shape,
        "baseline_leverage": baseline,
        "surface_diag": surface_diag,
        "relative_shape_variance": relative,
        "log_relative_shape_variance": log_relative,
        "surface_shape_rms_log_relative": surface_rms,
        "metrics": metrics,
    }


def state_draw_surface_shape_metric_draws(
    Xdraws: np.ndarray,
    V: np.ndarray,
    H: int,
    pvars: int,
    labels: list[str],
    metric_names: list[str],
    chunk_draws: int = 32,
    estimator_mode: str = "log_spd_legacy",
) -> np.ndarray:
    Xdraws = np.asarray(Xdraws)
    B, T, d = Xdraws.shape
    R = V.shape[1]
    out = np.empty((B, T, len(metric_names)))
    for start in range(0, B, chunk_draws):
        stop = min(start + chunk_draws, B)
        Xc = Xdraws[start:stop]
        n = stop - start
        if estimator_mode == "arithmetic_outer_product":
            A = arithmetic_matrix_series_from_state(Xc.reshape(n * T, d), R).reshape(n, T, R, R)
        elif estimator_mode == "log_spd_legacy":
            mats = batched_spd_exp(smat(Xc.reshape(n * T, d), R)).reshape(n, T, R, R)
            Cinv = batched_spd_invsqrt(mats.mean(axis=1), eps=1e-10)
            A = np.einsum("bij,btjk,bkl->btil", Cinv, mats, Cinv, optimize=True)
        else:
            raise ValueError(f"Unknown estimator mode: {estimator_mode}")
        surface = surface_shape_from_A(A.reshape(n * T, R, R), V, H, pvars, labels)
        for k, name in enumerate(metric_names):
            out[start:stop, :, k] = surface["metrics"][name].reshape(n, T)
    return out


def principal_angles(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    k = min(Va.shape[1], Vb.shape[1])
    s = np.linalg.svd(Va[:, :k].T @ Vb[:, :k], compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def match_modes(Uboot: np.ndarray, Vbase: np.ndarray) -> np.ndarray:
    corr = np.abs(Vbase.T @ Uboot)
    row_ind, col_ind = linear_sum_assignment(-corr)
    matched = np.zeros_like(Vbase)
    for r, c in zip(row_ind, col_ind):
        v = Uboot[:, c]
        if np.dot(v, Vbase[:, r]) < 0:
            v = -v
        matched[:, r] = v
    return matched


def gaussian_logpdf(e: np.ndarray, S: np.ndarray) -> float:
    S = stabilize_cov(S, 1e-10)
    d = len(e)
    c, lower = cho_factor(S, lower=True, check_finite=False)
    logdet = 2.0 * np.log(np.diag(c)).sum()
    sol = cho_solve((c, lower), e, check_finite=False)
    return float(-0.5 * (d * np.log(2 * np.pi) + logdet + e @ sol))


def student_t_logpdf(e: np.ndarray, S: np.ndarray, nu: float) -> float:
    S = stabilize_cov(S, 1e-10)
    d = len(e)
    c, lower = cho_factor(S, lower=True, check_finite=False)
    logdet = 2.0 * np.log(np.diag(c)).sum()
    q = float(e @ cho_solve((c, lower), e, check_finite=False))
    return float(
        gammaln((nu + d) / 2)
        - gammaln(nu / 2)
        - 0.5 * (d * np.log(nu * np.pi) + logdet)
        - ((nu + d) / 2) * np.log1p(q / nu)
    )


# -----------------------------
# Data and LP score construction
# -----------------------------
ALL_OUTCOME_COLUMNS = (
    "ip",
    "cpi",
    "cpi10",
    "unrate",
    "gs2",
    "baa10y",
    "expinf5yr",
    "mich",
)
EPISODES = [
    ("1994 tightening", "1994-02", "1995-02"),
    ("1998 LTCM/Russia", "1998-08", "1998-10"),
    ("2007-08 financial crisis", "2007-08", "2009-03"),
    ("2019 repo/pre-COVID", "2019-07", "2020-02"),
    ("2020 COVID", "2020-03", "2020-06"),
    ("2022 inflation/tightening", "2022-03", "2023-07"),
]


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()


def complete_monthly_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Preserve one row per calendar month so LP row offsets equal month offsets."""
    if panel.empty or "date" not in panel.columns:
        return panel.copy()
    out = panel.copy()
    out["date"] = _month_start(out["date"])
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        return out
    calendar = pd.DataFrame({"date": pd.date_range(out["date"].min(), out["date"].max(), freq="MS")})
    return calendar.merge(out, on="date", how="left")


def resolve_data_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def panel_with_outcomes(panel: pd.DataFrame, outcome_columns: tuple[str, ...]) -> pd.DataFrame:
    """Return panel with unrequested recognized outcome columns removed."""
    keep = set(outcome_columns)
    missing = [c for c in outcome_columns if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing requested outcome columns: {', '.join(missing)}")
    drop = [c for c in ALL_OUTCOME_COLUMNS if c in panel.columns and c not in keep]
    return panel.drop(columns=drop)


def outcome_columns_for_labels(labels: list[str]) -> list[str]:
    label_to_col = {
        "IP": "ip",
        "CPI": "cpi",
        "Median CPI10": "cpi10",
        "Unemployment": "unrate",
        "2Y yield": "gs2",
        "BAA-10Y spread": "baa10y",
        "5Y expected inflation": "expinf5yr",
        "Michigan inflation expectations": "mich",
    }
    return [label_to_col.get(label, label) for label in labels]


def outcome_block_traces(q_scores: np.ndarray, labels: list[str]) -> np.ndarray:
    centered = np.asarray(q_scores, float) - np.asarray(q_scores, float).mean(axis=0)
    diag = (centered**2).mean(axis=0).reshape(H + 1, len(labels))
    return diag.sum(axis=0)


def standardization_weights(q_scores: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    traces = outcome_block_traces(q_scores, labels)
    target = float(np.mean(traces))
    weights_by_outcome = np.sqrt(target / np.maximum(traces, 1e-14))
    weights = np.tile(weights_by_outcome, H + 1)
    return weights, traces, target


def smooth_score_surfaces(q_scores: np.ndarray, labels: list[str], smoothing: float = SMOOTH_SPLINE_S) -> np.ndarray:
    """Apply cubic B-spline horizon smoothing within each outcome block."""
    Q = np.asarray(q_scores, float)
    pvars = len(labels)
    cube = Q.reshape(Q.shape[0], H + 1, pvars)
    horizons = np.arange(H + 1, dtype=float)
    out = np.empty_like(cube)
    s_value = max(float(smoothing), 0.0) * (H + 1)
    for t in range(cube.shape[0]):
        for j in range(pvars):
            y = cube[t, :, j]
            try:
                out[t, :, j] = UnivariateSpline(horizons, y, k=3, s=s_value)(horizons)
            except Exception:
                out[t, :, j] = y
    return out.reshape(Q.shape)


def score_roughness(q_scores: np.ndarray, labels: list[str]) -> float:
    cube = np.asarray(q_scores, float).reshape(q_scores.shape[0], H + 1, len(labels))
    if cube.shape[1] < 3:
        return float("nan")
    second_diff = np.diff(cube, n=2, axis=1)
    return float(np.mean(second_diff**2))


def apply_score_transform(q_scores: np.ndarray, labels: list[str], transform: dict[str, Any] | None) -> np.ndarray:
    if not transform or transform.get("kind", "raw") == "raw":
        return np.asarray(q_scores, float)
    kind = transform.get("kind")
    if kind == "standardized":
        weights = np.asarray(transform["weights"], float)
        if weights.shape[0] != q_scores.shape[1]:
            raise ValueError("Outcome standardization weights do not match score dimension.")
        return np.asarray(q_scores, float) * weights[None, :]
    if kind == "smooth":
        return smooth_score_surfaces(q_scores, labels, smoothing=float(transform.get("smoothing", SMOOTH_SPLINE_S)))
    raise ValueError(f"Unknown score transform: {kind}")


def add_placebo_shocks(panel: pd.DataFrame, seed: int = PLACEBO_SEED, shift_months: int = PLACEBO_SHIFT_MONTHS) -> pd.DataFrame:
    out = panel.copy()
    base = pd.to_numeric(out["MP_median_fallback"], errors="coerce").to_numpy(float)
    perm = base.copy()
    finite = np.isfinite(perm)
    rng = np.random.default_rng(seed)
    perm[finite] = perm[finite][rng.permutation(finite.sum())]
    out["MP_placebo_permuted"] = perm
    out["MP_placebo_shift84"] = np.roll(base, int(shift_months))
    out["CBI_placebo_control"] = out["CBI_median_fallback"]
    return out


def load_sf_fed_surprises(path: Path) -> pd.DataFrame:
    """Parse a vendored SF Fed/Bauer-Swanson surprise snapshot into monthly shocks.

    The official files have changed column names over time, so the parser uses
    conservative heuristics: one date column plus the first numeric surprise-like
    column. Tests cover the supported fixture shape.
    """
    path = resolve_data_path(path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "SF_raw_surprise"])
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        xl = pd.ExcelFile(path)
        monthly_sheets = [s for s in xl.sheet_names if "monthly" in s.lower() and "update" in s.lower()]
        sheet = monthly_sheets[0] if monthly_sheets else next((s for s in xl.sheet_names if "monthly" in s.lower()), xl.sheet_names[0])
        raw = pd.read_excel(path, sheet_name=sheet)
    else:
        raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "SF_raw_surprise"])
    cols_lower = {str(c).lower(): c for c in raw.columns}
    if {"year", "month"}.issubset(cols_lower):
        date_values = pd.to_datetime(
            dict(
                year=pd.to_numeric(raw[cols_lower["year"]], errors="coerce"),
                month=pd.to_numeric(raw[cols_lower["month"]], errors="coerce"),
                day=1,
            ),
            errors="coerce",
        )
        raw_col = cols_lower.get("mps")
        orth_col = cols_lower.get("mps_orth")
        if raw_col is not None:
            out = pd.DataFrame({"date": _month_start(pd.Series(date_values)), "SF_raw_surprise": pd.to_numeric(raw[raw_col], errors="coerce")})
            if orth_col is not None:
                out["SF_orthogonalized_source"] = pd.to_numeric(raw[orth_col], errors="coerce")
            out = out.dropna(subset=["date", "SF_raw_surprise"]).groupby("date", as_index=False).sum(numeric_only=True)
            out["SF_source_column"] = str(raw_col)
            out["SF_source_file"] = str(path)
            return out.sort_values("date").reset_index(drop=True)
    date_col = None
    for needle in ["date", "fomc", "meeting"]:
        matches = [orig for low, orig in cols_lower.items() if needle in low]
        if matches:
            date_col = matches[0]
            break
    if date_col is None:
        date_col = raw.columns[0]
    candidates = []
    for col in raw.columns:
        if col == date_col:
            continue
        low = str(col).lower()
        numeric = pd.to_numeric(raw[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        score = int("surprise" in low) + int("mp" in low or "monetary" in low) + int("target" in low) - int("info" in low or "cbi" in low)
        candidates.append((score, col, numeric))
    if not candidates:
        return pd.DataFrame(columns=["date", "SF_raw_surprise"])
    _, value_col, values = sorted(candidates, key=lambda x: (-x[0], str(x[1]).lower()))[0]
    out = pd.DataFrame({"date": _month_start(raw[date_col]), "SF_raw_surprise": values})
    out = out.dropna(subset=["date", "SF_raw_surprise"]).groupby("date", as_index=False)["SF_raw_surprise"].sum()
    out["SF_source_column"] = str(value_col)
    out["SF_source_file"] = str(path)
    return out.sort_values("date").reset_index(drop=True)


def add_sf_fed_shocks(panel: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    sf = load_sf_fed_surprises(path)
    merge_cols = [c for c in ["date", "SF_raw_surprise", "SF_orthogonalized_source"] if c in sf.columns]
    out = panel.merge(sf[merge_cols] if merge_cols else sf, on="date", how="left")
    meta = {
        "sf_fed_path": str(resolve_data_path(path)),
        "sf_fed_available": bool(len(sf)),
        "sf_fed_rows": int(len(sf)),
        "sf_fed_source_column": str(sf["SF_source_column"].iloc[0]) if len(sf) and "SF_source_column" in sf.columns else "",
        "sf_fed_primary_source": "https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/",
        "sf_fed_primary_source_updated": "2025-08-25",
        "sf_fed_usmpd_context_source": "https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/",
        "sf_fed_usmpd_context_source_updated": "2026-04-30",
    }
    if "SF_raw_surprise" not in out.columns:
        out["SF_raw_surprise"] = np.nan
    x = pd.to_numeric(out["SF_raw_surprise"], errors="coerce")
    if "SF_orthogonalized_source" in out.columns and pd.to_numeric(out["SF_orthogonalized_source"], errors="coerce").notna().sum() >= 12:
        ortho = pd.to_numeric(out["SF_orthogonalized_source"], errors="coerce").to_numpy(float)
        meta["sf_fed_orthogonalized_on"] = "source MPS_ORTH"
    else:
        c = pd.to_numeric(out["CBI_median_fallback"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(c)
        ortho = x.to_numpy(float).copy()
        if mask.sum() >= 12 and np.nanvar(c[mask]) > 0:
            X = np.column_stack([np.ones(mask.sum()), c[mask]])
            coef = np.linalg.lstsq(X, x[mask], rcond=None)[0]
            ortho[mask.to_numpy()] = x[mask] - X @ coef
            meta["sf_fed_orthogonalized_on"] = "CBI_median_fallback"
        else:
            meta["sf_fed_orthogonalized_on"] = "not enough overlapping CBI observations"
    out["SF_orthogonalized_surprise"] = ortho
    return out, meta


def variant_specs(panel: pd.DataFrame, sf_meta: dict[str, Any], overlap_dates: pd.Series) -> list[VariantSpec]:
    base5 = tuple(BASE_OUTCOME_COLUMNS)
    all8 = tuple(c for c in ALL_OUTCOME_COLUMNS if c in panel.columns)
    headline_ranks = tuple(RANKS) if HEADLINE_OUTCOMES == "base5" else (HEADLINE_R,)
    specs = [
        VariantSpec("base5_headline", "Original five outcomes, original sample", base5, "MP_median_fallback", "CBI_median_fallback", group="headline", run_ranks=headline_ranks),
        VariantSpec("base5_expectation_overlap", "Original five outcomes, expectations-overlap sample", base5, "MP_median_fallback", "CBI_median_fallback", sample_dates_key="expectation_overlap", group="same_sample"),
        VariantSpec("all8_expectation_overlap", "Eight outcomes, expectations-overlap sample", all8, "MP_median_fallback", "CBI_median_fallback", sample_dates_key="expectation_overlap", group="same_sample"),
        VariantSpec("base5_headline_standardized", "Original five outcomes, standardized outcome traces", base5, "MP_median_fallback", "CBI_median_fallback", transform="standardized", group="standardization"),
        VariantSpec("base5_expectation_overlap_standardized", "Original five outcomes overlap, standardized outcome traces", base5, "MP_median_fallback", "CBI_median_fallback", sample_dates_key="expectation_overlap", transform="standardized", group="standardization"),
        VariantSpec("all8_expectation_overlap_standardized", "Eight outcomes overlap, standardized outcome traces", all8, "MP_median_fallback", "CBI_median_fallback", sample_dates_key="expectation_overlap", transform="standardized", group="standardization"),
        VariantSpec("base5_mp_pm_only", "MP_pm only with CBI_pm controls", base5, "MP_pm", "CBI_pm", group="shock_definition"),
        VariantSpec("base5_event_manual", "Event-level manual aggregation", base5, "MP_event_manual", "CBI_event_manual", group="shock_definition"),
        VariantSpec("policy_without_cbi", "Policy shock without CBI controls", base5, "MP_median_fallback", None, group="policy_cbi_split"),
        VariantSpec("cbi_with_policy", "CBI shock with policy controls", base5, "CBI_median_fallback", "MP_median_fallback", group="policy_cbi_split"),
        VariantSpec("cbi_without_policy", "CBI shock without policy controls", base5, "CBI_median_fallback", None, group="policy_cbi_split"),
        VariantSpec("placebo_permuted", "Seeded month-permuted policy shock", base5, "MP_placebo_permuted", "CBI_placebo_control", group="placebo"),
        VariantSpec("placebo_shift84", "Policy shock circularly shifted 84 months", base5, "MP_placebo_shift84", "CBI_placebo_control", group="placebo"),
        VariantSpec("base5_headline_smooth", "Original five outcomes with cubic B-spline smoothed LP surfaces", base5, "MP_median_fallback", "CBI_median_fallback", transform="smooth", group="smooth_lp"),
    ]
    sf_available = bool(sf_meta.get("sf_fed_available"))
    specs.extend(
        [
            VariantSpec("sf_fed_raw", "SF Fed/Bauer-Swanson raw monthly surprise", base5, "SF_raw_surprise", "CBI_median_fallback", group="sf_fed", available=sf_available, skip_reason="" if sf_available else "Vendored SF Fed surprise file not found or empty."),
            VariantSpec("sf_fed_orthogonalized", "SF Fed/Bauer-Swanson surprise orthogonalized to JK CBI", base5, "SF_orthogonalized_surprise", "CBI_median_fallback", group="sf_fed", available=sf_available, skip_reason="" if sf_available else "Vendored SF Fed surprise file not found or empty."),
        ]
    )
    return specs


def lp_scores_from_design(
    Xs: np.ndarray,
    Yall: np.ndarray,
    ridge: float = 1e-8,
    row_xx: np.ndarray | None = None,
    row_xy: np.ndarray | None = None,
    ix: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    Xs = np.asarray(Xs, float)
    Yall = np.asarray(Yall, float)
    if ix is None:
        Xb = Xs
        Yb = Yall
        XtX = Xb.T @ Xb if row_xx is None else row_xx.sum(axis=0)
        XtY = Xb.T @ Yb if row_xy is None else row_xy.sum(axis=0)
    else:
        ix = np.asarray(ix, dtype=int)
        Xb = Xs[ix]
        Yb = Yall[ix]
        XtX = Xb.T @ Xb if row_xx is None else row_xx[ix].sum(axis=0)
        XtY = Xb.T @ Yb if row_xy is None else row_xy[ix].sum(axis=0)
    Bcoef = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    resid = Yb - Xb @ Bcoef
    m_res = resid[:, 0]
    Y_res = resid[:, 1:]
    sigma_m2 = float(np.mean(m_res**2))
    Q_scores = (m_res[:, None] * Y_res) / sigma_m2
    return Q_scores, sigma_m2


def lp_row_sufficient_stats(Xs: np.ndarray, Yall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = np.asarray(Xs, float)
    Yall = np.asarray(Yall, float)
    row_xx = np.einsum("ti,tj->tij", Xs, Xs, optimize=True)
    row_xy = np.einsum("ti,tj->tij", Xs, Yall, optimize=True)
    return row_xx, row_xy


def load_panels_from_zip(src_zip: Path) -> dict[str, pd.DataFrame]:
    if not src_zip.exists():
        raise FileNotFoundError(src_zip)
    raw = ROOT / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip) as z:
        z.extractall(raw)
    data = raw / "data"

    fred = pd.read_csv(data / "fred_macro_monthly.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    fred = merge_extra_outcome_data(fred, data_dir=data, balanced=True)
    fred = complete_monthly_calendar(fred)
    set_outcome_labels(outcome_labels_for_panel(fred))
    jk_m = pd.read_csv(data / "shocks_fed_jk_m.csv")
    jk_m["date"] = pd.to_datetime(dict(year=jk_m["year"].astype(int), month=jk_m["month"].astype(int), day=1))
    jk_m["MP_median_fallback"] = jk_m["MP_median"].fillna(jk_m["MP_pm"])
    jk_m["CBI_median_fallback"] = jk_m["CBI_median"].fillna(jk_m["CBI_pm"])
    jk_m["MP_used"] = jk_m["MP_median_fallback"]
    jk_m["CBI_used"] = jk_m["CBI_median_fallback"]
    jk_m["used_pm_fallback"] = jk_m["MP_median"].isna() | jk_m["CBI_median"].isna()
    jk_m["fallback_flag"] = jk_m["used_pm_fallback"]

    events = pd.read_csv(data / "shocks_fed_jk_t.csv")
    events["datetime"] = pd.to_datetime(events["start"])
    events["date"] = events["datetime"].dt.to_period("M").dt.to_timestamp()
    events["MP_event_used"] = events["MP_median"].fillna(events["MP_pm"])
    events["CBI_event_used"] = events["CBI_median"].fillna(events["CBI_pm"])
    events["fallback_event"] = events["MP_median"].isna() | events["CBI_median"].isna()
    event_agg = events.groupby("date", as_index=False).agg(
        MP_event_manual=("MP_event_used", "sum"),
        CBI_event_manual=("CBI_event_used", "sum"),
        MP_pm_sum=("MP_pm", "sum"),
        CBI_pm_sum=("CBI_pm", "sum"),
        fallback_event=("fallback_event", "max"),
        n_events=("MP_event_used", "size"),
        nonmissing_median_events=("MP_median", lambda x: int(x.notna().sum())),
        missing_median_events=("MP_median", lambda x: int(x.isna().sum())),
    )
    month_range = pd.DataFrame({"date": pd.date_range(jk_m["date"].min(), jk_m["date"].max(), freq="MS")})
    event_monthly = month_range.merge(event_agg, on="date", how="left")
    fill_cols = ["MP_event_manual", "CBI_event_manual", "MP_pm_sum", "CBI_pm_sum", "n_events", "nonmissing_median_events", "missing_median_events"]
    event_monthly[fill_cols] = event_monthly[fill_cols].fillna(0.0)
    event_monthly["fallback_event"] = event_monthly["fallback_event"].fillna(False).astype(bool)
    event_monthly["mixed_missing_and_nonmissing_events"] = (
        (event_monthly["missing_median_events"] > 0) & (event_monthly["nonmissing_median_events"] > 0)
    )

    panel = fred.merge(
        jk_m[["date", "MP_median_fallback", "CBI_median_fallback", "MP_pm", "CBI_pm", "MP_used", "CBI_used", "used_pm_fallback", "fallback_flag"]],
        on="date",
        how="left",
    ).merge(
        event_monthly[["date", "MP_event_manual", "CBI_event_manual", "n_events", "fallback_event", "mixed_missing_and_nonmissing_events"]],
        on="date",
        how="left",
    )
    panel = complete_monthly_calendar(panel)
    return {"panel": panel, "monthly": jk_m, "events": events, "event_monthly": event_monthly}


def build_lp_scores(
    panel: pd.DataFrame,
    shock_col: str,
    cbi_col: str | None,
    H: int = 24,
    L: int = 12,
    outcome_columns: tuple[str, ...] | None = None,
    sample_dates: pd.Series | np.ndarray | list[Any] | None = None,
) -> dict[str, Any]:
    panel = complete_monthly_calendar(panel)
    if outcome_columns is not None:
        panel = panel_with_outcomes(panel, tuple(outcome_columns))
    Ybase = build_outcome_frame(panel)
    set_outcome_labels(list(Ybase.columns))
    Yarr = Ybase.to_numpy(float)
    dY = np.vstack([np.full((1, Yarr.shape[1]), np.nan), np.diff(Yarr, axis=0)])
    mvals = pd.to_numeric(panel[shock_col], errors="coerce").to_numpy(float)
    mstd = mvals / np.nanstd(mvals)
    if cbi_col is not None:
        cvals = pd.to_numeric(panel[cbi_col], errors="coerce").to_numpy(float)
        cstd = cvals / np.nanstd(cvals)
    else:
        cstd = np.full(len(panel), np.nan)
    sample_months: set[pd.Timestamp] | None = None
    if sample_dates is not None:
        sample_months = set(pd.to_datetime(pd.Series(sample_dates)).dt.to_period("M").dt.to_timestamp())

    valid = []
    for t in range(len(panel)):
        if t - L < 0 or t - 1 < 0 or t + H >= len(panel):
            continue
        if sample_months is not None and panel["date"].iloc[t].to_period("M").to_timestamp() not in sample_months:
            continue
        checks = [
            np.isfinite(mstd[t]),
            np.isfinite(mstd[t - L : t]).all(),
            np.isfinite(Yarr[t - 1 : t + H + 1, :]).all(),
            np.isfinite(Yarr[t - L : t, :]).all(),
            np.isfinite(dY[t - L : t, :]).all(),
        ]
        if cbi_col is not None:
            checks += [
                np.isfinite(cstd[t]),
                np.isfinite(cstd[t - L : t]).all(),
            ]
        if all(checks):
            valid.append(t)
    valid = np.array(valid, dtype=int)
    dates = panel["date"].iloc[valid].reset_index(drop=True)

    controls = [np.ones(len(valid)), valid.astype(float)]
    if cbi_col is not None:
        controls.append(cstd[valid])
    for lag in range(1, L + 1):
        controls.append(mstd[valid - lag])
        if cbi_col is not None:
            controls.append(cstd[valid - lag])
    for lag in range(1, L + 1):
        controls += [Yarr[valid - lag, :], dY[valid - lag, :]]
    X = np.hstack([np.asarray(a)[:, None] if np.asarray(a).ndim == 1 else np.asarray(a) for a in controls])
    Yresp = np.hstack([Yarr[valid + hh, :] - Yarr[valid - 1, :] for hh in range(H + 1)])
    Xs = X.copy()
    muX = Xs[:, 1:].mean(axis=0)
    sdX = Xs[:, 1:].std(axis=0)
    sdX[sdX == 0] = 1.0
    Xs[:, 1:] = (Xs[:, 1:] - muX) / sdX
    Yall = np.column_stack([mstd[valid], Yresp])
    Q_scores, sigma_m2 = lp_scores_from_design(Xs, Yall)
    return {
        "Q_scores": Q_scores,
        "dates": dates,
        "valid_idx": valid,
        "mstd": mstd[valid],
        "cstd": cstd[valid],
        "sigma_m2": sigma_m2,
        "X_design": Xs,
        "Y_all": Yall,
        "outcome_labels": list(Ybase.columns),
        "outcome_columns": outcome_columns_for_labels(list(Ybase.columns)),
        "shock_col": shock_col,
        "cbi_col": cbi_col or "",
        "H": H,
        "L": L,
        "pvars": len(Ybase.columns),
    }


def date_range_text(dates: pd.Series | pd.Index | np.ndarray | list[Any]) -> str:
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if parsed.empty:
        return "not available"
    return f"{parsed.min().strftime('%Y-%m-%d')} to {parsed.max().strftime('%Y-%m-%d')}"


def finite_date_range_text(panel: pd.DataFrame, columns: list[str], require_all: bool = True) -> tuple[int, str]:
    present = [c for c in columns if c in panel.columns]
    if not present or "date" not in panel.columns:
        return 0, "not available"
    finite = pd.DataFrame({c: pd.to_numeric(panel[c], errors="coerce").notna() for c in present})
    mask = finite.all(axis=1) if require_all else finite.any(axis=1)
    if not mask.any():
        return 0, "not available"
    dates = pd.to_datetime(panel.loc[mask, "date"], errors="coerce").dropna()
    if dates.empty:
        return 0, "not available"
    return int(mask.sum()), date_range_text(dates)


def incomplete_dates_text(panel: pd.DataFrame, columns: list[str], max_show: int = 8) -> str:
    present = [c for c in columns if c in panel.columns]
    if not present or "date" not in panel.columns:
        return "not available"
    finite = pd.DataFrame({c: pd.to_numeric(panel[c], errors="coerce").notna() for c in present})
    dates = pd.to_datetime(panel["date"], errors="coerce")
    missing_dates = dates.loc[dates.notna() & ~finite.all(axis=1)]
    if missing_dates.empty:
        return "none"
    shown = [d.strftime("%Y-%m-%d") for d in missing_dates.head(max_show)]
    suffix = "" if len(missing_dates) <= max_show else f"; plus {len(missing_dates) - max_show} more"
    unit = "month" if len(missing_dates) == 1 else "months"
    return f"{', '.join(shown)} ({len(missing_dates)} {unit}{suffix})"


def sample_value(sample_coverage: pd.DataFrame, item: str) -> str:
    if sample_coverage is None or sample_coverage.empty or "item" not in sample_coverage or "value" not in sample_coverage:
        return ""
    values = sample_coverage.loc[sample_coverage["item"].eq(item), "value"]
    return "" if values.empty else str(values.iloc[0])


def build_publication_sample_coverage(panel: pd.DataFrame, scores: dict[str, Any]) -> pd.DataFrame:
    panel = complete_monthly_calendar(panel)
    dates = pd.to_datetime(scores["dates"]).reset_index(drop=True)
    valid_idx = np.asarray(scores.get("valid_idx", []), dtype=int)
    outcome_cols = list(scores.get("outcome_columns", outcome_columns_for_labels(list(scores.get("outcome_labels", OUTCOME_LABELS)))))
    labels = list(scores.get("outcome_labels", OUTCOME_LABELS))
    pvars = int(scores.get("pvars", len(labels)))
    h = int(scores.get("H", H))
    lag_count = int(scores.get("L", L))
    score_dim = int(np.asarray(scores["Q_scores"]).shape[1])
    outcome_n, outcome_range = finite_date_range_text(panel, outcome_cols, require_all=True)
    shock_col = str(scores.get("shock_col", ""))
    cbi_col = str(scores.get("cbi_col", ""))
    shock_n, shock_range = finite_date_range_text(panel, [shock_col], require_all=True) if shock_col else (0, "not available")
    cbi_n, cbi_range = finite_date_range_text(panel, [cbi_col], require_all=True) if cbi_col else (0, "not used")
    if len(valid_idx):
        raw_start_idx = max(int(valid_idx.min()) - lag_count - 1, 0)
        raw_end_idx = min(int(valid_idx.max()) + h, len(panel) - 1)
        raw_required = date_range_text(panel["date"].iloc[[raw_start_idx, raw_end_idx]])
    else:
        raw_required = "not available"
    complete_range = date_range_text(dates)
    rows = [
        {"item": "Merged publication panel calendar coverage", "value": f"{date_range_text(panel['date'])} ({len(panel)} monthly rows)"},
        {"item": "Headline outcomes finite coverage", "value": f"{outcome_range} ({outcome_n} rows with all {pvars} outcomes finite)"},
        {"item": "Headline outcome incomplete months", "value": incomplete_dates_text(panel, outcome_cols)},
        {"item": "Monetary-policy shock-date coverage", "value": f"{shock_range} ({shock_n} finite {shock_col} rows)"},
        {"item": "Control-shock coverage", "value": f"{cbi_range} ({cbi_n} finite {cbi_col} rows)" if cbi_col else cbi_range},
        {"item": "LP score observations", "value": len(dates)},
        {"item": "Score-surface dimension", "value": score_dim},
        {"item": "Coordinate grid", "value": f"{pvars} outcomes x {h + 1} horizons = {score_dim} coordinates"},
        {"item": "Common complete-coordinate score coverage", "value": complete_range},
        {"item": "Usable LP sample range", "value": complete_range},
        {"item": "State index attached to", "value": "base month t (the plotted date)"},
        {"item": "Raw outcome window consumed by score sample", "value": raw_required},
        {"item": "Horizons", "value": f"0 to {h} months"},
        {"item": "Lagged controls", "value": f"{lag_count} monthly lags of policy/control/outcomes and outcome differences"},
        {"item": "Outcomes", "value": ", ".join(labels)},
        {"item": "Shock", "value": shock_col},
        {"item": "Control shock", "value": cbi_col or "not used"},
        {"item": "Residualized shock variance", "value": scores.get("sigma_m2", np.nan)},
    ]
    return pd.DataFrame(rows)


def score_data_to_npz_arrays(score_data: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "variants_json": np.array(json.dumps(sorted(score_data.keys()))),
        "outcome_labels_json": np.array(json.dumps(OUTCOME_LABELS)),
    }
    for key, value in score_data.items():
        prefix = safe_name(key)
        arrays[f"{prefix}__Q_scores"] = np.asarray(value["Q_scores"], dtype=float)
        arrays[f"{prefix}__Q_scores_raw"] = np.asarray(value.get("Q_scores_raw", value["Q_scores"]), dtype=float)
        arrays[f"{prefix}__dates_ns"] = pd.to_datetime(value["dates"]).to_numpy(dtype="datetime64[ns]")
        arrays[f"{prefix}__valid_idx"] = np.asarray(value["valid_idx"], dtype=np.int64)
        arrays[f"{prefix}__mstd"] = np.asarray(value["mstd"], dtype=float)
        arrays[f"{prefix}__cstd"] = np.asarray(value["cstd"], dtype=float)
        arrays[f"{prefix}__sigma_m2"] = np.asarray([value["sigma_m2"]], dtype=float)
        arrays[f"{prefix}__X_design"] = np.asarray(value["X_design"], dtype=float)
        arrays[f"{prefix}__Y_all"] = np.asarray(value["Y_all"], dtype=float)
        arrays[f"{prefix}__outcome_labels_json"] = np.array(json.dumps(value.get("outcome_labels", OUTCOME_LABELS)))
        arrays[f"{prefix}__score_transform_json"] = np.array(json.dumps(value.get("score_transform", {"kind": "raw"})))
        arrays[f"{prefix}__shock_col"] = np.array(str(value.get("shock_col", "")))
        arrays[f"{prefix}__cbi_col"] = np.array(str(value.get("cbi_col", "")))
    return arrays


def score_data_from_npz_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    variants = json.loads(str(arrays["variants_json"]))
    if "outcome_labels_json" in arrays:
        set_outcome_labels(json.loads(str(arrays["outcome_labels_json"])))
    score_data: dict[str, dict[str, Any]] = {}
    for key in variants:
        prefix = safe_name(key)
        score_data[key] = {
            "Q_scores": np.asarray(arrays[f"{prefix}__Q_scores"], dtype=float),
            "Q_scores_raw": np.asarray(arrays.get(f"{prefix}__Q_scores_raw", arrays[f"{prefix}__Q_scores"]), dtype=float),
            "dates": pd.Series(pd.to_datetime(arrays[f"{prefix}__dates_ns"].astype("datetime64[ns]"))),
            "valid_idx": np.asarray(arrays[f"{prefix}__valid_idx"], dtype=np.int64),
            "mstd": np.asarray(arrays[f"{prefix}__mstd"], dtype=float),
            "cstd": np.asarray(arrays[f"{prefix}__cstd"], dtype=float),
            "sigma_m2": float(np.asarray(arrays[f"{prefix}__sigma_m2"])[0]),
            "X_design": np.asarray(arrays[f"{prefix}__X_design"], dtype=float),
            "Y_all": np.asarray(arrays[f"{prefix}__Y_all"], dtype=float),
            "outcome_labels": json.loads(str(arrays.get(f"{prefix}__outcome_labels_json", arrays.get("outcome_labels_json", np.array(json.dumps(OUTCOME_LABELS)))))),
            "score_transform": json.loads(str(arrays.get(f"{prefix}__score_transform_json", np.array(json.dumps({"kind": "raw"}))))),
            "shock_col": str(arrays.get(f"{prefix}__shock_col", np.array(""))),
            "cbi_col": str(arrays.get(f"{prefix}__cbi_col", np.array(""))),
        }
    return score_data


def bootstrap_stats_cache_key(data_hash: str, variant: str, scores: dict[str, Any]) -> str:
    labels = tuple(scores.get("outcome_labels", OUTCOME_LABELS))
    m_dim = int(np.asarray(scores["Q_scores"]).shape[1])
    return cache_key(
        "lp_bootstrap_sufficient_stats",
        data_hash,
        H,
        L,
        labels,
        m_dim,
        variant,
        scores.get("shock_col", ""),
        scores.get("cbi_col", ""),
        json.dumps(scores.get("score_transform", {"kind": "raw"}), sort_keys=True),
    )


def load_or_build_bootstrap_stats(data_hash: str, variant: str, scores: dict[str, Any]) -> tuple[dict[str, np.ndarray], bool]:
    key = bootstrap_stats_cache_key(data_hash, variant, scores)
    cached = None if DISABLE_CACHE else cache_get_npz(key)
    if cached is not None:
        return cached, True
    row_xx, row_xy = lp_row_sufficient_stats(scores["X_design"], scores["Y_all"])
    arrays = {
        "X_design": np.asarray(scores["X_design"], dtype=float),
        "Y_all": np.asarray(scores["Y_all"], dtype=float),
        "row_xx": row_xx,
        "row_xy": row_xy,
    }
    if not DISABLE_CACHE:
        cache_set_npz(key, arrays)
    return arrays, False


def _leading_eigensystem(K: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = min(K.shape[0], max(rank + 1, 10))
    eigvals, eigvecs = scipy_eigh(K, subset_by_index=[K.shape[0] - keep, K.shape[0] - 1])
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]
    for r in range(eigvecs.shape[1]):
        j = np.argmax(np.abs(eigvecs[:, r]))
        if eigvecs[j, r] < 0:
            eigvecs[:, r] *= -1
    trace_total = np.maximum(np.trace(K), 1e-12)
    shares = eigvals / trace_total
    return eigvals, eigvecs[:, :rank], shares


def covariance_basis(
    Q_scores: np.ndarray,
    rank: int,
    reference_weights: np.ndarray | None = None,
    estimator_mode: str = "log_spd_legacy",
    ridge_scale: float = ARITHMETIC_REFERENCE_RIDGE_SCALE,
) -> dict[str, np.ndarray | float | str]:
    Q = np.asarray(Q_scores, dtype=float)
    weights = normalize_reference_weights(len(Q), reference_weights)
    beta = weights @ Q
    if estimator_mode == "arithmetic_outer_product":
        C = sym((Q.T * weights) @ Q)
        eigvals, V, shares = _leading_eigensystem(C, rank)
        lam = eigvals[:rank]
        rho = float(max(ridge_scale, 0.0) * max(float(np.trace(C)) / max(C.shape[0], 1), 1e-12))
        denom = np.sqrt(np.maximum(lam + rho, 1e-12))
        whitening_map = V @ np.diag(1.0 / denom)
        unwhitening_map = V @ np.diag(denom)
        Z = Q @ whitening_map
        return {
            "beta": beta,
            "E": Q - beta,
            "eigvals": eigvals,
            "shares": shares,
            "V": V,
            "Z": Z,
            "reference_weights": weights,
            "reference_covariance": C,
            "whitening_ridge": rho,
            "whitening_map": whitening_map,
            "unwhitening_map": unwhitening_map,
            "estimator_mode": estimator_mode,
        }
    if estimator_mode != "log_spd_legacy":
        raise ValueError(f"Unknown estimator mode: {estimator_mode}")
    E = Q - beta
    K = sym((E.T * weights) @ E)
    eigvals, V, shares = _leading_eigensystem(K, rank)
    lam = eigvals[:rank]
    whitening_map = V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 1e-12)))
    Z = E @ whitening_map
    return {
        "beta": beta,
        "E": E,
        "eigvals": eigvals,
        "shares": shares,
        "V": V,
        "Z": Z,
        "reference_weights": weights,
        "reference_covariance": K,
        "whitening_ridge": 0.0,
        "whitening_map": whitening_map,
        "unwhitening_map": V @ np.diag(np.sqrt(np.maximum(lam, 1e-12))),
        "estimator_mode": estimator_mode,
    }


def log_observations_from_Z(Z: np.ndarray, alpha: float) -> np.ndarray:
    R = Z.shape[1]
    G = alpha * np.eye(R)[None, :, :] + (1.0 - alpha) * np.einsum("ti,tj->tij", Z, Z)
    Ginv = spd_invsqrt(G.mean(axis=0), eps=1e-10)
    Gnorm = Ginv @ G @ Ginv
    return svec_batch(batched_spd_log(Gnorm))


def arithmetic_outer_product_observations(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float)
    Y_mats = np.einsum("ti,tj->tij", U, U, optimize=True)
    return Y_mats, svec_batch(Y_mats)


def map_whitened_matrices_to_original(A: np.ndarray, unwhitening_map: np.ndarray) -> np.ndarray:
    B = np.asarray(unwhitening_map, dtype=float)
    return sym_last(np.einsum("mr,trs,ns->tmn", B, np.asarray(A, dtype=float), B, optimize=True))


def arithmetic_components_from_fits(
    total_fit: StateFit,
    mean_fit: StateFit,
    rank: int,
    unwhitening_map: np.ndarray,
) -> dict[str, np.ndarray]:
    total_w = arithmetic_matrix_series_from_state(total_fit.xs, rank)
    mu_w = np.asarray(mean_fit.xs, dtype=float)
    mean_w = np.einsum("ti,tj->tij", mu_w, mu_w, optimize=True)
    within_w = project_psd(total_w - mean_w, floor=0.0)
    total_w = project_psd(total_w, floor=0.0)
    mean_w = project_psd(mean_w, floor=0.0)
    return {
        "total_second_moment_whitened": total_w,
        "mean_component_whitened": mean_w,
        "within_covariance_whitened": within_w,
        "smoothed_mean_whitened": mu_w,
        "total_second_moment_original": map_whitened_matrices_to_original(total_w, unwhitening_map),
        "mean_component_original": map_whitened_matrices_to_original(mean_w, unwhitening_map),
        "within_covariance_original": map_whitened_matrices_to_original(within_w, unwhitening_map),
        "smoothed_mean_original": mu_w @ np.asarray(unwhitening_map, dtype=float).T,
    }


def full_coordinate_ridge(C_hat: np.ndarray, ridge_scale: float = ARITHMETIC_REFERENCE_RIDGE_SCALE) -> float:
    C = sym(np.asarray(C_hat, dtype=float))
    p = C.shape[0]
    avg_variance = max(float(np.trace(C)) / max(p, 1), 1e-12)
    return float(max(float(ridge_scale), 0.0) * avg_variance + 1e-12)


def full_coordinate_reference_objects(
    chi: np.ndarray,
    ridge_scale: float = ARITHMETIC_REFERENCE_RIDGE_SCALE,
) -> dict[str, np.ndarray | float]:
    scores = np.asarray(chi, dtype=float)
    if scores.ndim != 2:
        raise ValueError("chi must be a two-dimensional observation-by-coordinate array.")
    C_hat = sym((scores.T @ scores) / max(len(scores), 1))
    rho = full_coordinate_ridge(C_hat, ridge_scale=ridge_scale)
    D_rho = sym(C_hat + rho * np.eye(C_hat.shape[0]))
    eigvals, eigvecs = spd_eigh(C_hat, eps=0.0)
    d_vals = eigvals + rho
    if float(np.min(d_vals)) <= 0.0:
        raise ValueError("C_hat + rho I must be positive definite.")
    D_invsqrt = sym((eigvecs * (1.0 / np.sqrt(d_vals))[None, :]) @ eigvecs.T)
    d_rho = float(np.sum(eigvals / d_vals))
    if not np.isfinite(d_rho) or d_rho <= 0.0:
        raise ValueError(f"d_rho must be positive and finite; got {d_rho}.")
    return {
        "C_hat": C_hat,
        "D_rho": D_rho,
        "D_invsqrt": D_invsqrt,
        "rho": rho,
        "d_rho": d_rho,
    }


def path_graph_laplacian_for_dates(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates).reset_index(drop=True)
    month_index = dt.dt.year.to_numpy(int) * 12 + dt.dt.month.to_numpy(int)
    gaps = np.diff(month_index)
    if np.any(gaps <= 0):
        raise RuntimeError("Full-coordinate temporal-kernel dates must be strictly increasing.")
    T = len(dt)
    Lmat = np.zeros((T, T), dtype=float)
    edge_weights = 1.0 / np.maximum(gaps.astype(float), 1.0) ** 2
    for i, w in enumerate(edge_weights):
        Lmat[i, i] += w
        Lmat[i + 1, i + 1] += w
        Lmat[i, i + 1] -= w
        Lmat[i + 1, i] -= w
    return sym(Lmat)


def doubly_stochastic_symmetric(W: np.ndarray, tol: float = 1e-12, max_iter: int = 1000) -> np.ndarray:
    out = sym(np.asarray(W, dtype=float))
    tiny_negative = (out < 0.0) & (out >= -tol)
    out[tiny_negative] = 0.0
    if float(out.min()) < -tol:
        raise ValueError(f"Temporal weights must be nonnegative; min weight is {out.min():.6g}.")
    for _ in range(max_iter):
        row_sum = out.sum(axis=1)
        if np.all(np.abs(row_sum - 1.0) <= tol):
            return sym(out)
        if np.any(row_sum <= 0.0):
            raise ValueError("Temporal weights contain a zero-mass row.")
        scale = 1.0 / np.sqrt(row_sum)
        out = scale[:, None] * out * scale[None, :]
    row_sum = out.sum(axis=1)
    if np.max(np.abs(row_sum - 1.0)) > 1e-8:
        raise RuntimeError("Temporal weights failed to converge to a symmetric stochastic matrix.")
    return sym(out)


def full_coordinate_temporal_weights(dates: pd.Series, eta: float = FULL_COORDINATE_KERNEL_ETA) -> np.ndarray:
    T = len(dates)
    if T == 0:
        raise ValueError("Cannot build temporal weights for an empty date path.")
    eta = float(eta)
    if eta <= 0.0:
        return np.eye(T)
    Lmat = path_graph_laplacian_for_dates(dates)
    W = eta * np.linalg.solve(eta * np.eye(T) + Lmat, np.eye(T))
    return doubly_stochastic_symmetric(W)


def full_coordinate_K_from_weights(chi: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.asarray(chi, dtype=float)
    W = np.asarray(weights, dtype=float)
    if W.shape != (len(scores), len(scores)):
        raise ValueError("Temporal weight matrix shape does not match chi observations.")
    outer = np.einsum("ti,tj->tij", scores, scores, optimize=True)
    return sym_last(np.einsum("ts,sij->tij", W, outer, optimize=True))


def full_coordinate_A_from_K(K_hat: np.ndarray, D_invsqrt: np.ndarray) -> np.ndarray:
    Dm = np.asarray(D_invsqrt, dtype=float)
    return sym_last(np.einsum("ij,tjk,kl->til", Dm, np.asarray(K_hat, dtype=float), Dm, optimize=True))


def full_coordinate_tau_soft(K_hat: np.ndarray, D_rho: np.ndarray, d_rho: float) -> np.ndarray:
    cD, lowerD = cho_factor(sym(D_rho), lower=True, check_finite=False)
    out = np.empty(K_hat.shape[0], dtype=float)
    for t, Kt in enumerate(np.asarray(K_hat, dtype=float)):
        solved = cho_solve((cD, lowerD), sym(Kt), check_finite=False)
        out[t] = float(np.trace(solved) / d_rho)
    return out


def full_coordinate_cell_probes(
    K_hat: np.ndarray,
    C_hat: np.ndarray,
    tau_soft: np.ndarray,
    tol: float = FULL_COORDINATE_CELL_VARIANCE_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag_C = np.diag(sym(C_hat))
    scale = max(float(np.nanmean(np.maximum(diag_C, 0.0))), 1.0)
    threshold = max(float(tol), float(tol) * scale)
    low = diag_C <= threshold
    diag_K = np.diagonal(np.asarray(K_hat, dtype=float), axis1=1, axis2=2)
    amp = np.full_like(diag_K, np.nan, dtype=float)
    amp[:, ~low] = diag_K[:, ~low] / diag_C[None, ~low]
    shape = np.full_like(amp, np.nan, dtype=float)
    valid = np.isfinite(amp) & (amp > 0.0) & (tau_soft[:, None] > 0.0)
    shape[valid] = np.log(amp[valid] / np.repeat(tau_soft[:, None], amp.shape[1], axis=1)[valid])
    return amp, shape, low


def estimate_full_coordinate_kernel_model(
    chi: np.ndarray,
    dates: pd.Series,
    variant: str,
    label: str,
    outcome_labels: list[str],
    ridge_scale: float = ARITHMETIC_REFERENCE_RIDGE_SCALE,
    kernel_eta: float = FULL_COORDINATE_KERNEL_ETA,
) -> FullCoordinateResult:
    scores = np.asarray(chi, dtype=float)
    ref = full_coordinate_reference_objects(scores, ridge_scale=ridge_scale)
    W = full_coordinate_temporal_weights(dates, eta=kernel_eta)
    K_hat = full_coordinate_K_from_weights(scores, W)
    A_hat = full_coordinate_A_from_K(K_hat, np.asarray(ref["D_invsqrt"], dtype=float))
    tau = full_coordinate_tau_soft(K_hat, np.asarray(ref["D_rho"], dtype=float), float(ref["d_rho"]))
    cell_amp, cell_shape, low_cells = full_coordinate_cell_probes(
        K_hat,
        np.asarray(ref["C_hat"], dtype=float),
        tau,
    )
    return FullCoordinateResult(
        variant=variant,
        label=label,
        dates=pd.Series(dates).reset_index(drop=True),
        outcome_labels=list(outcome_labels),
        chi=scores,
        C_hat=np.asarray(ref["C_hat"], dtype=float),
        D_rho=np.asarray(ref["D_rho"], dtype=float),
        D_invsqrt=np.asarray(ref["D_invsqrt"], dtype=float),
        rho=float(ref["rho"]),
        d_rho=float(ref["d_rho"]),
        temporal_weights=W,
        K_hat=K_hat,
        A_hat=A_hat,
        tau_soft=tau,
        cell_amp=cell_amp,
        cell_shape=cell_shape,
        low_variance_cell_mask=low_cells,
        kernel_eta=float(kernel_eta),
    )


def full_coordinate_block_bootstrap_tau_draws(
    chi: np.ndarray,
    dates: pd.Series,
    n_draws: int,
    seed: int,
    block_len: int,
    ridge_scale: float = ARITHMETIC_REFERENCE_RIDGE_SCALE,
    kernel_eta: float = FULL_COORDINATE_KERNEL_ETA,
) -> np.ndarray:
    scores = np.asarray(chi, dtype=float)
    T = len(scores)
    draws = max(int(n_draws), 1)
    rng = np.random.default_rng(seed)
    W = full_coordinate_temporal_weights(dates, eta=kernel_eta)
    out = np.empty((draws, T), dtype=float)
    for b in range(draws):
        ix = circular_block_indices(T, block_len, rng)
        boot = scores[ix]
        ref = full_coordinate_reference_objects(boot, ridge_scale=ridge_scale)
        cD, lowerD = cho_factor(np.asarray(ref["D_rho"], dtype=float), lower=True, check_finite=False)
        solved_chi = cho_solve((cD, lowerD), boot.T, check_finite=False).T
        quadratic = np.einsum("ti,ti->t", boot, solved_chi, optimize=True)
        out[b] = (W @ quadratic) / float(ref["d_rho"])
    return out


def full_coordinate_shape_metrics(result: FullCoordinateResult, H: int, pvars: int, labels: list[str]) -> dict[str, Any]:
    cell_amp = np.asarray(result.cell_amp, dtype=float)
    tau = np.asarray(result.tau_soft, dtype=float)
    cell_shape = np.asarray(result.cell_shape, dtype=float)
    positive_amp = np.where(np.isfinite(cell_amp) & (cell_amp > 0.0), cell_amp, 0.0)
    metrics = surface_shape_metrics_from_diag(positive_amp, H, pvars, labels)
    centered = np.where(np.isfinite(cell_shape), cell_shape, 0.0)
    valid = np.isfinite(cell_shape)
    denom = np.maximum(valid.sum(axis=1), 1)
    metrics["full_coordinate_shape_rms_log_relative"] = np.sqrt(np.sum(centered**2, axis=1) / denom)
    metrics["finite_working_grid_cell_effective_support"] = normalized_effective_support(positive_amp, axis=1)
    metrics["tau_soft"] = tau
    return {
        "cell_amp": cell_amp,
        "cell_shape": cell_shape,
        "metrics": metrics,
        "shape_rms": metrics["full_coordinate_shape_rms_log_relative"],
    }


def coordinate_indices_for_variables(H: int, pvars: int, variable_indices: list[int]) -> np.ndarray:
    idx = [h * pvars + j for h in range(H + 1) for j in variable_indices]
    return np.asarray(idx, dtype=int)


def coordinate_indices_for_horizons(H: int, pvars: int, horizon_mask: np.ndarray) -> np.ndarray:
    idx = [h * pvars + j for h in np.flatnonzero(horizon_mask) for j in range(pvars)]
    return np.asarray(idx, dtype=int)


def full_coordinate_block_shape_paths(
    result: FullCoordinateResult,
    H: int,
    pvars: int,
    labels: list[str],
) -> pd.DataFrame:
    dates = pd.to_datetime(result.dates)
    diag_C = np.diag(result.C_hat)
    diag_K = np.diagonal(result.K_hat, axis1=1, axis2=2)
    macro_idx, financial_idx = outcome_group_indices(labels)
    horizons = np.arange(H + 1)
    selectors: list[tuple[str, np.ndarray]] = []
    if macro_idx:
        selectors.append(("macro_outcomes", coordinate_indices_for_variables(H, pvars, macro_idx)))
    if financial_idx:
        selectors.append(("financial_outcomes", coordinate_indices_for_variables(H, pvars, financial_idx)))
    horizon_specs = [
        ("horizons_00_03", horizons <= min(3, H)),
        ("horizons_04_12", (horizons >= 4) & (horizons <= min(12, H))),
        ("horizons_13_24", horizons >= 13),
    ]
    for name, mask in horizon_specs:
        idx = coordinate_indices_for_horizons(H, pvars, mask)
        if len(idx):
            selectors.append((name, idx))
    rows: list[dict[str, Any]] = []
    for name, idx in selectors:
        idx = np.asarray(idx, dtype=int)
        denom = float(np.sum(diag_C[idx]))
        if denom <= 0.0:
            raise ValueError(f"Block-probe denominator is non-positive for {name}: {denom}")
        block_amp = np.sum(diag_K[:, idx], axis=1) / denom
        block_shape = block_amp / np.maximum(result.tau_soft, 1e-12)
        for t, date in enumerate(dates):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "block": name,
                    "block_amp": float(block_amp[t]),
                    "tau_soft": float(result.tau_soft[t]),
                    "block_shape": float(block_shape[t]),
                    "benchmark": 1.0,
                    "denominator": denom,
                    "n_coordinates": int(len(idx)),
                }
            )
    return pd.DataFrame(rows)


def selected_full_coordinate_episodes(dates: pd.Series, tau: np.ndarray, shape_rms: np.ndarray, march_idx: int | None) -> list[dict[str, Any]]:
    log_tau = np.log(np.maximum(np.asarray(tau, dtype=float), 1e-12))
    neutral_score = np.hypot(log_tau, np.asarray(shape_rms, dtype=float))
    candidates = [
        ("reference_neutral_full_coordinate", int(np.nanargmin(neutral_score))),
        ("max_tau_soft", int(np.nanargmax(tau))),
        ("max_full_coordinate_shape_dispersion", int(np.nanargmax(shape_rms))),
    ]
    if march_idx is not None:
        candidates.append(("march_2020", int(march_idx)))
    episodes: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    for label, idx in candidates:
        if idx in seen:
            episodes[seen[idx]]["episode"] += f"+{label}"
            continue
        seen[idx] = len(episodes)
        episodes.append(
            {
                "episode": label,
                "idx": idx,
                "date": pd.to_datetime(dates.iloc[idx]).strftime("%Y-%m"),
                "tau_soft": float(tau[idx]),
                "full_coordinate_shape_rms_log_relative": float(shape_rms[idx]),
            }
        )
    return episodes


def write_full_coordinate_npz(result: FullCoordinateResult, path: Path) -> None:
    np.savez_compressed(
        path,
        chi=np.asarray(result.chi, dtype=float),
        C_hat=np.asarray(result.C_hat, dtype=float),
        D_rho=np.asarray(result.D_rho, dtype=float),
        D_invsqrt=np.asarray(result.D_invsqrt, dtype=float),
        K_hat=np.asarray(result.K_hat, dtype=float),
        A_hat=np.asarray(result.A_hat, dtype=float),
        tau_soft=np.asarray(result.tau_soft, dtype=float),
        temporal_weights=np.asarray(result.temporal_weights, dtype=float),
        cell_amp=np.asarray(result.cell_amp, dtype=float),
        cell_shape=np.asarray(result.cell_shape, dtype=float),
        low_variance_cell_mask=np.asarray(result.low_variance_cell_mask, dtype=bool),
        rho=np.asarray([result.rho], dtype=float),
        d_rho=np.asarray([result.d_rho], dtype=float),
        kernel_eta=np.asarray([result.kernel_eta], dtype=float),
    )


def build_full_coordinate_section31_outputs(
    scores: dict[str, Any],
    variant: str,
    label: str,
    dates: pd.Series,
    outcome_labels: list[str],
    march_idx: int | None,
    bootstrap_draws: int = B_BOOT,
) -> dict[str, Any]:
    result = estimate_full_coordinate_kernel_model(
        scores["Q_scores"],
        dates,
        variant=variant,
        label=label,
        outcome_labels=outcome_labels,
    )
    tau_draws = full_coordinate_block_bootstrap_tau_draws(
        result.chi,
        result.dates,
        n_draws=bootstrap_draws,
        seed=BOOT_SEED,
        block_len=BOOT_BLOCK_LEN,
        kernel_eta=result.kernel_eta,
    )
    tau_band = positive_simultaneous_band(tau_draws, result.tau_soft, level=0.90)
    pvars = len(outcome_labels)
    shape = full_coordinate_shape_metrics(result, H, pvars, outcome_labels)
    metrics_data: dict[str, Any] = {"date": pd.to_datetime(result.dates).dt.strftime("%Y-%m-%d")}
    for name, values in shape["metrics"].items():
        metrics_data[name] = values
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(TABLES / "publication_grade_full_coordinate_shape_metrics.csv", index=False)

    horizons = np.repeat(np.arange(H + 1), pvars)
    variables = np.tile(np.array(outcome_labels), H + 1)
    date_str = pd.to_datetime(result.dates).dt.strftime("%Y-%m-%d").to_numpy()
    low_flags = np.tile(result.low_variance_cell_mask, len(result.dates))
    alloc_df = pd.DataFrame(
        {
            "date": np.repeat(date_str, result.chi.shape[1]),
            "horizon_months": np.tile(horizons, len(result.dates)),
            "variable": np.tile(variables, len(result.dates)),
            "cell_amp": result.cell_amp.ravel(),
            "cell_shape_log_amp_over_tau_soft": result.cell_shape.ravel(),
            "tau_soft": np.repeat(result.tau_soft, result.chi.shape[1]),
            "low_reference_variance_flag": low_flags,
            "reference_variance_tolerance": FULL_COORDINATE_CELL_VARIANCE_TOL,
        }
    )
    alloc_df.to_csv(TABLES / "publication_grade_full_coordinate_cell_shape_allocations.csv", index=False)

    block_df = full_coordinate_block_shape_paths(result, H, pvars, outcome_labels)
    block_df.to_csv(TABLES / "publication_grade_full_coordinate_block_shape_paths.csv", index=False)

    concentration_cols = [
        "date",
        "cell_effective_support",
        "variable_effective_support",
        "horizon_effective_support",
        "finite_working_grid_cell_effective_support",
    ]
    concentration_df = metrics_df[[c for c in concentration_cols if c in metrics_df.columns]].copy()
    concentration_df.to_csv(TABLES / "publication_grade_full_coordinate_concentration.csv", index=False)

    episodes = selected_full_coordinate_episodes(result.dates, result.tau_soft, shape["shape_rms"], march_idx)
    pd.DataFrame(episodes).drop(columns=["idx"]).to_csv(TABLES / "publication_grade_full_coordinate_shape_episodes.csv", index=False)

    path_df = pd.DataFrame(
        {
            "date": pd.to_datetime(result.dates).dt.strftime("%Y-%m-%d"),
            "tau_soft": result.tau_soft,
            "tau": result.tau_soft,
            "tau_point_p05": tau_band["point_low"],
            "tau_point_median": tau_band["point_med"],
            "tau_point_p95": tau_band["point_high"],
            "tau_simul_p05": tau_band["sim_low"],
            "tau_simul_p95": tau_band["sim_high"],
            "tau_full_pipeline_p05": tau_band["point_low"],
            "tau_full_pipeline_p95": tau_band["point_high"],
            "tau_full_pipeline_simul_p05": tau_band["sim_low"],
            "tau_full_pipeline_simul_p95": tau_band["sim_high"],
            "scale_log_tau": np.log(np.maximum(result.tau_soft, 1e-12)),
            "shape_distance": shape["shape_rms"],
            "full_coordinate_shape_rms_log_relative": shape["shape_rms"],
            "backend": result.backend,
            "rho": result.rho,
            "d_rho": result.d_rho,
            "kernel_eta": result.kernel_eta,
        }
    )
    for name in ["macro_variable_share", "financial_variable_share", "short_horizon_share", "medium_horizon_share", "long_horizon_share"]:
        if name in metrics_df:
            path_df[name] = metrics_df[name]
    path_df.to_csv(TABLES / "publication_grade_headline_state_path.csv", index=False)

    min_eig_C = float(np.linalg.eigvalsh(result.C_hat).min())
    min_eig_K = float(np.linalg.eigvalsh(result.K_hat).min())
    min_eig_A = float(np.linalg.eigvalsh(result.A_hat).min())
    mean_K = sym(np.mean(result.K_hat, axis=0))
    mean_tau = float(np.mean(result.tau_soft))
    mean_tau_error = abs(mean_tau - 1.0)
    diagnostics = pd.DataFrame(
        [
            {"check": "C_hat_symmetric", "value": float(np.max(np.abs(result.C_hat - result.C_hat.T))), "tolerance": 1e-10},
            {"check": "K_hat_symmetric", "value": float(np.max(np.abs(result.K_hat - np.swapaxes(result.K_hat, 1, 2)))), "tolerance": 1e-10},
            {"check": "C_hat_min_eigenvalue", "value": min_eig_C, "tolerance": -1e-8},
            {"check": "K_hat_min_eigenvalue", "value": min_eig_K, "tolerance": -1e-8},
            {"check": "A_hat_min_eigenvalue", "value": min_eig_A, "tolerance": -1e-8},
            {"check": "d_rho_positive", "value": float(result.d_rho), "tolerance": 0.0},
            {"check": "mean_K_matches_C_hat_fro_relative", "value": float(np.linalg.norm(mean_K - result.C_hat, ord="fro") / max(np.linalg.norm(result.C_hat, ord="fro"), 1e-12)), "tolerance": 1e-8},
            {"check": "mean_tau_soft_equals_one_when_mean_K_matches_C_hat", "value": mean_tau_error, "tolerance": 1e-8},
            {"check": "temporal_weight_row_sum_error", "value": float(np.max(np.abs(result.temporal_weights.sum(axis=1) - 1.0))), "tolerance": 1e-8},
            {"check": "temporal_weight_column_sum_error", "value": float(np.max(np.abs(result.temporal_weights.sum(axis=0) - 1.0))), "tolerance": 1e-8},
            {"check": "low_reference_variance_cell_count", "value": int(np.sum(result.low_variance_cell_mask)), "tolerance": 0},
        ]
    )
    diagnostics.to_csv(TABLES / "publication_grade_full_coordinate_diagnostics.csv", index=False)

    uncertainty = pd.DataFrame(
        [
            {"quantity": "full_coordinate_backend", "value": result.backend},
            {"quantity": "full_coordinate_dimension_p", "value": int(result.chi.shape[1])},
            {"quantity": "ridge_rho", "value": float(result.rho)},
            {"quantity": "soft_effective_dimension_d_rho", "value": float(result.d_rho)},
            {"quantity": "kernel_eta", "value": float(result.kernel_eta)},
            {"quantity": "block_bootstrap_draws", "value": int(max(bootstrap_draws, 1))},
            {"quantity": "bootstrap_block_length_months", "value": int(BOOT_BLOCK_LEN)},
            {"quantity": "tau_log_simultaneous_90_sup_crit", "value": tau_band["sup_crit"]},
            {"quantity": "headline_tau_soft_max", "value": float(result.tau_soft.max())},
            {"quantity": "headline_tau_soft_max_month", "value": pd.to_datetime(result.dates.iloc[int(np.argmax(result.tau_soft))]).strftime("%Y-%m")},
            {"quantity": "headline_full_coordinate_shape_max", "value": float(np.max(shape["shape_rms"]))},
            {"quantity": "headline_full_coordinate_shape_max_month", "value": pd.to_datetime(result.dates.iloc[int(np.argmax(shape["shape_rms"]))]).strftime("%Y-%m")},
            {"quantity": "low_reference_variance_cell_count", "value": int(np.sum(result.low_variance_cell_mask))},
        ]
    )
    uncertainty.to_csv(TABLES / "publication_grade_uncertainty_summary.csv", index=False)

    write_full_coordinate_npz(result, TABLES / "publication_grade_full_coordinate_covariance_components.npz")
    return {
        "result": result,
        "tau_draws": tau_draws,
        "tau_band": tau_band,
        "shape": shape,
        "metrics_df": metrics_df,
        "alloc_df": alloc_df,
        "block_df": block_df,
        "concentration_df": concentration_df,
        "episodes": episodes,
        "path_df": path_df,
        "diagnostics": diagnostics,
        "uncertainty": uncertainty,
    }


# -----------------------------
# Robust state-space estimation
# -----------------------------
def shrink_transition(F: np.ndarray, rank: int, cross_shrink: float = 0.15, diag_shrink: float = 0.10, max_radius: float = 0.965) -> tuple[np.ndarray, dict[str, float]]:
    d = F.shape[0]
    mask = np.zeros_like(F)
    mask[:rank, :rank] = 1.0
    if d > rank:
        mask[rank:, rank:] = 1.0
    F_block = F * mask + cross_shrink * F * (1.0 - mask)
    F_diag = np.diag(np.diag(F_block))
    F_shrunk = (1.0 - diag_shrink) * F_block + diag_shrink * F_diag
    eig = np.linalg.eigvals(F_shrunk)
    rad0 = float(np.max(np.abs(eig))) if len(eig) else 0.0
    scale = 1.0
    if rad0 > max_radius:
        scale = max_radius / rad0
        F_shrunk *= scale
    return F_shrunk, {"cross_shrink": cross_shrink, "diag_shrink": diag_shrink, "pre_cap_radius": rad0, "radius_scale": scale}


def initial_state_matrices(
    Y: np.ndarray,
    rank: int,
    ridge: float = 0.5,
    observation_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    ow = objective_weight_scale(observation_weights, len(Y))
    mu = weighted_mean(Y, ow)
    X0 = Y[:-1] - mu
    Y1 = Y[1:] - mu
    tw = ow[1:]
    X0w = X0 * np.sqrt(tw)[:, None]
    Y1w = Y1 * np.sqrt(tw)[:, None]
    F_T = np.linalg.solve(X0w.T @ X0w + ridge * np.eye(Y.shape[1]), X0w.T @ Y1w)
    F, shrink_meta = shrink_transition(F_T.T, rank)
    resid = Y1 - X0 @ F.T
    Sigma = weighted_covariance(resid, tw, 1e-8)
    Q = stabilize_cov(0.35 * Sigma + 1e-5 * np.eye(Y.shape[1]), 1e-8)
    Rmat = stabilize_cov(0.65 * Sigma + 1e-5 * np.eye(Y.shape[1]), 1e-8)
    return mu, F, Q, Rmat, shrink_meta


def stationary_state_cov(F: np.ndarray, Q: np.ndarray, n_iter: int = 250) -> np.ndarray:
    try:
        return stabilize_cov(solve_discrete_lyapunov(F, Q), 1e-10)
    except Exception:
        P = stabilize_cov(Q, 1e-10)
        for _ in range(n_iter):
            P_next = stabilize_cov(F @ P @ F.T + Q, 1e-10)
            if np.linalg.norm(P_next - P, ord="fro") < 1e-10:
                return P_next
            P = P_next
        return P


def kalman_smoother_identity_robust(
    Y: np.ndarray,
    mu: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    Rmat: np.ndarray,
    nu: float | None = None,
    min_weight: float = MIN_STUDENT_WEIGHT,
    observation_weights: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    Y = np.asarray(Y, float)
    T, d = Y.shape
    obs_scale = objective_weight_scale(observation_weights, T)
    I = np.eye(d)
    xf = np.zeros((T, d))
    Pf = np.zeros((T, d, d))
    xp = np.zeros((T, d))
    Pp = np.zeros((T, d, d))
    R_eff = np.zeros((T, d, d))
    weights = np.ones(T)
    P0 = stationary_state_cov(F, Q)
    xp[0] = mu
    Pp[0] = P0
    loglik = 0.0
    robust_loglik = 0.0
    for t in range(T):
        if t > 0:
            xp[t] = mu + F @ (xf[t - 1] - mu)
            Pp[t] = stabilize_cov(F @ Pf[t - 1] @ F.T + Q, 1e-10)
        v0 = Y[t] - xp[t]
        S0 = stabilize_cov(Pp[t] + Rmat, 1e-10)
        c0, lower0 = cho_factor(S0, lower=True, check_finite=False)
        if nu is not None and np.isfinite(nu):
            maha = float(v0 @ cho_solve((c0, lower0), v0, check_finite=False))
            w = min(1.0, max(min_weight, (nu + d) / (nu + maha)))
        else:
            w = 1.0
        weights[t] = w
        combined_weight = max(float(w * obs_scale[t]), 1e-12)
        Rt = stabilize_cov(Rmat / combined_weight, 1e-10)
        R_eff[t] = Rt
        S = stabilize_cov(Pp[t] + Rt, 1e-10)
        cS, lowerS = cho_factor(S, lower=True, check_finite=False)
        K = cho_solve((cS, lowerS), Pp[t].T, check_finite=False).T
        v = Y[t] - xp[t]
        xf[t] = xp[t] + K @ v
        Pf[t] = stabilize_cov((I - K) @ Pp[t] @ (I - K).T + K @ Rt @ K.T, 1e-12)
        loglik += float(obs_scale[t]) * gaussian_logpdf(v, S)
        robust_loglik += float(obs_scale[t]) * (student_t_logpdf(v, S0, nu) if nu is not None else gaussian_logpdf(v, S0))
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        Pnext = stabilize_cov(Pp[t + 1], 1e-10)
        c_next, lower_next = cho_factor(Pnext, lower=True, check_finite=False)
        J = cho_solve((c_next, lower_next), (Pf[t] @ F.T).T, check_finite=False).T
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = stabilize_cov(Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T, 1e-12)
    return {
        "xs": xs,
        "Ps": Ps,
        "xf": xf,
        "Pf": Pf,
        "xp": xp,
        "Pp": Pp,
        "R_eff": R_eff,
        "weights": weights,
        "objective_weights": obs_scale,
        "loglik": float(loglik),
        "robust_loglik": float(robust_loglik),
    }


def fit_em_state_space(
    Y: np.ndarray,
    rank: int,
    em_iters: int,
    nu: float,
    observation_weights: np.ndarray | None = None,
    estimator_mode: str = "log_spd_legacy",
    observation_kind: str = "log_spd",
) -> StateFit:
    ow = objective_weight_scale(observation_weights, len(Y))
    mu, F, Q, Rmat, shrink_meta = initial_state_matrices(Y, rank, observation_weights=ow)
    base_sigma = weighted_covariance(Y, ow, 1e-8)
    process_floor = 0.08 * base_sigma
    measurement_floor = 0.01 * base_sigma
    out = kalman_smoother_identity_robust(Y, mu, F, Q, Rmat, nu=nu, observation_weights=ow)
    for _ in range(em_iters):
        xs = np.asarray(out["xs"])
        Ps = np.asarray(out["Ps"])
        weights = np.asarray(out["weights"])
        mu = weighted_mean(xs, ow)
        X0 = xs[:-1] - mu
        X1 = xs[1:] - mu
        tw = ow[1:]
        denom = (X0.T * tw) @ X0 + np.einsum("t,tij->ij", tw, Ps[:-1], optimize=True) + 0.5 * np.eye(Y.shape[1])
        numer = (X1.T * tw) @ X0
        F_full = numer @ np.linalg.inv(stabilize_cov(denom, 1e-8))
        F, shrink_meta = shrink_transition(F_full, rank)
        state_resid = X1 - X0 @ F.T
        Q_est = (state_resid.T * tw) @ state_resid / max(float(tw.sum()), 1e-12) + 0.05 * np.average(Ps[1:], axis=0, weights=tw)
        Q = stabilize_cov(Q_est + process_floor, 1e-8)
        obs_resid = Y - xs
        w = (weights * ow) / np.maximum(np.mean(weights * ow), 1e-12)
        R_est = (obs_resid.T * w) @ obs_resid / max(float(w.sum()), 1e-12) + 0.05 * np.average(Ps, axis=0, weights=ow)
        Rmat = stabilize_cov(R_est + measurement_floor, 1e-8)
        out = kalman_smoother_identity_robust(Y, mu, F, Q, Rmat, nu=nu, observation_weights=ow)
    rad = float(np.max(np.abs(np.linalg.eigvals(F))))
    return StateFit(
        alpha=np.nan,
        ylog=Y,
        mu=mu,
        F=F,
        Q=Q,
        Rmat=Rmat,
        xs=np.asarray(out["xs"]),
        Ps=np.asarray(out["Ps"]),
        xf=np.asarray(out["xf"]),
        Pf=np.asarray(out["Pf"]),
        xp=np.asarray(out["xp"]),
        Pp=np.asarray(out["Pp"]),
        weights=np.asarray(out["weights"]),
        loglik=float(out["loglik"]),
        robust_loglik=float(out["robust_loglik"]),
        factor_log_score=np.nan,
        spectral_radius=rad,
        transition_shrinkage=shrink_meta,
        estimator_mode=estimator_mode,
        observation_kind=observation_kind,
        objective_weights=np.asarray(out["objective_weights"]),
    )


def factor_predictive_log_score(Z: np.ndarray, fit: StateFit, rank: int, nu: float, estimator_mode: str = "log_spd_legacy") -> float:
    A_pred = matrix_series_from_state(fit.xp, rank, estimator_mode=estimator_mode)
    return float(sum(student_t_logpdf(Z[t], A_pred[t], nu) for t in range(len(Z))))


def estimate_alpha_and_state(
    Z: np.ndarray,
    rank: int,
    em_iters: int,
    nu: float,
    alpha_grid: np.ndarray = ALPHA_GRID,
    reference_weights: np.ndarray | None = None,
) -> StateFit:
    best_fit: StateFit | None = None
    best_alpha = None
    best_score = -np.inf
    for alpha in alpha_grid:
        Y = log_observations_from_Z(Z, float(alpha))
        fit = fit_em_state_space(
            Y,
            rank,
            em_iters=max(2, em_iters // 2),
            nu=nu,
            observation_weights=reference_weights,
            estimator_mode="log_spd_legacy",
            observation_kind="log_spd",
        )
        score = factor_predictive_log_score(Z, fit, rank, nu)
        if best_fit is None or score > best_score:
            best_fit = fit
            best_alpha = float(alpha)
            best_score = score
    assert best_alpha is not None
    Y = log_observations_from_Z(Z, best_alpha)
    fit = fit_em_state_space(
        Y,
        rank,
        em_iters=em_iters,
        nu=nu,
        observation_weights=reference_weights,
        estimator_mode="log_spd_legacy",
        observation_kind="log_spd",
    )
    fit.alpha = best_alpha
    fit.factor_log_score = factor_predictive_log_score(Z, fit, rank, nu)
    return fit


def estimate_rank_model(
    Q_scores: np.ndarray,
    dates: pd.Series,
    variant: str,
    label: str,
    rank: int,
    em_iters: int = EM_ITERS,
    outcome_labels: list[str] | None = None,
    reference_weights: np.ndarray | None = None,
    estimator_mode: str = ESTIMATOR_MODE,
) -> RankResult:
    if estimator_mode not in VALID_ESTIMATOR_MODES:
        raise ValueError(f"Unknown estimator mode: {estimator_mode}")
    Q_array = np.asarray(Q_scores, dtype=float)
    basis = covariance_basis(Q_array, rank, reference_weights=reference_weights, estimator_mode=estimator_mode)
    weights = np.asarray(basis["reference_weights"], dtype=float)
    if estimator_mode == "arithmetic_outer_product":
        _, y = arithmetic_outer_product_observations(np.asarray(basis["Z"], dtype=float))
        fit = fit_em_state_space(
            y,
            rank=rank,
            em_iters=em_iters,
            nu=ROBUST_NU,
            observation_weights=weights,
            estimator_mode=estimator_mode,
            observation_kind="arithmetic_outer_product",
        )
        fit.alpha = np.nan
        fit.factor_log_score = fit.robust_loglik
        mean_fit = fit_em_state_space(
            np.asarray(basis["Z"], dtype=float),
            rank=rank,
            em_iters=em_iters,
            nu=ROBUST_NU,
            observation_weights=weights,
            estimator_mode=estimator_mode,
            observation_kind="arithmetic_mean",
        )
        mean_fit.alpha = np.nan
        mean_fit.factor_log_score = mean_fit.robust_loglik
        components = arithmetic_components_from_fits(
            fit,
            mean_fit,
            rank,
            np.asarray(basis["unwhitening_map"], dtype=float),
        )
        A = components["total_second_moment_whitened"]
        tau, scale_log_tau, shape_distance = scale_shape_from_A(A, estimator_mode=estimator_mode)
    else:
        fit = estimate_alpha_and_state(
            np.asarray(basis["Z"], dtype=float),
            rank=rank,
            em_iters=em_iters,
            nu=ROBUST_NU,
            reference_weights=weights,
        )
        A = matrix_series_from_state(fit.xs, rank, estimator_mode=estimator_mode)
        tau, scale_log_tau, shape_distance = scale_shape_from_A(A, estimator_mode=estimator_mode)
        mean_fit = None
        components = {
            "total_second_moment_whitened": A,
            "mean_component_whitened": np.zeros_like(A),
            "within_covariance_whitened": A,
            "smoothed_mean_whitened": np.zeros((len(A), rank)),
            "total_second_moment_original": map_whitened_matrices_to_original(A, np.asarray(basis["unwhitening_map"], dtype=float)),
            "mean_component_original": np.zeros((len(A), Q_array.shape[1], Q_array.shape[1])),
            "within_covariance_original": map_whitened_matrices_to_original(A, np.asarray(basis["unwhitening_map"], dtype=float)),
            "smoothed_mean_original": np.zeros((len(A), Q_array.shape[1])),
        }
    labels = list(outcome_labels or OUTCOME_LABELS)
    return RankResult(
        variant=variant,
        label=label,
        rank=rank,
        dates=dates,
        outcome_labels=labels,
        Q_scores=Q_array,
        beta=basis["beta"],
        eigvals=basis["eigvals"],
        shares=basis["shares"],
        V=basis["V"],
        Z=basis["Z"],
        fit=fit,
        A=A,
        tau=tau,
        scale_log_tau=scale_log_tau,
        shape_distance=shape_distance,
        estimator_mode=estimator_mode,
        reference_weights=weights,
        reference_covariance=np.asarray(basis["reference_covariance"], dtype=float),
        whitening_ridge=float(basis["whitening_ridge"]),
        whitening_map=np.asarray(basis["whitening_map"], dtype=float),
        unwhitening_map=np.asarray(basis["unwhitening_map"], dtype=float),
        total_second_moment_whitened=components["total_second_moment_whitened"],
        mean_component_whitened=components["mean_component_whitened"],
        within_covariance_whitened=components["within_covariance_whitened"],
        total_second_moment_original=components["total_second_moment_original"],
        mean_component_original=components["mean_component_original"],
        within_covariance_original=components["within_covariance_original"],
        smoothed_mean_whitened=components["smoothed_mean_whitened"],
        smoothed_mean_original=components["smoothed_mean_original"],
        mean_fit=mean_fit,
        reference_weighted_mean=np.asarray(basis["beta"], dtype=float),
    )


def rank_result_cache_key(
    data_hash: str,
    variant: str,
    rank: int,
    em_iters: int,
    outcome_labels: list[str] | None = None,
    m_dim: int | None = None,
    transform_signature: str = "raw",
    estimator_mode: str = ESTIMATOR_MODE,
) -> str:
    labels = tuple(outcome_labels or OUTCOME_LABELS)
    return cache_key(
        "rank_result",
        data_hash,
        H,
        L,
        labels,
        int(m_dim or ((H + 1) * len(labels))),
        variant,
        transform_signature,
        estimator_mode,
        ARITHMETIC_REFERENCE_RIDGE_SCALE if estimator_mode == "arithmetic_outer_product" else 0.0,
        rank,
        em_iters,
        ROBUST_NU,
        MIN_STUDENT_WEIGHT,
        ALPHA_GRID.tolist(),
    )


def rank_summary_row(res: RankResult) -> dict[str, Any]:
    eigengap = res.eigvals[res.rank - 1] - res.eigvals[res.rank] if len(res.eigvals) > res.rank else np.nan
    return {
        "variant": res.variant,
        "label": res.label,
        "rank": res.rank,
        "estimator_mode": getattr(res, "estimator_mode", "log_spd_legacy"),
        "state_dim": res.rank * (res.rank + 1) // 2,
        "alpha_hat": res.fit.alpha,
        "reference_whitening_ridge": getattr(res, "whitening_ridge", 0.0),
        "robust_loglik": res.fit.robust_loglik,
        "factor_log_score": res.fit.factor_log_score,
        "avg_factor_log_score_per_rank_dim": res.fit.factor_log_score / (len(res.dates) * res.rank),
        "avg_log_score_per_state_dim": res.fit.robust_loglik / (len(res.dates) * (res.rank * (res.rank + 1) // 2)),
        "transition_spectral_radius": res.fit.spectral_radius,
        "retained_trace_share": float(res.shares[: res.rank].sum()),
        "eigengap_R_to_Rplus1": float(eigengap),
        "tau_mean": float(res.tau.mean()),
        "tau_sd": float(res.tau.std(ddof=0)),
        "tau_max": float(res.tau.max()),
        "tau_max_month": pd.to_datetime(res.dates.iloc[int(np.argmax(res.tau))]).strftime("%Y-%m"),
        "min_robust_weight": float(res.fit.weights.min()),
        "median_robust_weight": float(np.median(res.fit.weights)),
    }


def write_rank_component_npz(res: RankResult, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {
        "A_total_whitened": np.asarray(res.total_second_moment_whitened if res.total_second_moment_whitened is not None else res.A, dtype=float),
        "A_mean_whitened": np.asarray(res.mean_component_whitened if res.mean_component_whitened is not None else np.zeros_like(res.A), dtype=float),
        "A_within_whitened": np.asarray(res.within_covariance_whitened if res.within_covariance_whitened is not None else res.A, dtype=float),
        "smoothed_mean_whitened": np.asarray(res.smoothed_mean_whitened if res.smoothed_mean_whitened is not None else np.zeros((len(res.A), res.rank)), dtype=float),
        "reference_weights": np.asarray(res.reference_weights if res.reference_weights is not None else normalize_reference_weights(len(res.A)), dtype=float),
        "reference_covariance": np.asarray(res.reference_covariance if res.reference_covariance is not None else np.empty((0, 0)), dtype=float),
        "whitening_map": np.asarray(res.whitening_map if res.whitening_map is not None else np.empty((0, 0)), dtype=float),
        "unwhitening_map": np.asarray(res.unwhitening_map if res.unwhitening_map is not None else np.empty((0, 0)), dtype=float),
    }
    if res.total_second_moment_original is not None:
        arrays["A_total_original"] = np.asarray(res.total_second_moment_original, dtype=float)
    if res.mean_component_original is not None:
        arrays["A_mean_original"] = np.asarray(res.mean_component_original, dtype=float)
    if res.within_covariance_original is not None:
        arrays["A_within_original"] = np.asarray(res.within_covariance_original, dtype=float)
    if res.smoothed_mean_original is not None:
        arrays["smoothed_mean_original"] = np.asarray(res.smoothed_mean_original, dtype=float)
    np.savez_compressed(path, **arrays)


def init_rank_worker(score_data: dict[str, dict[str, Any]], outcome_labels: list[str] | None = None) -> None:
    for _thread_var in [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[_thread_var] = "1"
    if outcome_labels is not None:
        set_outcome_labels(list(outcome_labels))
    _RANK_CONTEXT["score_data"] = score_data


def init_bootstrap_worker(context: dict[str, Any]) -> None:
    for _thread_var in [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[_thread_var] = "1"
    if "outcome_labels" in context:
        set_outcome_labels(list(context["outcome_labels"]))
    loaded = dict(context)
    for key, path in context.get("array_paths", {}).items():
        loaded[key] = np.load(path, mmap_mode="r")
    loaded["dates"] = pd.Series(pd.to_datetime(loaded["dates_ns"].astype("datetime64[ns]")))
    loaded["base_top10"] = set(loaded["base_top10"])
    _BOOTSTRAP_CONTEXT.clear()
    _BOOTSTRAP_CONTEXT.update(loaded)


def close_bootstrap_worker_context() -> None:
    for value in list(_BOOTSTRAP_CONTEXT.values()):
        if isinstance(value, np.memmap):
            mmap_obj = getattr(value, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
    _BOOTSTRAP_CONTEXT.clear()
    gc.collect()


def rmtree_with_memmap_retry(path: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            close_bootstrap_worker_context()
            time.sleep(0.2 * (attempt + 1))
    shutil.rmtree(path)


def estimate_rank_task(task: dict[str, Any]) -> tuple[tuple[str, int], RankResult, bool]:
    score_data = _RANK_CONTEXT.get("score_data", {})
    outcome_labels = list(task.get("outcome_labels", score_data.get(task["variant"], {}).get("outcome_labels", OUTCOME_LABELS)))
    m_dim = int(task.get("m_dim", (H + 1) * len(outcome_labels)))
    key = rank_result_cache_key(
        task["data_hash"],
        task["variant"],
        task["rank"],
        task["em_iters"],
        outcome_labels=outcome_labels,
        m_dim=m_dim,
        transform_signature=task.get("transform_signature", "raw"),
        estimator_mode=task.get("estimator_mode", ESTIMATOR_MODE),
    )
    cached = None if DISABLE_CACHE else cache_get(key)
    if cached is not None:
        return (task["variant"], task["rank"]), cached, True
    Q_scores = task.get("Q_scores")
    dates = task.get("dates")
    if Q_scores is None:
        Q_scores = score_data[task["variant"]]["Q_scores"]
    if dates is None:
        dates = score_data[task["variant"]]["dates"]
    res = estimate_rank_model(
        Q_scores,
        dates,
        task["variant"],
        task["label"],
        task["rank"],
        em_iters=task["em_iters"],
        outcome_labels=outcome_labels,
        estimator_mode=task.get("estimator_mode", ESTIMATOR_MODE),
    )
    if not DISABLE_CACHE:
        cache_set(key, res)
    return (task["variant"], task["rank"]), res, False


def bootstrap_draw_cache_key(
    data_hash: str,
    variant: str,
    draw: int,
    ix: np.ndarray,
    em_iters: int,
    outcome_labels: list[str],
    m_dim: int,
    rank: int,
    transform_signature: str,
    estimator_mode: str,
) -> str:
    ix_hash = hashlib.sha256(np.asarray(ix, dtype=np.int64).tobytes()).hexdigest()
    return cache_key(
        "bootstrap_draw",
        "ols_sufficient_stats",
        "variant_payload_v2",
        data_hash,
        variant,
        draw,
        ix_hash,
        H,
        L,
        tuple(outcome_labels),
        m_dim,
        rank,
        transform_signature,
        estimator_mode,
        ARITHMETIC_REFERENCE_RIDGE_SCALE if estimator_mode == "arithmetic_outer_product" else 0.0,
        em_iters,
        ROBUST_NU,
        MIN_STUDENT_WEIGHT,
        ALPHA_GRID.tolist(),
    )


def bootstrap_draw_task(task: dict[str, Any]) -> tuple[int, dict[str, np.ndarray], dict[str, Any], bool]:
    ctx = _BOOTSTRAP_CONTEXT
    draw = int(task["draw"])
    ix = task.get("ix")
    if ix is None:
        ix = np.asarray(ctx["boot_indices"][draw], dtype=int)
    variant = str(task.get("variant", ctx.get("variant", "base5_headline")))
    outcome_labels = list(ctx.get("outcome_labels", OUTCOME_LABELS))
    m_dim = int(ctx.get("m_dim", (H + 1) * len(outcome_labels)))
    rank = int(ctx.get("rank", HEADLINE_R))
    transform_signature = str(ctx.get("transform_signature", "raw"))
    estimator_mode = str(ctx.get("estimator_mode", ESTIMATOR_MODE))
    key = bootstrap_draw_cache_key(
        task["data_hash"],
        variant,
        draw,
        ix,
        task["em_iters"],
        outcome_labels,
        m_dim,
        rank,
        transform_signature,
        estimator_mode,
    )
    cached = None if DISABLE_CACHE else cache_get(key)
    if cached is not None:
        payload = {
            "tau": cached["tau"],
            "beta": cached.get("beta", np.full(m_dim, np.nan)),
            "shares": cached.get("shares", np.full(10, np.nan)),
            "modes": cached.get("modes", np.full((rank, m_dim), np.nan)),
            "subspace_angle": np.asarray([cached.get("subspace_angle", np.nan)], dtype=float),
        }
        return draw, payload, cached["row"], True
    if ctx.get("bootstrap_method") == "ols_sufficient_stats":
        Qb, _ = lp_scores_from_design(
            ctx["X_design"],
            ctx["Y_all"],
            row_xx=ctx["row_xx"],
            row_xy=ctx["row_xy"],
            ix=ix,
        )
        Qb = apply_score_transform(Qb, outcome_labels, ctx.get("score_transform"))
        dates_for_fit = ctx["dates"]
        headline_V = ctx["headline_V"]
        base_top10 = ctx["base_top10"]
        march_idx = ctx["march_idx"]
    else:
        Qb = task["beta"] + task["E"][ix]
        Qb = apply_score_transform(Qb, outcome_labels, ctx.get("score_transform"))
        dates_for_fit = task["dates"]
        headline_V = task["headline_V"]
        base_top10 = task["base_top10"]
        march_idx = task["march_idx"]
    try:
        rb = estimate_rank_model(
            Qb,
            dates_for_fit,
            variant,
            "bootstrap",
            rank,
            em_iters=task["em_iters"],
            outcome_labels=outcome_labels,
            estimator_mode=estimator_mode,
        )
        dates = pd.to_datetime(dates_for_fit)
        tau = rb.tau
        top10 = set(dates.iloc[np.argsort(tau)[::-1][:10]].dt.strftime("%Y-%m"))
        angles = principal_angles(headline_V, rb.V)
        matched_modes = match_modes(rb.V, headline_V)
        row = {
            "draw": draw + 1,
            "alpha_hat": rb.fit.alpha,
            "retained_trace_share": float(rb.shares[:rank].sum()),
            "max_tau": float(tau.max()),
            "max_month": dates.iloc[int(np.argmax(tau))].strftime("%Y-%m"),
            "march_2020_rank": int(np.where(np.argsort(tau)[::-1] == march_idx)[0][0] + 1) if march_idx is not None else np.nan,
            "march_2020_tau": float(tau[march_idx]) if march_idx is not None else np.nan,
            "top10_overlap_with_baseline": int(len(base_top10 & top10)),
            "max_subspace_angle_degrees": float(np.max(angles)),
            "robust_loglik": rb.fit.robust_loglik,
        }
        payload = {
            "tau": tau,
            "beta": rb.beta,
            "shares": rb.shares[:10],
            "modes": matched_modes[:, :rank].T if matched_modes.shape[1] >= rank else matched_modes.T,
            "subspace_angle": np.asarray([float(np.max(angles))], dtype=float),
        }
    except Exception as exc:
        n_dates = len(ctx["dates"]) if ctx else len(task["dates"])
        tau = np.full(n_dates, np.nan)
        row = {"draw": draw + 1, "error": str(exc)}
        payload = {
            "tau": tau,
            "beta": np.full(m_dim, np.nan),
            "shares": np.full(10, np.nan),
            "modes": np.full((rank, m_dim), np.nan),
            "subspace_angle": np.asarray([np.nan], dtype=float),
        }
    if not DISABLE_CACHE:
        cache_set(key, {**payload, "row": row})
    return draw, payload, row, False


def should_bootstrap_variant(spec: VariantSpec) -> bool:
    if BOOTSTRAP_VARIANTS in {"", "none", "0", "false", "no"}:
        return False
    if BOOTSTRAP_VARIANTS == "all":
        return True
    requested = {x.strip() for x in BOOTSTRAP_VARIANTS.split(",") if x.strip()}
    return spec.key in requested


def run_full_pipeline_bootstrap(
    data_hash: str,
    spec: VariantSpec,
    scores: dict[str, Any],
    headline: RankResult,
    workers: int,
    cache_hits: dict[str, int],
    cache_misses: dict[str, int],
    base_top10: set[str] | None = None,
    march_idx: int | None = None,
) -> dict[str, Any]:
    """Rebuild LP scores, eigensystem, state model, and tau path for one variant."""
    dates = pd.to_datetime(headline.dates)
    q = np.asarray(headline.Q_scores, float)
    labels = list(headline.outcome_labels)
    m_dim = int(q.shape[1])
    rank = int(headline.rank)
    if base_top10 is None:
        base_top10 = set(dates.iloc[np.argsort(headline.tau)[::-1][:10]].dt.strftime("%Y-%m"))
    if march_idx is None:
        march_mask = dates.dt.strftime("%Y-%m") == "2020-03"
        march_idx = int(np.where(march_mask.to_numpy())[0][0]) if march_mask.any() else None

    boot_stats, stats_from_cache = load_or_build_bootstrap_stats(data_hash, spec.key, scores)
    cache_hits["bootstrap_sufficient_stats"] += int(stats_from_cache)
    cache_misses["bootstrap_sufficient_stats"] += int(not stats_from_cache)
    rng = np.random.default_rng(BOOT_SEED)
    boot_indices = np.asarray(
        [circular_block_indices(len(q), BOOT_BLOCK_LEN, rng) for _ in range(B_BOOT)],
        dtype=np.int64,
    )
    namespace = cache_key(
        WORKER_ARRAY_VERSION,
        data_hash,
        spec.key,
        H,
        L,
        tuple(labels),
        m_dim,
        rank,
        B_BOOT,
        BOOT_BLOCK_LEN,
        BOOT_SEED,
        spec.transform_signature,
    )
    array_paths = {
        "X_design": str(write_worker_array(namespace, "X_design", boot_stats["X_design"])),
        "Y_all": str(write_worker_array(namespace, "Y_all", boot_stats["Y_all"])),
        "row_xx": str(write_worker_array(namespace, "row_xx", boot_stats["row_xx"])),
        "row_xy": str(write_worker_array(namespace, "row_xy", boot_stats["row_xy"])),
        "boot_indices": str(write_worker_array(namespace, "boot_indices", boot_indices)),
    }
    boot_context = {
        "variant": spec.key,
        "array_paths": array_paths,
        "dates_ns": dates.to_numpy(dtype="datetime64[ns]"),
        "headline_V": headline.V,
        "base_top10": sorted(base_top10),
        "march_idx": march_idx,
        "bootstrap_method": "ols_sufficient_stats",
        "outcome_labels": labels,
        "m_dim": m_dim,
        "rank": rank,
        "score_transform": scores.get("score_transform", {"kind": "raw"}),
        "transform_signature": spec.transform_signature,
        "estimator_mode": headline.estimator_mode,
    }
    tau_boot = np.empty((B_BOOT, len(dates)))
    beta_boot = np.empty((B_BOOT, m_dim))
    share_boot = np.empty((B_BOOT, 10))
    mode_boot = np.empty((B_BOOT, rank, m_dim))
    subspace_angle_boot = np.empty(B_BOOT)
    tasks = [{"data_hash": data_hash, "variant": spec.key, "draw": b, "em_iters": BOOT_EM_ITERS} for b in range(B_BOOT)]
    rows = []
    try:
        boot_results = run_parallel_tasks(
            bootstrap_draw_task,
            tasks,
            workers,
            initializer=init_bootstrap_worker,
            initargs=(boot_context,),
        )
        for draw, payload, row, from_cache in boot_results:
            tau_boot[draw] = payload["tau"]
            beta_boot[draw] = payload["beta"]
            share_boot[draw] = payload["shares"]
            mode_boot[draw] = payload["modes"]
            subspace_angle_boot[draw] = float(np.ravel(payload["subspace_angle"])[0])
            rows.append(row)
            cache_hits["bootstrap_draws"] += int(from_cache)
            cache_misses["bootstrap_draws"] += int(not from_cache)
    finally:
        close_bootstrap_worker_context()
    rows = sorted(rows, key=lambda r: r.get("draw", 0))
    boot_df = pd.DataFrame(rows)
    valid_boot = np.isfinite(tau_boot).all(axis=1)
    tau_boot_valid = tau_boot[valid_boot]
    band = positive_simultaneous_band(tau_boot_valid, headline.tau, level=0.90) if len(tau_boot_valid) else positive_simultaneous_band(np.asarray([headline.tau]), headline.tau, level=0.90)
    return {
        "boot_df": boot_df,
        "tau_boot": tau_boot,
        "tau_boot_valid": tau_boot_valid,
        "beta_boot": beta_boot,
        "share_boot": share_boot,
        "mode_boot": mode_boot,
        "subspace_angle_boot": subspace_angle_boot,
        "band": band,
    }


def ffbs_state_draws(fit: StateFit, ndraws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, d = fit.xs.shape
    draws = np.empty((ndraws, T, d))
    gains = np.zeros((T - 1, d, d))
    cond_chol = np.zeros((T, d, d))
    C_last = stabilize_cov(fit.Pf[-1], 1e-12)
    cond_chol[-1] = np.linalg.cholesky(C_last)
    for t in range(T - 2, -1, -1):
        Pnext = stabilize_cov(fit.Pp[t + 1], 1e-10)
        c_next, lower_next = cho_factor(Pnext, lower=True, check_finite=False)
        J = cho_solve((c_next, lower_next), (fit.Pf[t] @ fit.F.T).T, check_finite=False).T
        gains[t] = J
        cov = stabilize_cov(fit.Pf[t] - J @ fit.Pp[t + 1] @ J.T, 1e-12)
        cond_chol[t] = np.linalg.cholesky(cov)
    for b in range(ndraws):
        path = np.zeros((T, d))
        path[-1] = fit.xf[-1] + cond_chol[-1] @ rng.normal(size=d)
        for t in range(T - 2, -1, -1):
            J = gains[t]
            mean = fit.xf[t] + J @ (path[t + 1] - fit.xp[t + 1])
            path[t] = mean + cond_chol[t] @ rng.normal(size=d)
        draws[b] = path
    return draws


def simultaneous_band(draws: np.ndarray, center: np.ndarray | None = None, level: float = 0.90) -> dict[str, np.ndarray | float]:
    if center is None:
        center = np.median(draws, axis=0)
    point = np.quantile(draws, [(1 - level) / 2, 0.5, 1 - (1 - level) / 2], axis=0)
    scale = np.std(draws, axis=0, ddof=1 if len(draws) > 1 else 0)
    scale = np.maximum(scale, np.nanmedian(scale[scale > 0]) * 0.1 if np.any(scale > 0) else 1e-6)
    sup = np.max(np.abs((draws - center[None, :]) / scale[None, :]), axis=1)
    crit = float(np.quantile(sup, level))
    return {
        "point_low": point[0],
        "point_med": point[1],
        "point_high": point[2],
        "sim_low": center - crit * scale,
        "sim_high": center + crit * scale,
        "sup_crit": crit,
    }


def positive_simultaneous_band(
    draws: np.ndarray,
    center: np.ndarray | None = None,
    level: float = 0.90,
    floor: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Pointwise positive-scale quantiles plus log-scale simultaneous path bands."""
    draws = np.asarray(draws, float)
    draws_safe = np.maximum(draws, floor)
    if center is None:
        center = np.median(draws_safe, axis=0)
    center_safe = np.maximum(np.asarray(center, float), floor)
    point = np.quantile(draws_safe, [(1 - level) / 2, 0.5, 1 - (1 - level) / 2], axis=0)
    log_draws = np.log(draws_safe)
    log_center = np.log(center_safe)
    log_scale = np.std(log_draws, axis=0, ddof=1 if len(log_draws) > 1 else 0)
    log_scale = np.maximum(log_scale, np.nanmedian(log_scale[log_scale > 0]) * 0.1 if np.any(log_scale > 0) else 1e-6)
    sup = np.max(np.abs((log_draws - log_center[None, :]) / log_scale[None, :]), axis=1)
    crit = float(np.quantile(sup, level))
    return {
        "point_low": point[0],
        "point_med": point[1],
        "point_high": point[2],
        "sim_low": np.exp(log_center - crit * log_scale),
        "sim_high": np.exp(log_center + crit * log_scale),
        "sup_crit": crit,
        "scale": "log",
    }


def worker_benchmark_task(task: dict[str, Any]) -> float:
    rng = np.random.default_rng(int(task["seed"]))
    n = int(task.get("n", 160))
    repeats = int(task.get("repeats", 3))
    t0 = time.perf_counter()
    total = 0.0
    for _ in range(repeats):
        A = rng.normal(size=(n, n))
        S = A @ A.T + 0.1 * np.eye(n)
        total += float(np.linalg.eigvalsh(S)[-1])
    return float(total + time.perf_counter() - t0)


def benchmark_worker_count(max_workers: int) -> dict[str, Any]:
    candidates = [w for w in [2, 4, 6, 8] if w <= max_workers]
    if not candidates:
        candidates = [1]
    task_count = max(12, max(candidates) * 3)
    rows = []
    for workers in candidates:
        tasks = [{"seed": 1100 + i, "n": 160, "repeats": 3} for i in range(task_count)]
        t0 = time.perf_counter()
        run_parallel_tasks(worker_benchmark_task, tasks, workers)
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "workers": workers,
                "elapsed_seconds": round(elapsed, 4),
                "tasks": len(tasks),
                "seconds_per_task": elapsed / len(tasks),
            }
        )
    best = min(rows, key=lambda row: row["seconds_per_task"])
    return {"selected_workers": int(best["workers"]), "candidates": rows}


def run_parallel_tasks(fn, tasks: list[dict[str, Any]], workers: int, initializer=None, initargs: tuple[Any, ...] = ()):
    if workers <= 1 or len(tasks) <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(task) for task in tasks]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
        future_map = {pool.submit(fn, task): i for i, task in enumerate(tasks)}
        ordered = [None] * len(tasks)
        for fut in as_completed(future_map):
            ordered[future_map[fut]] = fut.result()
        results = ordered
    return results


def savefig(fig: plt.Figure, name: str, tight: bool = True) -> Path:
    path = CHARTS / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def selected_shape_episodes(dates: pd.Series, tau: np.ndarray, scale_log_tau: np.ndarray, surface_rms: np.ndarray, march_idx: int | None) -> list[dict[str, Any]]:
    neutral_score = np.hypot(scale_log_tau, surface_rms)
    candidates = [
        ("reference_neutral", int(np.nanargmin(neutral_score))),
        ("max_tau", int(np.nanargmax(tau))),
        ("max_surface_redistribution", int(np.nanargmax(surface_rms))),
    ]
    if march_idx is not None:
        candidates.append(("march_2020", int(march_idx)))
    episodes: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    for label, idx in candidates:
        if idx in seen:
            episodes[seen[idx]]["episode"] += f"+{label}"
            continue
        seen[idx] = len(episodes)
        episodes.append(
            {
                "episode": label,
                "idx": idx,
                "date": pd.to_datetime(dates.iloc[idx]).strftime("%Y-%m"),
                "tau": float(tau[idx]),
                "scale_log_tau": float(scale_log_tau[idx]),
                "surface_shape_rms_log_relative": float(surface_rms[idx]),
            }
        )
    return episodes


def plot_shape_heatmap_atlas(log_relative: np.ndarray, dates: pd.Series, episodes: list[dict[str, Any]], H: int, labels: list[str]) -> Path:
    pvars = len(labels)
    n = max(1, len(episodes))
    ncols = n if n <= 3 else 2
    nrows = int(math.ceil(n / ncols))
    selected = np.vstack([log_relative[ep["idx"]].reshape(H + 1, pvars) for ep in episodes])
    vmax = float(np.nanquantile(np.abs(selected), 0.98)) if selected.size else 1.0
    vmax = max(vmax, 0.25)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols + 0.7, 3.3 * nrows), squeeze=False)
    image = None
    yticks = [h for h in [0, 3, 6, 12, 18, H] if h <= H]
    for ax, ep in zip(axes.ravel(), episodes):
        mat = log_relative[ep["idx"]].reshape(H + 1, pvars)
        image = ax.imshow(mat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{ep['episode'].replace('_', ' ')}: {ep['date']}")
        ax.set_xticks(np.arange(pvars), labels, rotation=28, ha="right")
        ax.set_yticks(yticks)
        ax.set_ylabel("Horizon")
    for ax in axes.ravel()[len(episodes):]:
        ax.axis("off")
    if image is not None:
        fig.subplots_adjust(right=0.90, top=0.82 if nrows == 1 else 0.88, bottom=0.22 if nrows == 1 else 0.14, wspace=0.34, hspace=0.55)
        cax = fig.add_axes([0.925, 0.22 if nrows == 1 else 0.16, 0.018, 0.58 if nrows == 1 else 0.68])
        fig.colorbar(image, cax=cax, label="log cell_amp / tau_soft")
    fig.suptitle("Full-coordinate shape allocation after removing tau_soft scale", y=0.98)
    return savefig(fig, "02a_full_coordinate_cell_shape_heatmap_atlas", tight=False)


def plot_shape_marginals(dates: pd.Series, metrics_df: pd.DataFrame) -> Path:
    date_vals = pd.to_datetime(dates)
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), sharex=True)
    ax = axes[0]
    for col, label, color in [
        ("financial_variable_share", "financial variables", "tab:blue"),
        ("macro_variable_share", "macro variables", "tab:orange"),
    ]:
        ax.plot(date_vals, metrics_df[col], label=label, color=color)
        lo = f"{col}_p05"
        hi = f"{col}_p95"
        if lo in metrics_df and hi in metrics_df:
            ax.fill_between(date_vals, metrics_df[lo], metrics_df[hi], color=color, alpha=0.12)
    ax.set_ylabel("Share")
    ax.set_title("Variable allocation of normalized shape variance")
    ax.legend(ncol=2)

    ax = axes[1]
    for col, label in [
        ("short_horizon_share", "0-3"),
        ("medium_horizon_share", "4-12"),
        ("long_horizon_share", "13-H"),
    ]:
        ax.plot(date_vals, metrics_df[col], label=label)
        lo = f"{col}_p05"
        hi = f"{col}_p95"
        if lo in metrics_df and hi in metrics_df:
            ax.fill_between(date_vals, metrics_df[lo], metrics_df[hi], alpha=0.08)
    ax.set_ylabel("Share")
    ax.set_title("Horizon allocation of normalized shape variance")
    ax.legend(ncol=3)

    ax = axes[2]
    for col, label in [
        ("cell_effective_support", "cells"),
        ("variable_effective_support", "variables"),
        ("horizon_effective_support", "horizons"),
    ]:
        ax.plot(date_vals, metrics_df[col], label=label)
        lo = f"{col}_p05"
        hi = f"{col}_p95"
        if lo in metrics_df and hi in metrics_df:
            ax.fill_between(date_vals, metrics_df[lo], metrics_df[hi], alpha=0.08)
    ax.set_ylabel("Effective support / N")
    ax.set_title("How concentrated the normalized shape allocation is")
    ax.legend(ncol=3)
    return savefig(fig, "04_surface_shape_marginals")


def plot_full_coordinate_block_shape(block_df: pd.DataFrame, concentration_df: pd.DataFrame) -> Path:
    dates = pd.to_datetime(block_df["date"].drop_duplicates())
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), sharex=True)

    ax = axes[0]
    for block, label in [("macro_outcomes", "macro outcomes"), ("financial_outcomes", "financial outcomes")]:
        sub = block_df[block_df["block"].eq(block)]
        if len(sub):
            ax.plot(pd.to_datetime(sub["date"]), sub["block_shape"], label=label)
    ax.axhline(1.0, linewidth=0.8, color="black")
    ax.set_ylabel("Block shape")
    ax.set_title("Full-coordinate block probes by outcome group")
    ax.legend(ncol=2)

    ax = axes[1]
    for block, label in [("horizons_00_03", "0-3"), ("horizons_04_12", "4-12"), ("horizons_13_24", "13-24")]:
        sub = block_df[block_df["block"].eq(block)]
        if len(sub):
            ax.plot(pd.to_datetime(sub["date"]), sub["block_shape"], label=label)
    ax.axhline(1.0, linewidth=0.8, color="black")
    ax.set_ylabel("Block shape")
    ax.set_title("Full-coordinate block probes by horizon bucket")
    ax.legend(ncol=3)

    ax = axes[2]
    if len(concentration_df):
        conc_dates = pd.to_datetime(concentration_df["date"])
        for col, label in [
            ("cell_effective_support", "cells"),
            ("variable_effective_support", "variables"),
            ("horizon_effective_support", "horizons"),
        ]:
            if col in concentration_df:
                ax.plot(conc_dates, concentration_df[col], label=label)
    ax.set_ylabel("Effective support / N")
    ax.set_title("Finite-working-grid concentration of positive cell amplification")
    ax.legend(ncol=3)
    if len(dates):
        ax.set_xlim(dates.min(), dates.max())
    return savefig(fig, "02b_full_coordinate_block_shape_paths")


def plot_shape_directions(shape: np.ndarray, V: np.ndarray, episodes: list[dict[str, Any]], H: int, labels: list[str]) -> tuple[Path, pd.DataFrame]:
    pvars = len(labels)
    rows = []
    surfaces = []
    for ep in episodes:
        S = 0.5 * (shape[ep["idx"]] + shape[ep["idx"]].T)
        w, U = np.linalg.eigh(S - np.eye(S.shape[0]))
        for direction_label, eig_idx in [("under_amplified", int(np.argmin(w))), ("over_amplified", int(np.argmax(w)))]:
            surf = (V @ U[:, eig_idx]).reshape(H + 1, pvars)
            surfaces.append(surf)
            for hh in range(H + 1):
                for j, label in enumerate(labels):
                    rows.append(
                        {
                            "episode": ep["episode"],
                            "date": ep["date"],
                            "direction": direction_label,
                            "shape_eigenvalue_minus_one": float(w[eig_idx]),
                            "horizon_months": hh,
                            "variable": label,
                            "loading": float(surf[hh, j]),
                        }
                    )
    vmax = float(np.nanquantile(np.abs(np.concatenate([s.ravel() for s in surfaces])), 0.98)) if surfaces else 1.0
    vmax = max(vmax, 0.01)
    nrows = max(1, len(episodes))
    fig, axes = plt.subplots(nrows, 2, figsize=(11.4, 2.95 * nrows), squeeze=False)
    image = None
    yticks = [h for h in [0, 3, 6, 12, 18, H] if h <= H]
    for row_idx, ep in enumerate(episodes):
        S = 0.5 * (shape[ep["idx"]] + shape[ep["idx"]].T)
        w, U = np.linalg.eigh(S - np.eye(S.shape[0]))
        specs = [("under-amplified", int(np.argmin(w))), ("over-amplified", int(np.argmax(w)))]
        for col_idx, (label, eig_idx) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            mat = (V @ U[:, eig_idx]).reshape(H + 1, pvars)
            image = ax.imshow(mat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{ep['date']} {label} ({w[eig_idx]:+.2f})")
            ax.set_xticks(np.arange(pvars), labels, rotation=28, ha="right")
            ax.set_yticks(yticks)
            ax.set_ylabel("Horizon")
    if image is not None:
        fig.subplots_adjust(right=0.90, top=0.92, bottom=0.09, wspace=0.28, hspace=0.62)
        cax = fig.add_axes([0.925, 0.16, 0.018, 0.68])
        fig.colorbar(image, cax=cax, label="direction loading")
    fig.suptitle("Dominant normalized shape directions in response-surface space", y=0.985)
    return savefig(fig, "05_surface_shape_directions", tight=False), pd.DataFrame(rows)


def df_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    d = df.head(max_rows).copy()
    rows = ["<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns) + "</tr>"]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(f"<td>{float(v):.4f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


PDF_TABLE_WIDTH = 7.15 * inch


def _pdf_table_cell(value: Any, *, header: bool = False) -> Paragraph:
    if isinstance(value, (float, np.floating)):
        text = "" if not np.isfinite(float(value)) else f"{float(value):.4f}"
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        text = "" if is_missing else str(value)
    text = html.escape(text)
    if header:
        text = text.replace("_", "_<br/>")
    else:
        text = text.replace(", ", ",<br/>").replace("; ", ";<br/>")
    style = ParagraphStyle(
        "PdfTableHeader" if header else "PdfTableCell",
        fontName="Helvetica-Bold" if header else "Helvetica",
        fontSize=5.7 if header else 5.4,
        leading=6.4 if header else 6.1,
        wordWrap="CJK",
    )
    return Paragraph(text, style)


def _pdf_table_widths(columns: list[Any], total_width: float = PDF_TABLE_WIDTH) -> list[float]:
    weights = []
    for col in columns:
        name = str(col).lower()
        if any(token in name for token in ["variant", "label", "outcome", "source", "quantity"]):
            weights.append(1.45)
        elif any(token in name for token in ["month", "date", "episode", "sample"]):
            weights.append(1.05)
        else:
            weights.append(0.85)
    scale = total_width / max(sum(weights), 1.0)
    return [w * scale for w in weights]


def make_table(
    df: pd.DataFrame,
    max_rows: int = 12,
    cols: list[str] | None = None,
    max_cols: int = 9,
) -> Table:
    d = df.copy()
    if cols is not None:
        keep = [c for c in cols if c in d.columns]
        d = d[keep].copy() if keep else d.iloc[:, :max_cols].copy()
    elif len(d.columns) > max_cols:
        d = d.iloc[:, :max_cols].copy()
    d = d.head(max_rows)
    if d.empty and len(d.columns) == 0:
        d = pd.DataFrame({"note": ["No rows available."]})
    data = [[_pdf_table_cell(c, header=True) for c in d.columns]]
    for _, row in d.iterrows():
        data.append([_pdf_table_cell(v) for v in row.tolist()])
    tbl = Table(data, repeatRows=1, colWidths=_pdf_table_widths(list(d.columns)), hAlign="LEFT", splitByRow=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.3),
                ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
            ]
        )
    )
    return tbl


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_contents(src: Path, dst: Path) -> None:
    reset_dir(dst)
    if not src.exists():
        return
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def basis_energy_df(res: RankResult) -> pd.DataFrame:
    rows = []
    labels = list(res.outcome_labels)
    pvars = len(labels)
    for r in range(res.rank):
        surface = res.V[:, r].reshape(H + 1, pvars)
        energy = (surface**2).sum(axis=0)
        energy = energy / np.maximum(energy.sum(), 1e-12)
        rows.append(
            {
                "basis": r + 1,
                "trace_share": float(res.shares[r]),
                "dominant_variable": labels[int(np.argmax(energy))],
                "peak_horizon_months": int(np.argmax((surface**2).sum(axis=1))),
                **{f"energy_{labels[j]}": float(energy[j]) for j in range(pvars)},
            }
        )
    return pd.DataFrame(rows)


def write_top5_compatible_outputs(
    panels: dict[str, pd.DataFrame],
    score_data: dict[str, dict[str, Any]],
    headline: RankResult,
    rank_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    tau_draws: np.ndarray,
    diag_draws: np.ndarray,
    beta_boot: np.ndarray,
    share_boot: np.ndarray,
    mode_boot: np.ndarray,
    subspace_angle_boot: np.ndarray,
) -> None:
    reset_dir(TOP5_COMPAT_OUT)
    top_tables = TOP5_COMPAT_OUT / "tables"
    top_charts = TOP5_COMPAT_OUT / "charts"
    top_tables.mkdir(parents=True, exist_ok=True)
    top_charts.mkdir(parents=True, exist_ok=True)

    labels = list(headline.outcome_labels)
    pvars = len(labels)
    dates = pd.to_datetime(headline.dates)
    beta_surface = headline.beta.reshape(H + 1, pvars)
    beta_band = np.nanquantile(beta_boot, [0.05, 0.50, 0.95], axis=0)
    share_band = np.nanquantile(share_boot, [0.05, 0.50, 0.95], axis=0)
    cum_band = np.nanquantile(np.cumsum(share_boot, axis=1), [0.05, 0.50, 0.95], axis=0)
    mode_band = np.nanquantile(mode_boot, [0.05, 0.50, 0.95], axis=0)
    mode_corrs = np.einsum("brm,rm->br", mode_boot, headline.V.T, optimize=True)

    for src_name, out_name in [
        ("panel", "ovk_monetary_panel_monthly_fixed_full.csv"),
        ("monthly", "monthly_shocks_repaired.csv"),
        ("events", "event_shocks_with_manual_fields.csv"),
        ("event_monthly", "event_level_manual_monthly_aggregation.csv"),
    ]:
        if src_name in panels:
            panels[src_name].to_csv(TOP5_COMPAT_OUT / out_name, index=False)
    panels["panel"].dropna(subset=["MP_median_fallback"]).to_csv(
        TOP5_COMPAT_OUT / "ovk_monetary_panel_monthly_fixed_overlap.csv", index=False
    )

    irf_rows = []
    for hh in range(H + 1):
        for j, label in enumerate(labels):
            k = hh * pvars + j
            irf_rows.append(
                {
                    "horizon_months": hh,
                    "variable": label,
                    "estimate": beta_surface[hh, j],
                    "boot_p05": beta_band[0, k],
                    "boot_median": beta_band[1, k],
                    "boot_p95": beta_band[2, k],
                }
            )
    irf_df = pd.DataFrame(irf_rows)
    irf_df.to_csv(top_tables / "average_irf_with_block_bootstrap_bands.csv", index=False)

    eigen_df = pd.DataFrame(
        {
            "rank": np.arange(1, 11),
            "share_estimate": headline.shares[:10],
            "share_p05": share_band[0],
            "share_median": share_band[1],
            "share_p95": share_band[2],
            "cumulative_estimate": np.cumsum(headline.shares[:10]),
            "cumulative_p05": cum_band[0],
            "cumulative_median": cum_band[1],
            "cumulative_p95": cum_band[2],
        }
    )
    eigen_df.to_csv(top_tables / "average_ovk_eigenspectrum_with_bands.csv", index=False)

    mode_energy_boot = np.zeros((len(mode_boot), HEADLINE_R, pvars))
    for b in range(len(mode_boot)):
        for r in range(HEADLINE_R):
            surf = mode_boot[b, r].reshape(H + 1, pvars)
            en = (surf**2).sum(axis=0)
            mode_energy_boot[b, r] = en / np.maximum(en.sum(), 1e-12)
    energy_q = np.nanquantile(mode_energy_boot, [0.05, 0.50, 0.95], axis=0)
    mode_stability = pd.DataFrame(
        {
            "basis": np.arange(1, HEADLINE_R + 1),
            "median_abs_corr_with_bootstrap_basis": np.nanmedian(np.abs(mode_corrs), axis=0),
            "p05_abs_corr": np.nanquantile(np.abs(mode_corrs), 0.05, axis=0),
            "p95_abs_corr": np.nanquantile(np.abs(mode_corrs), 0.95, axis=0),
        }
    )
    mode_stability.to_csv(top_tables / "top5_basis_bootstrap_stability.csv", index=False)

    basis_load_rows = []
    basis_diag_rows = []
    for r in range(HEADLINE_R):
        surface = headline.V[:, r].reshape(H + 1, pvars)
        lo = mode_band[0, r].reshape(H + 1, pvars)
        med = mode_band[1, r].reshape(H + 1, pvars)
        hi = mode_band[2, r].reshape(H + 1, pvars)
        for hh in range(H + 1):
            for j, label in enumerate(labels):
                basis_load_rows.append(
                    {
                        "basis": r + 1,
                        "horizon_months": hh,
                        "variable": label,
                        "loading_estimate": surface[hh, j],
                        "boot_p05": lo[hh, j],
                        "boot_median": med[hh, j],
                        "boot_p95": hi[hh, j],
                    }
                )
        energy = (surface**2).sum(axis=0)
        energy = energy / np.maximum(energy.sum(), 1e-12)
        row = {
            "basis": r + 1,
            "eigen_share_estimate": headline.shares[r],
            "eigen_share_p05": share_band[0, r],
            "eigen_share_p95": share_band[2, r],
            "dominant_variable": labels[int(np.argmax(energy))],
            "peak_horizon_months": int(np.argmax((surface**2).sum(axis=1))),
            "median_abs_corr_with_bootstrap_basis": mode_stability.loc[r, "median_abs_corr_with_bootstrap_basis"],
            "p05_abs_corr_with_bootstrap_basis": mode_stability.loc[r, "p05_abs_corr"],
            "p95_abs_corr_with_bootstrap_basis": mode_stability.loc[r, "p95_abs_corr"],
        }
        for j, label in enumerate(labels):
            key = safe_name(label)
            row[f"{key}_energy_estimate"] = energy[j]
            row[f"{key}_energy_p05"] = energy_q[0, r, j]
            row[f"{key}_energy_median"] = energy_q[1, r, j]
            row[f"{key}_energy_p95"] = energy_q[2, r, j]
        basis_diag_rows.append(row)
    basis_load_df = pd.DataFrame(basis_load_rows)
    basis_diag_df = pd.DataFrame(basis_diag_rows)
    basis_load_df.to_csv(top_tables / "top5_basis_loadings_with_bootstrap_bands.csv", index=False)
    basis_diag_df.to_csv(top_tables / "top5_basis_diagnostics_with_bootstrap_bands.csv", index=False)

    tau_band90 = np.nanquantile(tau_draws, [0.05, 0.50, 0.95], axis=0)
    tau_band68 = np.nanquantile(tau_draws, [0.16, 0.84], axis=0)
    diag_band90 = np.nanquantile(diag_draws, [0.05, 0.50, 0.95], axis=0)
    diag_band68 = np.nanquantile(diag_draws, [0.16, 0.84], axis=0)
    baseline_scores = score_data[headline.variant]
    drift_data = {
        "date": dates,
        "trace_A_over_R": headline.tau,
        "trace_A_p05": tau_band90[0],
        "trace_A_median": tau_band90[1],
        "trace_A_p95": tau_band90[2],
        "trace_A_p16": tau_band68[0],
        "trace_A_p84": tau_band68[1],
        "trace_A_full_pipeline_p05": path_df["tau_full_pipeline_p05"],
        "trace_A_full_pipeline_p95": path_df["tau_full_pipeline_p95"],
        "MP_used_std": baseline_scores["mstd"],
        "CBI_used_std": baseline_scores["cstd"],
    }
    panel = panels["panel"]
    valid = np.asarray(baseline_scores["valid_idx"], dtype=int)
    if "fallback_flag" in panel.columns:
        drift_data["used_pm_fallback_current_month"] = panel["fallback_flag"].iloc[valid].fillna(False).to_numpy(bool)
    for r in range(HEADLINE_R):
        drift_data[f"A{r+1}{r+1}_basis{r+1}"] = headline.A[:, r, r]
        drift_data[f"A{r+1}{r+1}_p05"] = diag_band90[0, r]
        drift_data[f"A{r+1}{r+1}_median"] = diag_band90[1, r]
        drift_data[f"A{r+1}{r+1}_p95"] = diag_band90[2, r]
        drift_data[f"A{r+1}{r+1}_p16"] = diag_band68[0, r]
        drift_data[f"A{r+1}{r+1}_p84"] = diag_band68[1, r]
    for i in range(HEADLINE_R):
        for j in range(i + 1, HEADLINE_R):
            drift_data[f"A{i+1}{j+1}_basis{i+1}_basis{j+1}"] = headline.A[:, i, j]
    drift_df = pd.DataFrame(drift_data)
    drift_df.to_csv(top_tables / "state_space_A_t_top5_drift_estimates_with_bands.csv", index=False)
    write_rank_component_npz(headline, top_tables / "top5_second_moment_components.npz")
    top_months = drift_df.sort_values("trace_A_over_R", ascending=False).head(20).reset_index(drop=True)
    top_months["date_str"] = pd.to_datetime(top_months["date"]).dt.strftime("%Y-%m")
    top_months.to_csv(top_tables / "top_months_by_top5_state_space_kernel_amplification.csv", index=False)
    pd.DataFrame(
        {
            "diagnostic": ["top5_subspace_max_principal_angle_degrees"],
            "estimate": [0.0],
            "bootstrap_p05": [np.nanquantile(subspace_angle_boot, 0.05)],
            "bootstrap_median": [np.nanmedian(subspace_angle_boot)],
            "bootstrap_p95": [np.nanquantile(subspace_angle_boot, 0.95)],
            "note": ["Publication-grade full-pipeline bootstrap; each draw re-estimates the score surface, basis, alpha, F, Q, R, and A_t."],
        }
    ).to_csv(top_tables / "top5_subspace_bootstrap_stability.csv", index=False)

    min_eig_A = float(np.min(np.linalg.eigvalsh(headline.A)))
    model_label = (
        "Arithmetic outer-product Kalman state-space model for total whitened second moments"
        if headline.estimator_mode == "arithmetic_outer_product"
        else "Deprecated legacy robust log-Euclidean VAR(1) Kalman state-space model"
    )
    observation_label = (
        "y_t = svec(u_t u_t') = s_t + eps_t; R estimated by weighted EM-style iterations with Student-t robust weights"
        if headline.estimator_mode == "arithmetic_outer_product"
        else "y_t = svec(log(Gtilde_t)) = s_t + eps_t; R estimated by EM-style iterations with Student-t robust weights"
    )
    state_summary = pd.DataFrame(
        {
            "item": [
                "retained_bases_R",
                "state_dimension_R_times_Rplus1_over_2",
                "A_t_model",
                "alpha_hat",
                "state_equation",
                "observation_equation",
                "state_uncertainty_draws",
                "bootstrap_draws",
                "bootstrap_block_length_months",
                "student_t_degrees_of_freedom",
                "mean_trace_A_over_R",
                "sd_trace_A_over_R",
                "max_trace_A_over_R",
                "max_month",
                "min_eigenvalue_across_A_t",
                "top3_trace_share_estimate",
                "top5_trace_share_estimate",
            ],
            "value": [
                HEADLINE_R,
                HEADLINE_R * (HEADLINE_R + 1) // 2,
                model_label,
                headline.fit.alpha,
                "s_t = mu + F(s_{t-1}-mu) + eta_t; F,Q estimated by EM-style iterations with structured shrinkage",
                observation_label,
                B_STATE,
                B_BOOT,
                BOOT_BLOCK_LEN,
                ROBUST_NU,
                headline.tau.mean(),
                headline.tau.std(ddof=0),
                headline.tau.max(),
                dates.iloc[int(np.argmax(headline.tau))].strftime("%Y-%m"),
                min_eig_A,
                headline.shares[:3].sum(),
                headline.shares[:5].sum(),
            ],
        }
    )
    sample_summary = build_publication_sample_coverage(panel, baseline_scores)
    legacy_rows = pd.DataFrame(
        {
            "item": [
                "FRED_monthly_rows",
                "JK_monthly_shock_rows",
                "LP_usable_base_months",
                "LP_base_month_range",
                "outcomes",
                "horizons",
                "lags",
                "shock",
                "control_shock",
                "residualized_shock_variance",
            ],
            "value": [
                len(panels["panel"]),
                len(panels.get("monthly", [])),
                len(dates),
                f"{dates.iloc[0].date()} to {dates.iloc[-1].date()}",
                ", ".join(labels),
                f"0 to {H} months",
                L,
                "MP_median with MP_pm fallback when missing",
                "CBI_median with CBI_pm fallback when missing",
                baseline_scores["sigma_m2"],
            ],
        }
    )
    sample_summary = pd.concat([sample_summary, legacy_rows], ignore_index=True)
    state_summary.to_csv(top_tables / "top5_state_space_model_summary.csv", index=False)
    sample_summary.to_csv(top_tables / "sample_and_specification_top5.csv", index=False)
    pd.DataFrame(headline.fit.F).to_csv(top_tables / "top5_state_VAR_F_matrix.csv", index=False)
    pd.DataFrame(headline.fit.Q).to_csv(top_tables / "top5_state_process_covariance_Q.csv", index=False)
    pd.DataFrame(headline.fit.Rmat).to_csv(top_tables / "top5_state_measurement_covariance_R.csv", index=False)
    pd.DataFrame([{"note": "Deprecated legacy top-five estimator was not run; these outputs are generated by publication_grade_ovk with the upgraded algorithm."}]).to_csv(
        TOP5_COMPAT_OUT / "quick_summary_top5.csv", index=False
    )
    (TOP5_COMPAT_OUT / "README_top5_dynamic_state_model.txt").write_text(
        "Deprecated top-five output surface produced by publication_grade_ovk.\n"
        +
        (
            "All estimation uses the arithmetic outer-product second-moment state-space core by default: "
            "reference-weighted whitening, direct svec(u_t u_t') observations, EM-style F/Q/R, "
            "Student-t robust filtering, FFBS state bands, and full-pipeline bootstrap. "
            "Set OVK_COVARIANCE_ESTIMATOR_MODE=log_spd_legacy only for deprecated comparison runs.\n"
            if headline.estimator_mode == "arithmetic_outer_product"
            else "All estimation uses the deprecated robust log-Euclidean state-space core for comparison only.\n"
        ),
        encoding="utf-8",
    )

    def save_compat(fig: plt.Figure, name: str) -> Path:
        path = top_charts / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return path

    h = np.arange(H + 1)
    low = beta_band[0].reshape(H + 1, pvars)
    high = beta_band[2].reshape(H + 1, pvars)
    fig = plt.figure(figsize=(10, 5.8))
    for j, label in enumerate(labels):
        line = plt.plot(h, beta_surface[:, j], marker="o", label=label)[0]
        plt.fill_between(h, low[:, j], high[:, j], alpha=0.10, color=line.get_color())
    plt.axhline(0, linewidth=0.8)
    plt.title("Average LP responses with publication-grade full-pipeline bootstrap bands")
    plt.xlabel("Horizon, months")
    plt.ylabel("Response")
    plt.legend()
    save_compat(fig, "01_irf_all_variables_90pct_bands")

    x = np.arange(1, 11)
    fig = plt.figure(figsize=(8.8, 5.3))
    plt.bar(x, headline.shares[:10])
    plt.errorbar(
        x,
        headline.shares[:10],
        yerr=[
            np.maximum(headline.shares[:10] - share_band[0], 0),
            np.maximum(share_band[2] - headline.shares[:10], 0),
        ],
        fmt="none",
        capsize=3,
    )
    plt.axvline(5, linestyle="--", linewidth=1.0)
    plt.title("Publication-grade OVK eigenspectrum")
    plt.xlabel("Rank")
    plt.ylabel("Trace share")
    save_compat(fig, "02_eigenspectrum_share_top5_90pct_bands")

    fig = plt.figure(figsize=(8.8, 5.3))
    plt.plot(x, np.cumsum(headline.shares[:10]), marker="o")
    plt.fill_between(x, cum_band[0], cum_band[2], alpha=0.18)
    plt.axvline(5, linestyle="--", linewidth=1.0)
    plt.ylim(0, 1.02)
    plt.title("Cumulative retained trace share")
    plt.xlabel("Rank")
    plt.ylabel("Cumulative share")
    save_compat(fig, "03_cumulative_trace_share_top5_90pct_band")

    fig = plt.figure(figsize=(10.5, 5.6))
    plt.plot(dates, headline.tau, label="estimate", color="black")
    plt.fill_between(dates, tau_band90[0], tau_band90[2], alpha=0.24, label="90% pointwise FFBS state band")
    plt.plot(dates, path_df["tau_simul_p05"], linestyle="--", linewidth=1.1, label="90% log-simultaneous FFBS lower")
    plt.plot(dates, path_df["tau_simul_p95"], linestyle="--", linewidth=1.1, label="90% log-simultaneous FFBS upper")
    plt.axhline(1.0, linewidth=0.8)
    plt.legend()
    plt.title("Top-five total kernel amplification: FFBS state uncertainty")
    plt.ylabel("trace(A_t)/5")
    save_compat(fig, "04_total_kernel_amplification_top5_state_bands")

    fig = plt.figure(figsize=(10.5, 5.6))
    plt.plot(dates, headline.tau, label="estimate", color="black")
    plt.fill_between(dates, path_df["tau_full_pipeline_p05"], path_df["tau_full_pipeline_p95"], alpha=0.24, label="90% pointwise full-pipeline band")
    plt.plot(dates, path_df["tau_full_pipeline_simul_p05"], linestyle="--", linewidth=1.1, label="90% log-simultaneous full-pipeline lower")
    plt.plot(dates, path_df["tau_full_pipeline_simul_p95"], linestyle="--", linewidth=1.1, label="90% log-simultaneous full-pipeline upper")
    plt.axhline(1.0, linewidth=0.8)
    plt.legend()
    plt.title("Top-five total kernel amplification: full-pipeline uncertainty")
    plt.ylabel("trace(A_t)/5")
    save_compat(fig, "04b_total_kernel_amplification_top5_full_pipeline_bands")

    fig = plt.figure(figsize=(10.5, 5.6))
    for r in range(HEADLINE_R):
        plt.plot(dates, headline.A[:, r, r], label=f"A{r+1}{r+1}")
    plt.axhline(1.0, linewidth=0.8)
    plt.legend()
    plt.title("All top-five A_t diagonal terms")
    save_compat(fig, "05_A_diagonals_all_top5_state_bands")

    for r in range(HEADLINE_R):
        surface = headline.V[:, r].reshape(H + 1, pvars)
        lo = mode_band[0, r].reshape(H + 1, pvars)
        hi = mode_band[2, r].reshape(H + 1, pvars)
        fig = plt.figure(figsize=(10, 5.8))
        for j, label in enumerate(labels):
            line = plt.plot(h, surface[:, j], marker="o", label=label)[0]
            plt.fill_between(h, lo[:, j], hi[:, j], alpha=0.12, color=line.get_color())
        plt.axhline(0, linewidth=0.8)
        plt.title(f"Basis {r+1} loadings with publication-grade bootstrap bands")
        plt.xlabel("Horizon, months")
        plt.legend()
        save_compat(fig, f"06_basis_{r+1}_loadings_all_variables_90pct_bands")

        row = basis_diag_df.iloc[r]
        vals = np.array([row[f"{safe_name(label)}_energy_estimate"] for label in labels])
        loe = np.array([row[f"{safe_name(label)}_energy_p05"] for label in labels])
        hie = np.array([row[f"{safe_name(label)}_energy_p95"] for label in labels])
        xx = np.arange(pvars)
        fig = plt.figure(figsize=(9, 5.4))
        plt.bar(xx, vals)
        plt.errorbar(xx, vals, yerr=[np.maximum(vals - loe, 0), np.maximum(hie - vals, 0)], fmt="none", capsize=3)
        plt.xticks(xx, labels, rotation=25, ha="right")
        plt.ylabel("Energy share")
        plt.title(f"Basis {r+1} variable energy shares")
        save_compat(fig, f"08_basis_{r+1}_variable_energy_shares_90pct_bands")

    fig = plt.figure(figsize=(8.8, 5.3))
    plt.bar(mode_stability["basis"], mode_stability["median_abs_corr_with_bootstrap_basis"])
    plt.ylim(0, 1.02)
    plt.xlabel("Basis")
    plt.ylabel("Median absolute alignment")
    plt.title("Full-pipeline bootstrap stability of top five bases")
    save_compat(fig, "10_top5_basis_bootstrap_stability")

    copy_tree_contents(TOP5_COMPAT_OUT, OUT / "top5_compatible_outputs")
    shutil.copy2(FINAL_PDF, TOP5_COMPAT_PDF)
    shutil.copy2(FINAL_HTML, TOP5_COMPAT_HTML)
    if TOP5_COMPAT_ZIP.exists():
        TOP5_COMPAT_ZIP.unlink()
    with zipfile.ZipFile(TOP5_COMPAT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(TOP5_COMPAT_OUT.rglob("*")):
            if f.is_file():
                z.write(f, arcname=f.relative_to(TOP5_COMPAT_OUT))
        z.write(TOP5_COMPAT_PDF, arcname=TOP5_COMPAT_PDF.name)
        z.write(TOP5_COMPAT_HTML, arcname=TOP5_COMPAT_HTML.name)


def write_robustness_compatible_outputs(panels: dict[str, pd.DataFrame], results: dict[tuple[str, int], RankResult]) -> None:
    reset_dir(ROBUST_COMPAT_OUT)
    tables = ROBUST_COMPAT_OUT / "tables"
    charts = ROBUST_COMPAT_OUT / "charts"
    code_dir = ROBUST_COMPAT_OUT / "code"
    for d in [tables, charts, code_dir]:
        d.mkdir(parents=True, exist_ok=True)
    panel = panels["panel"].copy()
    panel.to_csv(ROBUST_COMPAT_OUT / "processed_panel_three_shock_definitions.csv", index=False)
    panels.get("monthly", pd.DataFrame()).to_csv(ROBUST_COMPAT_OUT / "monthly_shocks_repaired.csv", index=False)
    panels.get("events", pd.DataFrame()).to_csv(ROBUST_COMPAT_OUT / "event_shocks_with_manual_fields.csv", index=False)
    panels.get("event_monthly", pd.DataFrame()).to_csv(ROBUST_COMPAT_OUT / "event_level_manual_monthly_aggregation.csv", index=False)

    legacy_defs = [
        ("base5_headline", "median_fallback", "MP_median with fallback"),
        ("base5_mp_pm_only", "mp_pm_only", "MP_pm only"),
        ("base5_event_manual", "event_manual", "Event-level shocks aggregated manually"),
    ]
    base = results[("base5_headline", HEADLINE_R)]
    summary_rows = []
    for key, legacy_key, legacy_label in legacy_defs:
        res = results[(key, HEADLINE_R)]
        labels = list(res.outcome_labels)
        summary_rows.append(
            {
                "variant": legacy_key,
                "source_variant": key,
                "label": legacy_label,
                "n_valid": int(len(res.dates)),
                "sample_start": str(pd.to_datetime(res.dates.iloc[0]).date()),
                "sample_end": str(pd.to_datetime(res.dates.iloc[-1]).date()),
                "top1_trace_share": float(res.shares[0]),
                "top3_trace_share": float(res.shares[:3].sum()),
                "top5_trace_share": float(res.shares[:5].sum()),
                "tau_mean": float(res.tau.mean()),
                "tau_sd": float(res.tau.std(ddof=0)),
                "tau_max": float(res.tau.max()),
                "tau_max_month": pd.to_datetime(res.dates.iloc[int(np.argmax(res.tau))]).strftime("%Y-%m"),
                "state_spectral_radius": res.fit.spectral_radius,
                "min_A_eigenvalue": float(np.min(np.linalg.eigvalsh(res.A))),
                "alpha_hat": res.fit.alpha,
                "robust_loglik": res.fit.robust_loglik,
            }
        )
        pd.DataFrame(res.beta.reshape(H + 1, len(labels)), columns=labels).assign(horizon_months=np.arange(H + 1)).to_csv(
            tables / f"{legacy_key}_average_irf.csv", index=False
        )
        pd.DataFrame({"date": res.dates, "tau": res.tau, **{f"A{j+1}{j+1}": res.A[:, j, j] for j in range(HEADLINE_R)}}).to_csv(
            tables / f"{legacy_key}_tau_and_A_diagonals.csv", index=False
        )
        basis_energy_df(res).to_csv(tables / f"{legacy_key}_basis_energy.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "robustness_variant_summary.csv", index=False)

    comp_rows, diag_rows, top_rows, basis_match_rows = [], [], [], []
    base_top10 = set(pd.to_datetime(base.dates.iloc[np.argsort(base.tau)[::-1][:10]]).dt.strftime("%Y-%m"))
    for key, legacy_key, legacy_label in legacy_defs:
        res = results[(key, HEADLINE_R)]
        angles = principal_angles(base.V, res.V)
        C = np.abs(base.V.T @ res.V)
        row_ind, col_ind = linear_sum_assignment(-C)
        order = np.zeros(HEADLINE_R, dtype=int)
        corr = np.zeros(HEADLINE_R)
        for r, c in zip(row_ind, col_ind):
            order[r] = c
            corr[r] = C[r, c]
        common = pd.DataFrame({"date": pd.to_datetime(base.dates), "tau_base": base.tau}).merge(
            pd.DataFrame({"date": pd.to_datetime(res.dates), "tau_variant": res.tau}), on="date", how="inner"
        )
        tau_corr = 1.0 if key == "base5_headline" else float(np.corrcoef(common["tau_base"], common["tau_variant"])[0, 1])
        base_index = pd.Series(np.arange(len(base.dates)), index=pd.to_datetime(base.dates))
        res_index = pd.Series(np.arange(len(res.dates)), index=pd.to_datetime(res.dates))
        bidx = np.array([base_index[pd.to_datetime(d)] for d in common["date"]])
        ridx = np.array([res_index[pd.to_datetime(d)] for d in common["date"]])
        diag_corr = []
        for j in range(HEADLINE_R):
            cval = 1.0 if key == "base5_headline" else float(np.corrcoef(base.A[bidx, j, j], res.A[ridx, order[j], order[j]])[0, 1])
            diag_corr.append(cval)
            row = {
                "variant": legacy_key,
                "source_variant": key,
                "label": legacy_label,
                "baseline_basis": j + 1,
                "matched_variant_basis": int(order[j] + 1),
                "basis_vector_abs_corr": float(corr[j]),
                "A_diag_path_corr": cval,
            }
            diag_rows.append(row)
            basis_match_rows.append({k: row[k] for k in ["variant", "label", "baseline_basis", "matched_variant_basis", "basis_vector_abs_corr"]})
        res_top10 = set(pd.to_datetime(res.dates.iloc[np.argsort(res.tau)[::-1][:10]]).dt.strftime("%Y-%m"))
        march_mask = pd.to_datetime(res.dates).dt.strftime("%Y-%m") == "2020-03"
        comp_rows.append(
            {
                "variant": legacy_key,
                "source_variant": key,
                "label": legacy_label,
                "top5_trace_share": float(res.shares[:5].sum()),
                "top5_trace_share_diff_vs_baseline": float(res.shares[:5].sum() - base.shares[:5].sum()),
                "max_principal_angle_degrees": float(np.max(angles)),
                "mean_principal_angle_degrees": float(np.mean(angles)),
                **{f"angle_{i+1}_degrees": float(angles[i]) for i in range(HEADLINE_R)},
                "tau_path_corr_with_baseline": tau_corr,
                "top10_overlap_with_baseline": int(len(base_top10 & res_top10)),
                "top10_overlap_months": ", ".join(sorted(base_top10 & res_top10)),
                "march_2020_tau": float(res.tau[march_mask.to_numpy()][0]) if march_mask.any() else np.nan,
                "march_2020_rank": int(np.where(np.argsort(res.tau)[::-1] == np.where(march_mask.to_numpy())[0][0])[0][0] + 1) if march_mask.any() else np.nan,
                **{f"A{j+1}{j+1}_diag_corr_with_baseline": diag_corr[j] for j in range(HEADLINE_R)},
            }
        )
        top_idx = np.argsort(res.tau)[::-1][:15]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "variant": legacy_key,
                    "source_variant": key,
                    "label": legacy_label,
                    "rank": rank,
                    "date": pd.to_datetime(res.dates.iloc[idx]).strftime("%Y-%m"),
                    "tau": float(res.tau[idx]),
                    **{f"A{j+1}{j+1}": float(res.A[idx, j, j]) for j in range(HEADLINE_R)},
                }
            )
    comp = pd.DataFrame(comp_rows)
    diag_df = pd.DataFrame(diag_rows)
    basis_match = pd.DataFrame(basis_match_rows)
    top_months = pd.DataFrame(top_rows)
    comp.to_csv(tables / "robustness_comparison_metrics.csv", index=False)
    diag_df.to_csv(tables / "basis_specific_A_diag_path_correlations.csv", index=False)
    basis_match.to_csv(tables / "basis_matching_to_baseline.csv", index=False)
    top_months.to_csv(tables / "top15_amplification_months_by_variant.csv", index=False)
    pd.concat([basis_energy_df(results[(key, HEADLINE_R)]).assign(variant=legacy_key, source_variant=key, label=legacy_label) for key, legacy_key, legacy_label in legacy_defs]).to_csv(
        tables / "basis_energy_all_variants.csv", index=False
    )

    diff = panel[["date", "MP_median_fallback", "CBI_median_fallback", "MP_event_manual", "CBI_event_manual", "fallback_flag", "fallback_event", "mixed_missing_and_nonmissing_events"]].copy()
    diff["MP_baseline_minus_event_manual"] = diff["MP_median_fallback"] - diff["MP_event_manual"]
    diff["CBI_baseline_minus_event_manual"] = diff["CBI_median_fallback"] - diff["CBI_event_manual"]
    diff_months = diff[(diff["MP_baseline_minus_event_manual"].abs() > 1e-12) | (diff["CBI_baseline_minus_event_manual"].abs() > 1e-12)].copy()
    diff.to_csv(tables / "shock_definition_monthly_vs_event_manual_all_months.csv", index=False)
    diff_months.to_csv(tables / "months_where_monthly_fallback_differs_from_event_manual.csv", index=False)

    def save_robust(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(charts / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    plt.figure(figsize=(8.5, 5))
    plt.bar(comp["label"], comp["top5_trace_share"])
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Top-five trace share")
    plt.title("Publication-grade top-five trace share by shock definition")
    save_robust(plt.gcf(), "robustness_top5_trace_share_by_variant")

    plt.figure(figsize=(10, 5.5))
    for key, _, legacy_label in legacy_defs:
        res = results[(key, HEADLINE_R)]
        plt.plot(res.dates, res.tau, label=legacy_label)
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t = trace(A_t)/5")
    plt.title("Publication-grade total amplification by shock definition")
    plt.legend()
    save_robust(plt.gcf(), "robustness_tau_paths_by_variant")

    plt.figure(figsize=(8.5, 5))
    for _, row in comp.iterrows():
        vals = [row[f"angle_{j+1}_degrees"] for j in range(HEADLINE_R)]
        plt.plot(np.arange(1, HEADLINE_R + 1), vals, marker="o", label=row["label"])
    plt.xlabel("Principal angle index")
    plt.ylabel("Degrees")
    plt.title("Leading five-dimensional subspace angles vs baseline")
    plt.legend()
    save_robust(plt.gcf(), "robustness_principal_angles_vs_baseline")

    for j in range(HEADLINE_R):
        plt.figure(figsize=(10, 5.5))
        for key, _, legacy_label in legacy_defs:
            res = results[(key, HEADLINE_R)]
            if key == "base5_headline":
                series = res.A[:, j, j]
            else:
                C = np.abs(base.V.T @ res.V)
                _, col_ind = linear_sum_assignment(-C)
                series = res.A[:, col_ind[j], col_ind[j]]
            plt.plot(res.dates, series, label=legacy_label)
        plt.axhline(1.0, linewidth=0.8)
        plt.ylabel(f"A{j+1}{j+1}, matched basis {j+1}")
        plt.title(f"Basis-specific A_t diagonal path: basis {j+1}")
        plt.legend()
        save_robust(plt.gcf(), f"robustness_A{j+1}{j+1}_path_by_variant")

    plt.figure(figsize=(9, 5.4))
    if len(diff_months):
        dd = diff_months.copy()
        dd["date_label"] = pd.to_datetime(dd["date"]).dt.strftime("%Y-%m")
        x = np.arange(len(dd))
        plt.bar(x - 0.18, dd["MP_baseline_minus_event_manual"], width=0.36, label="MP baseline - event manual")
        plt.bar(x + 0.18, dd["CBI_baseline_minus_event_manual"], width=0.36, label="CBI baseline - event manual")
        plt.xticks(x, dd["date_label"], rotation=35, ha="right")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No monthly fallback/event-manual differences", ha="center", va="center", transform=plt.gca().transAxes)
        plt.xticks([])
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("Shock difference")
    plt.title("Monthly fallback versus event-level manual aggregation")
    save_robust(plt.gcf(), "shock_definition_monthly_fallback_vs_event_manual_differences")

    (ROBUST_COMPAT_OUT / "robustness_interpretation.txt").write_text(
        "Shock-definition robustness produced by publication_grade_ovk using the upgraded estimator. "
        "Legacy fixed-alpha robustness estimation is deprecated.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), code_dir / "run_publication_grade_ovk.py")
    helper = Path(__file__).with_name("ovk_data.py")
    if helper.exists():
        shutil.copy2(helper, code_dir / "ovk_data.py")


def variant_registry_table(specs: list[VariantSpec], sf_meta: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": spec.key,
                "label": spec.label,
                "group": spec.group,
                "outcome_columns": ", ".join(spec.outcome_columns),
                "shock_col": spec.shock_col,
                "control_col": spec.cbi_col or "",
                "sample_dates_key": spec.sample_dates_key or "natural",
                "transform": spec.transform,
                "run_ranks": ", ".join(str(x) for x in spec.run_ranks),
                "available": spec.available,
                "skip_reason": spec.skip_reason,
                "source_note": spec.source_note,
                "sf_fed_source_file": sf_meta.get("sf_fed_path", "") if spec.group == "sf_fed" else "",
                "sf_fed_primary_source": sf_meta.get("sf_fed_primary_source", "") if spec.group == "sf_fed" else "",
                "sf_fed_primary_source_updated": sf_meta.get("sf_fed_primary_source_updated", "") if spec.group == "sf_fed" else "",
            }
            for spec in specs
        ]
    )


def compact_metric_row(res: RankResult, spec: VariantSpec | None = None, base: RankResult | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": res.variant,
        "label": res.label if spec is None else spec.label,
        "group": spec.group if spec is not None else "",
        "estimator_mode": res.estimator_mode,
        "n_valid": int(len(res.dates)),
        "sample_start": pd.to_datetime(res.dates.iloc[0]).strftime("%Y-%m"),
        "sample_end": pd.to_datetime(res.dates.iloc[-1]).strftime("%Y-%m"),
        "outcome_count": int(len(res.outcome_labels)),
        "outcomes": ", ".join(res.outcome_labels),
        "top1_trace_share": float(res.shares[0]),
        "top3_trace_share": float(res.shares[:3].sum()),
        "top5_trace_share": float(res.shares[:5].sum()),
        "factor_log_score": float(res.fit.factor_log_score),
        "avg_factor_log_score_per_rank_dim": float(res.fit.factor_log_score / (len(res.dates) * res.rank)),
        "tau_mean": float(res.tau.mean()),
        "tau_sd": float(res.tau.std(ddof=0)),
        "tau_max": float(res.tau.max()),
        "tau_max_month": pd.to_datetime(res.dates.iloc[int(np.argmax(res.tau))]).strftime("%Y-%m"),
        "alpha_hat": float(res.fit.alpha),
        "reference_whitening_ridge": float(res.whitening_ridge),
        "state_spectral_radius": float(res.fit.spectral_radius),
    }
    if base is not None:
        common = pd.DataFrame({"date": pd.to_datetime(base.dates), "tau_base": base.tau}).merge(
            pd.DataFrame({"date": pd.to_datetime(res.dates), "tau_variant": res.tau}),
            on="date",
            how="inner",
        )
        if len(common) >= 3:
            row["tau_path_corr_with_base5_headline"] = float(np.corrcoef(common["tau_base"], common["tau_variant"])[0, 1])
            base_top10 = set(pd.to_datetime(base.dates.iloc[np.argsort(base.tau)[::-1][:10]]).dt.strftime("%Y-%m"))
            res_top10 = set(pd.to_datetime(res.dates.iloc[np.argsort(res.tau)[::-1][:10]]).dt.strftime("%Y-%m"))
            row["top10_overlap_with_base5_headline"] = int(len(base_top10 & res_top10))
    return row


def write_requested_robustness_tables(
    specs: list[VariantSpec],
    score_data: dict[str, dict[str, Any]],
    results: dict[tuple[str, int], RankResult],
    bootstrap_outputs: dict[str, dict[str, Any]],
    sf_meta: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    spec_map = {s.key: s for s in specs}
    base = results[("base5_headline", HEADLINE_R)]
    registry = variant_registry_table(specs, sf_meta)
    registry.to_csv(TABLES / "publication_grade_variant_registry.csv", index=False)

    def have(key: str) -> bool:
        return (key, HEADLINE_R) in results

    same_keys = ["base5_headline", "base5_expectation_overlap", "all8_expectation_overlap"]
    same_sample = pd.DataFrame([compact_metric_row(results[(k, HEADLINE_R)], spec_map[k], base=base) for k in same_keys if have(k)])
    same_sample.to_csv(TABLES / "same_sample_outcome_comparison.csv", index=False)

    std_pairs = [
        ("base5_headline", "base5_headline_standardized"),
        ("base5_expectation_overlap", "base5_expectation_overlap_standardized"),
        ("all8_expectation_overlap", "all8_expectation_overlap_standardized"),
    ]
    std_rows = []
    weight_rows = []
    for raw_key, std_key in std_pairs:
        if not have(raw_key) or not have(std_key):
            continue
        raw = results[(raw_key, HEADLINE_R)]
        std = results[(std_key, HEADLINE_R)]
        raw_traces = outcome_block_traces(score_data[raw_key]["Q_scores_raw"], raw.outcome_labels)
        std_traces = outcome_block_traces(std.Q_scores, std.outcome_labels)
        std_rows.append(
            {
                "raw_variant": raw_key,
                "standardized_variant": std_key,
                "raw_top5_trace_share": float(raw.shares[:5].sum()),
                "standardized_top5_trace_share": float(std.shares[:5].sum()),
                "raw_tau_sd": float(raw.tau.std(ddof=0)),
                "standardized_tau_sd": float(std.tau.std(ddof=0)),
                "raw_total_trace": float(raw_traces.sum()),
                "standardized_total_trace": float(std_traces.sum()),
                "standardized_min_block_trace": float(std_traces.min()),
                "standardized_max_block_trace": float(std_traces.max()),
                "standardized_block_trace_spread": float(std_traces.max() - std_traces.min()),
            }
        )
        transform = score_data[std_key].get("score_transform", {})
        weights = transform.get("weights_by_outcome", [])
        target = transform.get("target_trace", np.nan)
        for label, raw_trace, std_trace, weight in zip(std.outcome_labels, raw_traces, std_traces, weights):
            weight_rows.append(
                {
                    "standardized_variant": std_key,
                    "outcome": label,
                    "raw_block_trace": raw_trace,
                    "target_block_trace": target,
                    "weight": weight,
                    "standardized_block_trace": std_trace,
                }
            )
    raw_std = pd.DataFrame(std_rows)
    weights_df = pd.DataFrame(weight_rows)
    raw_std.to_csv(TABLES / "kernel_raw_vs_standardized_summary.csv", index=False)
    weights_df.to_csv(TABLES / "outcome_trace_standardization_weights.csv", index=False)

    placebo_keys = ["base5_headline", "placebo_permuted", "placebo_shift84"]
    placebo = pd.DataFrame([compact_metric_row(results[(k, HEADLINE_R)], spec_map.get(k), base=base) for k in placebo_keys if have(k)])
    placebo.to_csv(TABLES / "placebo_shock_comparison.csv", index=False)

    policy_keys = ["base5_headline", "policy_without_cbi", "cbi_with_policy", "cbi_without_policy"]
    policy = pd.DataFrame([compact_metric_row(results[(k, HEADLINE_R)], spec_map.get(k), base=base) for k in policy_keys if have(k)])
    policy.to_csv(TABLES / "policy_cbi_split_comparison.csv", index=False)

    sf_keys = ["sf_fed_raw", "sf_fed_orthogonalized"]
    sf_rows = []
    for key in sf_keys:
        if have(key):
            sf_rows.append({**compact_metric_row(results[(key, HEADLINE_R)], spec_map.get(key), base=base), **sf_meta})
        else:
            sf_rows.append(
                {
                    "variant": key,
                    "label": spec_map[key].label if key in spec_map else key,
                    "status": "skipped",
                    "skip_reason": spec_map[key].skip_reason if key in spec_map else "not registered",
                    **sf_meta,
                }
            )
    sf_df = pd.DataFrame(sf_rows)
    sf_df.to_csv(TABLES / "sf_fed_shock_comparison.csv", index=False)

    smooth_rows = []
    for key in ["base5_headline", "base5_headline_smooth"]:
        if have(key):
            row = compact_metric_row(results[(key, HEADLINE_R)], spec_map.get(key), base=base)
            row["score_surface_roughness"] = score_roughness(score_data[key]["Q_scores"], results[(key, HEADLINE_R)].outcome_labels)
            smooth_rows.append(row)
    smooth = pd.DataFrame(smooth_rows)
    smooth.to_csv(TABLES / "smooth_lp_comparison.csv", index=False)

    episode = episode_spike_uncertainty(base, bootstrap_outputs.get("base5_headline", {}).get("tau_boot_valid"))
    episode.to_csv(TABLES / "episode_spike_uncertainty.csv", index=False)

    return {
        "registry": registry,
        "same_sample": same_sample,
        "raw_std": raw_std,
        "weights": weights_df,
        "placebo": placebo,
        "policy_cbi": policy,
        "sf_fed": sf_df,
        "smooth": smooth,
        "episode": episode,
    }


def episode_spike_uncertainty(headline: RankResult, tau_boot_valid: np.ndarray | None) -> pd.DataFrame:
    dates = pd.to_datetime(headline.dates).reset_index(drop=True)
    tau_boot = np.asarray(tau_boot_valid if tau_boot_valid is not None else np.empty((0, len(dates))), float)
    rows = []
    for episode, start, end in EPISODES:
        start_ts = pd.Timestamp(f"{start}-01")
        end_ts = pd.Timestamp(f"{end}-01")
        mask = (dates >= start_ts) & (dates <= end_ts)
        idx = np.where(mask.to_numpy())[0]
        point_max = float(np.nanmax(headline.tau[idx])) if len(idx) else np.nan
        point_month = dates.iloc[idx[int(np.nanargmax(headline.tau[idx]))]].strftime("%Y-%m") if len(idx) else ""
        if tau_boot.size and len(idx):
            draw_ranks = np.argsort(np.argsort(-tau_boot, axis=1), axis=1) + 1
            episode_max = np.nanmax(tau_boot[:, idx], axis=1)
            rows.append(
                {
                    "episode": episode,
                    "start_month": start,
                    "end_month": end,
                    "months_in_sample": int(len(idx)),
                    "point_episode_max_tau": point_max,
                    "point_episode_max_month": point_month,
                    "probability_any_month_in_top10": float((draw_ranks[:, idx] <= 10).any(axis=1).mean()),
                    "median_episode_max_tau": float(np.nanmedian(episode_max)),
                    "episode_max_tau_p05": float(np.nanpercentile(episode_max, 5)),
                    "episode_max_tau_p95": float(np.nanpercentile(episode_max, 95)),
                    "bootstrap_draws": int(tau_boot.shape[0]),
                }
            )
        else:
            rows.append(
                {
                    "episode": episode,
                    "start_month": start,
                    "end_month": end,
                    "months_in_sample": int(len(idx)),
                    "point_episode_max_tau": point_max,
                    "point_episode_max_month": point_month,
                    "probability_any_month_in_top10": np.nan,
                    "median_episode_max_tau": np.nan,
                    "episode_max_tau_p05": np.nan,
                    "episode_max_tau_p95": np.nan,
                    "bootstrap_draws": 0,
                }
            )
    return pd.DataFrame(rows)


def build_report(
    summary: pd.DataFrame,
    path_df: pd.DataFrame,
    uncertainty: pd.DataFrame,
    shock_env: pd.DataFrame,
    boot_summary: pd.DataFrame,
    charts: dict[str, Path],
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    extra_tables = extra_tables or {}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=21, spaceAfter=12))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.0, leading=11.5, spaceAfter=5))
    styles.add(ParagraphStyle(name="Captionx", parent=styles["BodyText"], fontSize=7.5, leading=8.5, alignment=TA_CENTER, spaceAfter=6))

    def P(text: str, style: str = "Bodyx") -> Paragraph:
        return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), styles[style])

    def img(path: Path, max_w: float = 6.8 * inch, max_h: float = 4.0 * inch) -> Image:
        from PIL import Image as PILImage

        im = PILImage.open(path)
        scale = min(max_w / im.width, max_h / im.height)
        return Image(str(path), width=im.width * scale, height=im.height * scale)

    mode = str(summary["estimator_mode"].dropna().iloc[0]) if "estimator_mode" in summary.columns and len(summary) else ""
    estimator_intro = (
        "Full-coordinate ridge-soft covariance amplification for the Section 3.1 figures, with a separate temporal-kernel covariance backend and moving-block bootstrap bands. Rank-reduced fits are retained only for comparison and appendix diagnostics."
    )
    estimator_detail = (
        "The headline figures use the full 125-coordinate LP working grid: C_hat is the average outer product of chi_t, D_rho=C_hat+rho I supplies soft ridge whitening, and tau_soft(t)=tr(K_hat(t) solve(D_rho,I))/d_rho. The temporal-kernel backend uses normalized nonnegative weights, so its fitted K_hat path averages back to C_hat; no spectral cutoff enters Figures 1-2."
    )

    summary_cols = ["variant", "group", "rank", "outcome_count", "retained_trace_share", "tau_sd", "tau_max_month", "factor_log_score"]
    same_sample_cols = ["variant", "outcome_count", "sample_start", "sample_end", "top5_trace_share", "tau_sd", "tau_max_month", "top10_overlap_with_base5_headline"]
    raw_std_cols = ["raw_variant", "standardized_variant", "raw_top5_trace_share", "standardized_top5_trace_share", "raw_tau_sd", "standardized_tau_sd", "standardized_block_trace_spread"]
    placebo_cols = ["variant", "top5_trace_share", "tau_sd", "tau_max_month", "tau_path_corr_with_base5_headline", "top10_overlap_with_base5_headline"]
    policy_cbi_cols = ["variant", "top5_trace_share", "tau_sd", "tau_max_month", "tau_path_corr_with_base5_headline", "top10_overlap_with_base5_headline"]
    boot_cols = ["variant", "bootstrap_draws_valid", "trace_share_p05", "trace_share_p95", "max_tau_p05", "max_tau_p95", "march_2020_top10_probability_full_pipeline"]
    shock_env_cols = ["date", "tau_base5_headline", "tau_base5_mp_pm_only", "tau_base5_event_manual", "tau_sf_fed_raw", "tau_sf_fed_orthogonalized", "tau_spec_min", "tau_spec_max", "tau_spec_range"]
    sf_fed_cols = ["variant", "n_valid", "sample_end", "top5_trace_share", "tau_sd", "tau_max_month", "tau_path_corr_with_base5_headline", "sf_fed_orthogonalized_on"]
    smooth_cols = ["variant", "top5_trace_share", "tau_sd", "tau_max_month", "score_surface_roughness", "tau_path_corr_with_base5_headline"]
    episode_cols = ["episode", "start_month", "end_month", "point_episode_max_tau", "probability_any_month_in_top10", "median_episode_max_tau", "episode_max_tau_p05", "episode_max_tau_p95"]

    story = [P("Publication-Grade Full-Coordinate Monetary-Policy Response-Score Covariance", "TitleCenter")]
    story.append(P(estimator_intro + " The headline estimand is a time-varying response-score covariance, not an unqualified state-dependent structural IRF."))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    sample_coverage = extra_tables.get("sample_coverage", pd.DataFrame())
    if not sample_coverage.empty:
        complete_range = sample_value(sample_coverage, "Common complete-coordinate score coverage")
        raw_window = sample_value(sample_coverage, "Raw outcome window consumed by score sample")
        story.append(P("Sample Calendar", "H1x"))
        story.append(
            P(
                "The state index is attached to the base month t. "
                f"In this run the common complete-coordinate score coverage is {complete_range}; "
                f"the LP construction consumes raw outcome data over {raw_window}."
            )
        )
        story.append(make_table(sample_coverage, max_rows=30))
    story.append(P("Empirical anchor", "H1x"))
    story.append(P("Dynamic covariance wins; dynamic mean does not. The nested mean-covariance workflow is the empirical anchor: it separates a moving LP-response center from a dynamic covariance operator of LP surfaces, and the paper should repeatedly return to that distinction."))
    story.append(P("Headline estimation changes", "H1x"))
    story.append(P(estimator_detail + " The state-space core estimates F, Q, and R by EM-style iterations, applies structured transition shrinkage, and downweights extreme observations with Student-t filtering."))
    story.append(make_table(summary, max_rows=12, cols=summary_cols))
    if "same_sample" in extra_tables and not extra_tables["same_sample"].empty:
        story.append(P("Same-sample outcome comparison", "H1x"))
        story.append(P("This isolates whether changes in trace share and tau_t come from adding expectation outcomes or from changing the score-surface sample."))
        story.append(make_table(extra_tables["same_sample"], max_rows=8, cols=same_sample_cols))
    if "raw_std" in extra_tables and not extra_tables["raw_std"].empty:
        story.append(P("Raw versus standardized outcome-trace scaling", "H1x"))
        story.append(P("Raw scaling answers which outcomes dominate the empirical covariance geometry. Standardized scaling answers whether each outcome contains comparable dynamic amplification once given equal ex ante weight."))
        story.append(make_table(extra_tables["raw_std"], max_rows=8, cols=raw_std_cols))
    story.append(PageBreak())
    story.append(P("Headline tau path: full-coordinate block-bootstrap uncertainty", "H1x"))
    story.append(img(charts["tau_ffbs"]))
    story.append(P("Figure 1. Full-coordinate soft response-score covariance amplification on the full 125-coordinate working grid. The path uses soft ridge whitening and no spectral cutoff; shaded bands are moving-block bootstrap pointwise bands and dashed lines are log-scale simultaneous bootstrap bands.", "Captionx"))
    story.append(make_table(uncertainty, max_rows=12))
    story.append(PageBreak())
    story.append(P("Headline tau path: bootstrap duplicate for compatibility", "H1x"))
    story.append(img(charts["tau_full_pipeline"]))
    story.append(P("Compatibility display of the same full-coordinate tau_soft path and block-bootstrap bands. This is not an FFBS state-uncertainty band.", "Captionx"))
    story.append(PageBreak())
    story.append(P("Surface-space shape allocation", "H1x"))
    story.append(P("Cell shape is computed directly on the full coordinate grid as log((K_hat[t,m,m]/C_hat[m,m]) / tau_soft[t]). Cells with reference variance below the documented tolerance are flagged in the saved table."))
    story.append(img(charts["shape_heatmap"], max_h=4.6 * inch))
    story.append(P("Figure 2A. Full-coordinate heatmap atlas of log cell amplification after removing tau_soft scale. Positive cells have more amplification than the full-grid scalar benchmark; negative cells have less.", "Captionx"))
    story.append(PageBreak())
    story.append(img(charts["shape_marginals"], max_h=5.8 * inch))
    story.append(P("Figure 2B. Full-coordinate block_shape paths for macro versus financial outcomes and horizon buckets 0-3, 4-12, and 13-24. The benchmark is 1. The concentration panel is a finite-working-grid display, not a basis-invariant Hilbert-space quantity.", "Captionx"))
    story.append(PageBreak())
    story.append(img(charts["shape_directions"], max_h=6.4 * inch))
    story.append(P("Appendix display. Dominant post-estimation rank-five shape directions at the full-coordinate selected dates; these directions do not feed into Figures 1-2.", "Captionx"))
    story.append(P("Rank and shock-definition uncertainty", "H1x"))
    story.append(img(charts["rank"]))
    story.append(P("Figure 6. Rank sensitivity for R=3, R=5, and R=7.", "Captionx"))
    story.append(img(charts["shock"]))
    story.append(P("Figure 7. Shock-construction envelope for tau_t.", "Captionx"))
    if "placebo" in extra_tables and not extra_tables["placebo"].empty:
        story.append(P("Placebo shock falsification", "H1x"))
        story.append(make_table(extra_tables["placebo"], max_rows=8, cols=placebo_cols))
    if "policy_cbi" in extra_tables and not extra_tables["policy_cbi"].empty:
        story.append(P("Policy versus information-shock split", "H1x"))
        story.append(make_table(extra_tables["policy_cbi"], max_rows=8, cols=policy_cbi_cols))
    story.append(PageBreak())
    story.append(P("Full-pipeline bootstrap", "H1x"))
    story.append(P("The Section 3.1 figure bands are moving-block bootstrap bands for the full-coordinate temporal-kernel backend. They are not FFBS state-uncertainty bands. Rank-reduced full-pipeline bootstrap tables are retained as comparison diagnostics outside Figures 1-2."))
    story.append(make_table(boot_summary, max_rows=12, cols=boot_cols))
    story.append(img(charts["weights"]))
    story.append(P("Figure 8. Student-t observation weights; low values mark downweighted extreme observations.", "Captionx"))
    story.append(P("Shock envelope table", "H1x"))
    story.append(make_table(shock_env, max_rows=12, cols=shock_env_cols))
    if "sf_fed" in extra_tables and not extra_tables["sf_fed"].empty:
        story.append(P("SF Fed/Bauer-Swanson appendix shock", "H1x"))
        story.append(make_table(extra_tables["sf_fed"], max_rows=6, cols=sf_fed_cols))
    if "smooth" in extra_tables and not extra_tables["smooth"].empty:
        story.append(P("Smooth LP robustness", "H1x"))
        story.append(make_table(extra_tables["smooth"], max_rows=6, cols=smooth_cols))
    if "episode" in extra_tables and not extra_tables["episode"].empty:
        story.append(P("Episode-level spike uncertainty", "H1x"))
        story.append(make_table(extra_tables["episode"], max_rows=10, cols=episode_cols))

    SimpleDocTemplate(str(FINAL_PDF), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch).build(story)

    fig_html = "".join(f"<h3>{html.escape(k)}</h3><img src='data:image/png;base64,{img_b64(v)}' />" for k, v in charts.items())
    sample_note = ""
    if not sample_coverage.empty:
        complete_range = sample_value(sample_coverage, "Common complete-coordinate score coverage")
        raw_window = sample_value(sample_coverage, "Raw outcome window consumed by score sample")
        sample_note = (
            "<h2>Sample Calendar</h2>"
            f"<p>The state index is attached to the base month t. In this run the common complete-coordinate score coverage is {html.escape(complete_range)}; "
            f"the LP construction consumes raw outcome data over {html.escape(raw_window)}.</p>"
            f"{df_html(sample_coverage, 30)}"
        )
    html_text = f"""<!doctype html><html><head><meta charset='utf-8'><title>Publication-grade OVK upgrade</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;line-height:1.42;color:#222;}} table{{border-collapse:collapse;font-size:12px;margin:12px 0 20px 0;}} th,td{{border:1px solid #bbb;padding:5px 7px;vertical-align:top;}} th{{background:#eee;}} img{{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 18px 0;}}</style></head><body>
<h1>Publication-grade full-coordinate monetary-policy response-score covariance</h1>
<p>{html.escape(estimator_intro)} The headline object is a time-varying response-score covariance.</p>
{sample_note}
<h2>Empirical anchor</h2><p>Dynamic covariance wins; dynamic mean does not.</p>
<h2>Rank summary</h2>{df_html(summary, 20)}
<h2>Same-sample outcomes</h2>{df_html(extra_tables.get('same_sample', pd.DataFrame()), 20)}
<h2>Raw versus standardized scaling</h2>{df_html(extra_tables.get('raw_std', pd.DataFrame()), 20)}
<h2>Placebo shock</h2>{df_html(extra_tables.get('placebo', pd.DataFrame()), 20)}
<h2>Policy/CBI split</h2>{df_html(extra_tables.get('policy_cbi', pd.DataFrame()), 20)}
<h2>SF Fed appendix shock</h2>{df_html(extra_tables.get('sf_fed', pd.DataFrame()), 20)}
<h2>Smooth LP robustness</h2>{df_html(extra_tables.get('smooth', pd.DataFrame()), 20)}
<h2>Episode spike uncertainty</h2>{df_html(extra_tables.get('episode', pd.DataFrame()), 20)}
<h2>Uncertainty summary</h2>{df_html(uncertainty, 20)}
<h2>Full-pipeline bootstrap</h2>{df_html(boot_summary, 20)}
<h2>Shock envelope</h2>{df_html(shock_env, 20)}
<h2>Charts</h2>{fig_html}
</body></html>"""
    FINAL_HTML.write_text(html_text, encoding="utf-8")


def _legacy_main_deprecated() -> None:
    global PUBLICATION_WORKERS
    t0 = time.perf_counter()
    cache_hits = {"score_data": 0, "rank_fits": 0, "bootstrap_sufficient_stats": 0, "bootstrap_draws": 0}
    cache_misses = {"score_data": 0, "rank_fits": 0, "bootstrap_sufficient_stats": 0, "bootstrap_draws": 0}
    for old in [TABLES, CHARTS]:
        if old.exists():
            shutil.rmtree(old)
        old.mkdir(parents=True, exist_ok=True)

    worker_benchmark: dict[str, Any] | None = None
    if BENCHMARK_WORKERS and not _workers_env:
        worker_benchmark = benchmark_worker_count(_max_auto_workers)
        PUBLICATION_WORKERS = int(worker_benchmark["selected_workers"])

    data_hash = file_sha256(SRC_ZIP)
    panels = load_panels_from_zip(SRC_ZIP)
    panel = panels["panel"]
    outcome_signature = outcome_signature_for_panel(panel)
    variants = [
        ("median_fallback", "MP_median with fallback", "MP_median_fallback", "CBI_median_fallback"),
        ("mp_pm_only", "MP_pm only", "MP_pm", "CBI_pm"),
        ("event_manual", "Event-level manual aggregation", "MP_event_manual", "CBI_event_manual"),
    ]
    score_key = cache_key("score_data_npz", data_hash, H, L, outcome_signature, [(v[0], v[2], v[3]) for v in variants])
    score_arrays = cache_get_npz(score_key)
    if score_arrays is not None:
        cache_hits["score_data"] += 1
        score_data = score_data_from_npz_arrays(score_arrays)
    else:
        cache_misses["score_data"] += 1
        score_data = {}
        for key, label, shock_col, cbi_col in variants:
            score_data[key] = build_lp_scores(panel, shock_col, cbi_col, H=H, L=L)
        cache_set_npz(score_key, score_data_to_npz_arrays(score_data))

    results: dict[tuple[str, int], RankResult] = {}
    rank_tasks = []
    rank_context = {
        key: {"Q_scores": value["Q_scores"], "dates": value["dates"]}
        for key, value in score_data.items()
    }
    _RANK_CONTEXT["score_data"] = rank_context
    for key, label, _, _ in variants:
        ranks_for_variant = RANKS if key == "median_fallback" else [HEADLINE_R]
        for rank in ranks_for_variant:
            rank_tasks.append(
                {
                    "data_hash": data_hash,
                    "variant": key,
                    "label": label,
                    "rank": rank,
                    "em_iters": EM_ITERS,
                    "estimator_mode": ESTIMATOR_MODE,
                }
            )
    print(f"Estimating rank/shock models: {len(rank_tasks)} fits with workers={PUBLICATION_WORKERS}", flush=True)
    for (key, rank), res, from_cache in run_parallel_tasks(
        estimate_rank_task,
        rank_tasks,
        PUBLICATION_WORKERS,
        initializer=init_rank_worker,
        initargs=(rank_context, OUTCOME_LABELS),
    ):
        results[(key, rank)] = res
        cache_hits["rank_fits"] += int(from_cache)
        cache_misses["rank_fits"] += int(not from_cache)
    rank_rows = [rank_summary_row(results[(task["variant"], task["rank"])]) for task in rank_tasks]
    rank_summary = pd.DataFrame(rank_rows)
    rank_summary.to_csv(TABLES / "publication_grade_rank_summary.csv", index=False)

    headline = results[("median_fallback", HEADLINE_R)]
    state_draw_paths = ffbs_state_draws(headline.fit, B_STATE, STATE_SEED)
    tau_draws, scale_draws, shape_draws, diag_draws = state_draw_scale_shape(
        state_draw_paths,
        HEADLINE_R,
        estimator_mode=headline.estimator_mode,
    )
    tau_band = positive_simultaneous_band(tau_draws, headline.tau, level=0.90)
    scale_band = simultaneous_band(scale_draws, headline.scale_log_tau, level=0.90)
    shape_band = simultaneous_band(shape_draws, headline.shape_distance, level=0.90)

    dates = pd.to_datetime(headline.dates)
    march_mask = dates.dt.strftime("%Y-%m") == "2020-03"
    march_idx = int(np.where(march_mask.to_numpy())[0][0]) if march_mask.any() else None
    sample_coverage = build_publication_sample_coverage(panel, score_data["base5_headline"])
    sample_coverage.to_csv(TABLES / "publication_grade_sample_coverage.csv", index=False)
    tau_ranks = np.argsort(np.argsort(-tau_draws, axis=1), axis=1) + 1
    top10_probs = (tau_ranks <= 10).mean(axis=0)
    march_rank1_prob = float((tau_ranks[:, march_idx] == 1).mean()) if march_idx is not None else np.nan
    march_top10_prob = float((tau_ranks[:, march_idx] <= 10).mean()) if march_idx is not None else np.nan
    pvars = len(OUTCOME_LABELS)

    # Surface-space shape diagnostics: remove tau_t scale, then map shape back through V.
    surface_shape = surface_shape_from_A(headline.A, headline.V, H, pvars, OUTCOME_LABELS)
    shape_metric_names = [
        "surface_shape_rms_log_relative",
        "financial_variable_share",
        "macro_variable_share",
        "short_horizon_share",
        "medium_horizon_share",
        "long_horizon_share",
        "cell_effective_support",
        "variable_effective_support",
        "horizon_effective_support",
    ]
    shape_metric_draws = state_draw_surface_shape_metric_draws(
        state_draw_paths,
        headline.V,
        H,
        pvars,
        OUTCOME_LABELS,
        shape_metric_names,
        estimator_mode=headline.estimator_mode,
    )
    shape_metric_band = np.nanquantile(shape_metric_draws, [0.05, 0.50, 0.95], axis=0)
    shape_metrics_data: dict[str, Any] = {"date": dates.dt.strftime("%Y-%m-%d")}
    for name, values in surface_shape["metrics"].items():
        shape_metrics_data[name] = values
        if name in shape_metric_names:
            k = shape_metric_names.index(name)
            shape_metrics_data[f"{name}_p05"] = shape_metric_band[0, :, k]
            shape_metrics_data[f"{name}_median"] = shape_metric_band[1, :, k]
            shape_metrics_data[f"{name}_p95"] = shape_metric_band[2, :, k]
    shape_metrics_df = pd.DataFrame(shape_metrics_data)
    shape_metrics_df.to_csv(TABLES / "publication_grade_surface_shape_metrics.csv", index=False)

    horizons = np.repeat(np.arange(H + 1), pvars)
    variables = np.tile(np.array(OUTCOME_LABELS), H + 1)
    date_str = dates.dt.strftime("%Y-%m-%d").to_numpy()
    shape_alloc_df = pd.DataFrame(
        {
            "date": np.repeat(date_str, M_DIM),
            "horizon_months": np.tile(horizons, len(dates)),
            "variable": np.tile(variables, len(dates)),
            "shape_variance": surface_shape["surface_diag"].ravel(),
            "baseline_leverage": np.tile(surface_shape["baseline_leverage"], len(dates)),
            "relative_shape_variance": surface_shape["relative_shape_variance"].ravel(),
            "log_relative_shape_variance": surface_shape["log_relative_shape_variance"].ravel(),
            "shape_cell_share": (surface_shape["surface_diag"] / HEADLINE_R).ravel(),
            "baseline_cell_share": np.tile(surface_shape["baseline_leverage"] / HEADLINE_R, len(dates)),
        }
    )
    shape_alloc_df.to_csv(TABLES / "publication_grade_surface_shape_allocations.csv", index=False)

    episodes = selected_shape_episodes(
        dates,
        headline.tau,
        headline.scale_log_tau,
        surface_shape["surface_shape_rms_log_relative"],
        march_idx,
    )
    episode_df = pd.DataFrame(episodes).drop(columns=["idx"])
    episode_df.to_csv(TABLES / "publication_grade_surface_shape_episodes.csv", index=False)

    shape_trace = np.trace(surface_shape["shape"], axis1=1, axis2=2) / HEADLINE_R
    variable_share_cols = [f"variable_share_{safe_name(label)}" for label in OUTCOME_LABELS]
    horizon_share_cols = [f"horizon_{hh:02d}_share" for hh in range(H + 1)]
    neutral_surface = surface_shape_from_A(np.eye(HEADLINE_R)[None, :, :], headline.V, H, pvars, OUTCOME_LABELS)
    invariant_checks = pd.DataFrame(
        [
            {"check": "trace_S_over_R_equals_one", "max_abs_error": float(np.max(np.abs(shape_trace - 1.0)))},
            {"check": "sum_diag_C_equals_R", "max_abs_error": float(np.max(np.abs(surface_shape["surface_diag"].sum(axis=1) - HEADLINE_R)))},
            {"check": "variable_shares_sum_to_one", "max_abs_error": float(np.max(np.abs(shape_metrics_df[variable_share_cols].sum(axis=1) - 1.0)))},
            {"check": "horizon_shares_sum_to_one", "max_abs_error": float(np.max(np.abs(shape_metrics_df[horizon_share_cols].sum(axis=1) - 1.0)))},
            {"check": "neutral_shape_log_relative_zero", "max_abs_error": float(np.max(np.abs(neutral_surface["log_relative_shape_variance"])))},
        ]
    )
    invariant_checks.to_csv(TABLES / "publication_grade_surface_shape_invariant_checks.csv", index=False)
    if invariant_checks["max_abs_error"].max() > 1e-8:
        raise RuntimeError("Surface-shape invariant check failed")

    path_df = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "tau": headline.tau,
            "tau_point_p05": tau_band["point_low"],
            "tau_point_median": tau_band["point_med"],
            "tau_point_p95": tau_band["point_high"],
            "tau_simul_p05": tau_band["sim_low"],
            "tau_simul_p95": tau_band["sim_high"],
            "scale_log_tau": headline.scale_log_tau,
            "scale_point_p05": scale_band["point_low"],
            "scale_point_p95": scale_band["point_high"],
            "shape_distance": headline.shape_distance,
            "shape_point_p05": shape_band["point_low"],
            "shape_point_p95": shape_band["point_high"],
            "robust_observation_weight": headline.fit.weights,
            "reference_objective_weight": headline.fit.objective_weights
            if headline.fit.objective_weights is not None
            else np.ones(len(dates)),
            "tau_total_second_moment": np.trace(headline.total_second_moment_whitened, axis1=1, axis2=2) / HEADLINE_R
            if headline.total_second_moment_whitened is not None
            else headline.tau,
            "tau_mean_component": np.trace(headline.mean_component_whitened, axis1=1, axis2=2) / HEADLINE_R
            if headline.mean_component_whitened is not None
            else np.zeros(len(dates)),
            "tau_within_covariance": np.trace(headline.within_covariance_whitened, axis1=1, axis2=2) / HEADLINE_R
            if headline.within_covariance_whitened is not None
            else headline.tau,
            "prob_top10_by_state_draw": top10_probs,
        }
    )
    for name in shape_metric_names:
        path_df[name] = shape_metrics_df[name]
    for r in range(HEADLINE_R):
        path_df[f"A{r+1}{r+1}"] = headline.A[:, r, r]
    path_df.to_csv(TABLES / "publication_grade_headline_state_path.csv", index=False)
    write_rank_component_npz(headline, TABLES / "publication_grade_headline_second_moment_components.npz")

    uncertainty = pd.DataFrame(
        [
            {"quantity": "state_draws", "value": B_STATE},
            {"quantity": "tau_log_simultaneous_90_sup_crit", "value": tau_band["sup_crit"]},
            {"quantity": "tau_simultaneous_band_scale", "value": "log_tau_exponentiated"},
            {"quantity": "headline_tau_max", "value": float(headline.tau.max())},
            {"quantity": "headline_tau_max_month", "value": dates.iloc[int(np.argmax(headline.tau))].strftime("%Y-%m")},
            {"quantity": "headline_surface_shape_max", "value": float(surface_shape["surface_shape_rms_log_relative"].max())},
            {"quantity": "headline_surface_shape_max_month", "value": dates.iloc[int(np.argmax(surface_shape["surface_shape_rms_log_relative"]))].strftime("%Y-%m")},
            {"quantity": "march_2020_rank1_probability_state_draws", "value": march_rank1_prob},
            {"quantity": "march_2020_top10_probability_state_draws", "value": march_top10_prob},
            {"quantity": "minimum_student_t_weight", "value": float(headline.fit.weights.min())},
            {"quantity": "student_t_degrees_of_freedom", "value": ROBUST_NU},
            {"quantity": "student_t_minimum_weight_floor", "value": MIN_STUDENT_WEIGHT},
        ]
    )
    uncertainty.to_csv(TABLES / "publication_grade_uncertainty_summary.csv", index=False)

    # Shock construction envelope.
    shock_env = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")})
    for key, label, _, _ in variants:
        res = results[(key, HEADLINE_R)]
        shock_env[f"tau_{key}"] = res.tau
    tau_cols = [c for c in shock_env.columns if c.startswith("tau_")]
    shock_env["tau_spec_min"] = shock_env[tau_cols].min(axis=1)
    shock_env["tau_spec_max"] = shock_env[tau_cols].max(axis=1)
    shock_env["tau_spec_range"] = shock_env["tau_spec_max"] - shock_env["tau_spec_min"]
    shock_env.to_csv(TABLES / "publication_grade_shock_construction_envelope.csv", index=False)

    # Rank subspace comparisons versus headline R=5 baseline.
    subspace_rows = []
    base_V = headline.V
    for rank in RANKS:
        res = results[("median_fallback", rank)]
        angles = principal_angles(base_V, res.V)
        subspace_rows.append(
            {
                "variant": "median_fallback",
                "comparison": f"R{rank} vs headline R{HEADLINE_R}",
                "rank": rank,
                "max_angle_degrees": float(np.max(angles)),
                "mean_angle_degrees": float(np.mean(angles)),
                **{f"angle_{j+1}_degrees": float(angles[j]) for j in range(len(angles))},
            }
        )
    subspace_df = pd.DataFrame(subspace_rows)
    subspace_df.to_csv(TABLES / "publication_grade_rank_subspace_angles.csv", index=False)

    # Full-pipeline bootstrap for headline rank and baseline shock definition.
    print(f"Running full-pipeline bootstrap: B={B_BOOT}", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    Q = headline.Q_scores
    base_top10 = set(dates.iloc[np.argsort(headline.tau)[::-1][:10]].dt.strftime("%Y-%m"))
    boot_stats, stats_from_cache = load_or_build_bootstrap_stats(data_hash, "median_fallback", score_data["median_fallback"])
    cache_hits["bootstrap_sufficient_stats"] += int(stats_from_cache)
    cache_misses["bootstrap_sufficient_stats"] += int(not stats_from_cache)
    boot_indices = np.asarray(
        [circular_block_indices(len(Q), BOOT_BLOCK_LEN, rng) for _ in range(B_BOOT)],
        dtype=np.int64,
    )
    worker_namespace = cache_key(
        WORKER_ARRAY_VERSION,
        data_hash,
        H,
        L,
        tuple(OUTCOME_LABELS),
        M_DIM,
        HEADLINE_R,
        B_BOOT,
        BOOT_BLOCK_LEN,
        BOOT_SEED,
        "ols_sufficient_stats",
    )
    array_paths = {
        "X_design": str(write_worker_array(worker_namespace, "X_design", boot_stats["X_design"])),
        "Y_all": str(write_worker_array(worker_namespace, "Y_all", boot_stats["Y_all"])),
        "row_xx": str(write_worker_array(worker_namespace, "row_xx", boot_stats["row_xx"])),
        "row_xy": str(write_worker_array(worker_namespace, "row_xy", boot_stats["row_xy"])),
        "boot_indices": str(write_worker_array(worker_namespace, "boot_indices", boot_indices)),
    }
    boot_context = {
        "array_paths": array_paths,
        "dates_ns": dates.to_numpy(dtype="datetime64[ns]"),
        "headline_V": headline.V,
        "base_top10": sorted(base_top10),
        "march_idx": march_idx,
        "bootstrap_method": "ols_sufficient_stats",
        "outcome_labels": list(OUTCOME_LABELS),
        "m_dim": M_DIM,
    }
    tau_boot = np.empty((B_BOOT, len(dates)))
    beta_boot = np.empty((B_BOOT, M_DIM))
    share_boot = np.empty((B_BOOT, 10))
    mode_boot = np.empty((B_BOOT, HEADLINE_R, M_DIM))
    subspace_angle_boot = np.empty(B_BOOT)
    boot_tasks = [
        {
            "data_hash": data_hash,
            "draw": b,
            "em_iters": BOOT_EM_ITERS,
        }
        for b in range(B_BOOT)
    ]
    boot_rows = []
    try:
        boot_results = run_parallel_tasks(
            bootstrap_draw_task,
            boot_tasks,
            PUBLICATION_WORKERS,
            initializer=init_bootstrap_worker,
            initargs=(boot_context,),
        )
        for draw, payload, row, from_cache in boot_results:
            tau_boot[draw] = payload["tau"]
            beta_boot[draw] = payload["beta"]
            share_boot[draw] = payload["shares"]
            mode_boot[draw] = payload["modes"]
            subspace_angle_boot[draw] = float(np.ravel(payload["subspace_angle"])[0])
            boot_rows.append(row)
            cache_hits["bootstrap_draws"] += int(from_cache)
            cache_misses["bootstrap_draws"] += int(not from_cache)
    finally:
        close_bootstrap_worker_context()
    boot_rows = sorted(boot_rows, key=lambda r: r.get("draw", 0))
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(TABLES / "publication_grade_full_pipeline_bootstrap_draws.csv", index=False)
    valid_boot = np.isfinite(tau_boot).all(axis=1)
    tau_boot_valid = tau_boot[valid_boot]
    boot_band = positive_simultaneous_band(tau_boot_valid, headline.tau, level=0.90) if len(tau_boot_valid) else tau_band
    path_df["tau_full_pipeline_p05"] = boot_band["point_low"]
    path_df["tau_full_pipeline_p95"] = boot_band["point_high"]
    path_df["tau_full_pipeline_simul_p05"] = boot_band["sim_low"]
    path_df["tau_full_pipeline_simul_p95"] = boot_band["sim_high"]
    path_df.to_csv(TABLES / "publication_grade_headline_state_path.csv", index=False)
    write_rank_component_npz(headline, TABLES / "publication_grade_headline_second_moment_components.npz")

    boot_clean = boot_df[boot_df.get("error").isna()] if "error" in boot_df.columns else boot_df
    boot_summary = pd.DataFrame(
        [
            {"quantity": "bootstrap_draws_requested", "value": B_BOOT},
            {"quantity": "bootstrap_draws_valid", "value": int(len(boot_clean))},
            {"quantity": "alpha_p05", "value": float(boot_clean["alpha_hat"].quantile(0.05))},
            {"quantity": "alpha_p50", "value": float(boot_clean["alpha_hat"].quantile(0.50))},
            {"quantity": "alpha_p95", "value": float(boot_clean["alpha_hat"].quantile(0.95))},
            {"quantity": "trace_share_p05", "value": float(boot_clean["retained_trace_share"].quantile(0.05))},
            {"quantity": "trace_share_p95", "value": float(boot_clean["retained_trace_share"].quantile(0.95))},
            {"quantity": "max_tau_p05", "value": float(boot_clean["max_tau"].quantile(0.05))},
            {"quantity": "max_tau_p95", "value": float(boot_clean["max_tau"].quantile(0.95))},
            {"quantity": "march_2020_rank1_probability_full_pipeline", "value": float((boot_clean["march_2020_rank"] == 1).mean()) if march_idx is not None else np.nan},
            {"quantity": "march_2020_top10_probability_full_pipeline", "value": float((boot_clean["march_2020_rank"] <= 10).mean()) if march_idx is not None else np.nan},
            {"quantity": "top10_overlap_with_baseline_median", "value": float(boot_clean["top10_overlap_with_baseline"].median())},
            {"quantity": "tau_full_pipeline_log_simultaneous_90_sup_crit", "value": boot_band["sup_crit"]},
            {"quantity": "tau_full_pipeline_simultaneous_band_scale", "value": "log_tau_exponentiated"},
        ]
    )
    boot_summary.to_csv(TABLES / "publication_grade_full_pipeline_bootstrap_summary.csv", index=False)

    # Charts.
    fig = plt.figure(figsize=(10.5, 5.5))
    plt.plot(dates, headline.tau, label="tau_t estimate", color="black")
    plt.fill_between(dates, path_df["tau_point_p05"], path_df["tau_point_p95"], alpha=0.24, label="90% pointwise FFBS state band")
    plt.plot(dates, path_df["tau_simul_p05"], linestyle="--", linewidth=1.1, label="90% log-simultaneous FFBS lower")
    plt.plot(dates, path_df["tau_simul_p95"], linestyle="--", linewidth=1.1, label="90% log-simultaneous FFBS upper")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Rank-five total OVK amplification: FFBS state uncertainty")
    plt.ylabel("tau_t = trace(A_t)/5")
    plt.legend()
    tau_ffbs_chart = savefig(fig, "01_publication_tau_ffbs_state_bands")

    fig = plt.figure(figsize=(10.5, 5.5))
    plt.plot(dates, headline.tau, label="tau_t estimate", color="black")
    plt.fill_between(dates, path_df["tau_full_pipeline_p05"], path_df["tau_full_pipeline_p95"], alpha=0.24, label="90% pointwise full-pipeline band")
    plt.plot(dates, path_df["tau_full_pipeline_simul_p05"], linestyle="--", linewidth=1.1, label="90% log-simultaneous full-pipeline lower")
    plt.plot(dates, path_df["tau_full_pipeline_simul_p95"], linestyle="--", linewidth=1.1, label="90% log-simultaneous full-pipeline upper")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Rank-five total OVK amplification: full-pipeline uncertainty")
    plt.ylabel("tau_t = trace(A_t)/5")
    plt.legend()
    tau_full_pipeline_chart = savefig(fig, "02_publication_tau_full_pipeline_bands")

    shape_heatmap_chart = plot_shape_heatmap_atlas(
        surface_shape["log_relative_shape_variance"],
        dates,
        episodes,
        H,
        OUTCOME_LABELS,
    )
    shape_marginals_chart = plot_shape_marginals(dates, shape_metrics_df)
    shape_directions_chart, shape_direction_df = plot_shape_directions(
        surface_shape["shape"],
        headline.V,
        episodes,
        H,
        OUTCOME_LABELS,
    )
    shape_direction_df.to_csv(TABLES / "publication_grade_surface_shape_directions.csv", index=False)

    fig = plt.figure(figsize=(8.8, 5.2))
    base_rank = rank_summary[rank_summary["variant"] == "median_fallback"].copy()
    plt.bar(base_rank["rank"].astype(str), base_rank["retained_trace_share"])
    plt.ylabel("Retained trace share")
    plt.xlabel("Rank")
    plt.title("Rank sensitivity of retained OVK geometry")
    rank_chart = savefig(fig, "06_rank_sensitivity")

    fig = plt.figure(figsize=(10.5, 5.4))
    for c in tau_cols:
        plt.plot(pd.to_datetime(shock_env["date"]), shock_env[c], label=c.replace("tau_", ""))
    plt.fill_between(pd.to_datetime(shock_env["date"]), shock_env["tau_spec_min"], shock_env["tau_spec_max"], alpha=0.14, label="specification envelope")
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t")
    plt.title("Shock-construction uncertainty envelope")
    plt.legend()
    shock_chart = savefig(fig, "07_shock_construction_envelope")

    fig = plt.figure(figsize=(10.5, 4.8))
    plt.plot(dates, headline.fit.weights)
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Student-t robust observation weights")
    plt.ylabel("weight")
    weights_chart = savefig(fig, "08_student_t_weights")

    charts = {
        "tau_ffbs": tau_ffbs_chart,
        "tau_full_pipeline": tau_full_pipeline_chart,
        "shape_heatmap": shape_heatmap_chart,
        "shape_marginals": shape_marginals_chart,
        "shape_directions": shape_directions_chart,
        "rank": rank_chart,
        "shock": shock_chart,
        "weights": weights_chart,
    }
    build_report(rank_summary, path_df, uncertainty, shock_env, boot_summary, charts)
    write_top5_compatible_outputs(
        panels,
        score_data,
        headline,
        rank_summary,
        path_df,
        tau_draws,
        diag_draws,
        beta_boot,
        share_boot,
        mode_boot,
        subspace_angle_boot,
    )
    write_robustness_compatible_outputs(panels, results)

    # Metadata and bundle.
    metadata = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "headline_rank": HEADLINE_R,
        "outcome_labels": OUTCOME_LABELS,
        "outcome_surface_dimension": M_DIM,
        "ranks": RANKS,
        "state_draws": B_STATE,
        "full_pipeline_bootstrap_draws": B_BOOT,
        "bootstrap_block_length": BOOT_BLOCK_LEN,
        "student_t_degrees_of_freedom": ROBUST_NU,
        "student_t_minimum_weight_floor": MIN_STUDENT_WEIGHT,
        "covariance_estimator_mode": headline.estimator_mode,
        "default_covariance_estimator_mode": ESTIMATOR_MODE,
        "arithmetic_reference_ridge_scale": ARITHMETIC_REFERENCE_RIDGE_SCALE,
        "alpha_grid": ALPHA_GRID.tolist(),
        "workers": PUBLICATION_WORKERS,
        "worker_benchmark": worker_benchmark,
        "bootstrap_method": "moving_block_ols_sufficient_stats",
        "internal_array_cache_format": "npz/npy",
        "cache_dir": str(CACHE_DIR),
        "cache_disabled": DISABLE_CACHE,
        "legacy_top5_compatible_outputs_enabled": WRITE_LEGACY_TOP5_COMPAT,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "estimation_upgrades": [
            "arithmetic outer-product total second-moment estimator is the default; log_spd_legacy is deprecated comparison mode",
            "reference weights are normalized, used in C_hat, and used in the state smoother objective",
            "alpha selected only by deprecated log_spd_legacy mode",
            "F, Q, and R estimated by EM-style state-space iterations",
            "structured block/diagonal transition shrinkage with spectral-radius cap",
            "Student-t robust observation weighting",
            "scale tau_t and surface-space shape allocation reported separately",
            "shape heatmap atlas, marginal allocation paths, and dominant shape-direction surfaces",
            "rank R=3, R=5, R=7 sensitivity and subspace angles",
            "FFBS simulation-smoother path draws",
            "full-pipeline moving-block bootstrap with parameter and basis uncertainty",
            "pointwise tau_t bands and log-scale simultaneous tau_t bands exponentiated to preserve positivity",
            "shock-construction uncertainty envelope",
        ],
    }
    (OUT / "publication_grade_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), CODE / "run_publication_grade_ovk.py")
    helper = Path(__file__).with_name("ovk_data.py")
    if helper.exists():
        shutil.copy2(helper, CODE / "ovk_data.py")
    tmp_worker_arrays = OUT / "_worker_arrays"
    if tmp_worker_arrays.exists():
        rmtree_with_memmap_retry(tmp_worker_arrays)

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, arcname=f.relative_to(OUT))
        z.write(FINAL_PDF, arcname=FINAL_PDF.name)
        z.write(FINAL_HTML, arcname=FINAL_HTML.name)

    print("DONE")
    print("PDF", FINAL_PDF, FINAL_PDF.stat().st_size)
    print("HTML", FINAL_HTML, FINAL_HTML.stat().st_size)
    print("ZIP", FINAL_ZIP, FINAL_ZIP.stat().st_size)
    print(rank_summary[rank_summary["variant"].eq("median_fallback")].to_string(index=False))


def main() -> None:
    global PUBLICATION_WORKERS
    t0 = time.perf_counter()
    cache_hits = {"score_data": 0, "rank_fits": 0, "bootstrap_sufficient_stats": 0, "bootstrap_draws": 0}
    cache_misses = {"score_data": 0, "rank_fits": 0, "bootstrap_sufficient_stats": 0, "bootstrap_draws": 0}
    for old in [TABLES, CHARTS]:
        if old.exists():
            shutil.rmtree(old)
        old.mkdir(parents=True, exist_ok=True)

    worker_benchmark: dict[str, Any] | None = None
    if BENCHMARK_WORKERS and not _workers_env:
        worker_benchmark = benchmark_worker_count(_max_auto_workers)
        PUBLICATION_WORKERS = int(worker_benchmark["selected_workers"])

    data_hash = file_sha256(SRC_ZIP)
    panels = load_panels_from_zip(SRC_ZIP)
    panel = add_placebo_shocks(panels["panel"], seed=PLACEBO_SEED, shift_months=PLACEBO_SHIFT_MONTHS)
    panel, sf_meta = add_sf_fed_shocks(panel, SF_FED_SURPRISES)
    panels["panel"] = panel

    all8_cols = tuple(c for c in ALL_OUTCOME_COLUMNS if c in panel.columns)
    overlap_probe = build_lp_scores(panel, "MP_median_fallback", "CBI_median_fallback", H=H, L=L, outcome_columns=all8_cols)
    overlap_dates = pd.to_datetime(overlap_probe["dates"])
    specs_all = variant_specs(panel, sf_meta, overlap_dates)
    active_specs = [s for s in specs_all if s.available]
    spec_map = {s.key: s for s in specs_all}
    sample_dates_by_key = {"expectation_overlap": overlap_dates}

    score_key = cache_key(
        "score_data_npz_v2",
        data_hash,
        H,
        L,
        HEADLINE_OUTCOMES,
        [(s.key, s.outcome_columns, s.shock_col, s.cbi_col or "", s.sample_dates_key or "", s.transform_signature, s.available) for s in specs_all],
        sf_meta.get("sf_fed_path", ""),
        sf_meta.get("sf_fed_rows", 0),
    )
    score_arrays = None if DISABLE_CACHE else cache_get_npz(score_key)
    if score_arrays is not None:
        cache_hits["score_data"] += 1
        score_data = score_data_from_npz_arrays(score_arrays)
    else:
        cache_misses["score_data"] += 1
        score_data: dict[str, dict[str, Any]] = {}
        for spec in active_specs:
            sample_dates = sample_dates_by_key.get(spec.sample_dates_key or "")
            raw_scores = build_lp_scores(
                panel,
                spec.shock_col,
                spec.cbi_col,
                H=H,
                L=L,
                outcome_columns=spec.outcome_columns,
                sample_dates=sample_dates,
            )
            labels = list(raw_scores["outcome_labels"])
            q_raw = np.asarray(raw_scores["Q_scores"], float).copy()
            transform: dict[str, Any] = {"kind": "raw"}
            if spec.transform == "standardized":
                weights, traces, target = standardization_weights(q_raw, labels)
                transform = {
                    "kind": "standardized",
                    "weights": weights.tolist(),
                    "weights_by_outcome": weights[: len(labels)].tolist(),
                    "raw_block_traces": traces.tolist(),
                    "target_trace": target,
                }
            elif spec.transform == "smooth":
                transform = {"kind": "smooth", "smoothing": SMOOTH_SPLINE_S}
            raw_scores["Q_scores_raw"] = q_raw
            raw_scores["Q_scores"] = apply_score_transform(q_raw, labels, transform)
            raw_scores["score_transform"] = transform
            raw_scores["variant"] = spec.key
            score_data[spec.key] = raw_scores
        if not DISABLE_CACHE:
            cache_set_npz(score_key, score_data_to_npz_arrays(score_data))

    results: dict[tuple[str, int], RankResult] = {}
    rank_context = {
        key: {"Q_scores": value["Q_scores"], "dates": value["dates"], "outcome_labels": value["outcome_labels"]}
        for key, value in score_data.items()
    }
    _RANK_CONTEXT["score_data"] = rank_context
    rank_tasks: list[dict[str, Any]] = []
    for spec in active_specs:
        if spec.key not in score_data:
            continue
        labels = list(score_data[spec.key]["outcome_labels"])
        for rank in spec.run_ranks:
            rank_tasks.append(
                {
                    "data_hash": data_hash,
                    "variant": spec.key,
                    "label": spec.label,
                    "rank": rank,
                    "em_iters": EM_ITERS,
                    "outcome_labels": labels,
                    "m_dim": int(score_data[spec.key]["Q_scores"].shape[1]),
                    "transform_signature": spec.transform_signature,
                    "estimator_mode": ESTIMATOR_MODE,
                }
            )
    print(f"Estimating rank/variant models: {len(rank_tasks)} fits with workers={PUBLICATION_WORKERS}", flush=True)
    for (key, rank), res, from_cache in run_parallel_tasks(
        estimate_rank_task,
        rank_tasks,
        PUBLICATION_WORKERS,
        initializer=init_rank_worker,
        initargs=(rank_context, None),
    ):
        results[(key, rank)] = res
        cache_hits["rank_fits"] += int(from_cache)
        cache_misses["rank_fits"] += int(not from_cache)

    rank_rows = []
    for task in rank_tasks:
        row = rank_summary_row(results[(task["variant"], task["rank"])])
        spec = spec_map[task["variant"]]
        row["group"] = spec.group
        row["outcome_count"] = len(results[(task["variant"], task["rank"])].outcome_labels)
        row["transform"] = spec.transform
        rank_rows.append(row)
    rank_summary = pd.DataFrame(rank_rows)
    rank_summary.to_csv(TABLES / "publication_grade_rank_summary.csv", index=False)

    headline = results[("base5_headline", HEADLINE_R)]
    set_outcome_labels(headline.outcome_labels)
    labels = list(headline.outcome_labels)
    pvars = len(labels)
    m_dim = (H + 1) * pvars
    dates = pd.to_datetime(headline.dates)
    march_mask = dates.dt.strftime("%Y-%m") == "2020-03"
    march_idx = int(np.where(march_mask.to_numpy())[0][0]) if march_mask.any() else None
    sample_coverage = build_publication_sample_coverage(panel, score_data["base5_headline"])
    sample_coverage.to_csv(TABLES / "publication_grade_sample_coverage.csv", index=False)

    print(f"Running full-pipeline bootstraps for variants: B={B_BOOT}", flush=True)
    bootstrap_outputs: dict[str, dict[str, Any]] = {}
    base_top10 = set(dates.iloc[np.argsort(headline.tau)[::-1][:10]].dt.strftime("%Y-%m"))
    for spec in active_specs:
        if (spec.key, HEADLINE_R) not in results or not should_bootstrap_variant(spec):
            continue
        print(f"  bootstrap {spec.key}", flush=True)
        bootstrap_outputs[spec.key] = run_full_pipeline_bootstrap(
            data_hash,
            spec,
            score_data[spec.key],
            results[(spec.key, HEADLINE_R)],
            PUBLICATION_WORKERS,
            cache_hits,
            cache_misses,
            base_top10=base_top10,
            march_idx=march_idx,
        )
        suffix = "" if spec.key == "base5_headline" else f"_{spec.key}"
        bootstrap_outputs[spec.key]["boot_df"].to_csv(TABLES / f"publication_grade_full_pipeline_bootstrap_draws{suffix}.csv", index=False)

    if "base5_headline" not in bootstrap_outputs:
        fallback_band = positive_simultaneous_band(np.asarray([headline.tau]), headline.tau, level=0.90)
        bootstrap_outputs["base5_headline"] = {
            "boot_df": pd.DataFrame(),
            "tau_boot": np.asarray([headline.tau]),
            "tau_boot_valid": np.asarray([headline.tau]),
            "beta_boot": np.asarray([headline.beta]),
            "share_boot": np.asarray([headline.shares[:10]]),
            "mode_boot": np.asarray([headline.V.T[:HEADLINE_R]]),
            "subspace_angle_boot": np.asarray([0.0]),
            "band": fallback_band,
        }
    headline_boot = bootstrap_outputs["base5_headline"]
    boot_band = headline_boot["band"]
    beta_boot = headline_boot["beta_boot"]
    share_boot = headline_boot["share_boot"]
    mode_boot = headline_boot["mode_boot"]
    subspace_angle_boot = headline_boot["subspace_angle_boot"]

    state_draw_paths = ffbs_state_draws(headline.fit, B_STATE, STATE_SEED)
    tau_draws, scale_draws, shape_draws, diag_draws = state_draw_scale_shape(
        state_draw_paths,
        HEADLINE_R,
        estimator_mode=headline.estimator_mode,
    )
    tau_band = positive_simultaneous_band(tau_draws, headline.tau, level=0.90)
    scale_band = simultaneous_band(scale_draws, headline.scale_log_tau, level=0.90)
    shape_band = simultaneous_band(shape_draws, headline.shape_distance, level=0.90)
    tau_ranks = np.argsort(np.argsort(-tau_draws, axis=1), axis=1) + 1
    top10_probs = (tau_ranks <= 10).mean(axis=0)
    march_rank1_prob = float((tau_ranks[:, march_idx] == 1).mean()) if march_idx is not None else np.nan
    march_top10_prob = float((tau_ranks[:, march_idx] <= 10).mean()) if march_idx is not None else np.nan

    surface_shape = surface_shape_from_A(headline.A, headline.V, H, pvars, labels)
    shape_metric_names = [
        "surface_shape_rms_log_relative",
        "financial_variable_share",
        "macro_variable_share",
        "short_horizon_share",
        "medium_horizon_share",
        "long_horizon_share",
        "cell_effective_support",
        "variable_effective_support",
        "horizon_effective_support",
    ]
    shape_metric_draws = state_draw_surface_shape_metric_draws(
        state_draw_paths,
        headline.V,
        H,
        pvars,
        labels,
        shape_metric_names,
        estimator_mode=headline.estimator_mode,
    )
    shape_metric_band = np.nanquantile(shape_metric_draws, [0.05, 0.50, 0.95], axis=0)
    shape_metrics_data: dict[str, Any] = {"date": dates.dt.strftime("%Y-%m-%d")}
    for name, values in surface_shape["metrics"].items():
        shape_metrics_data[name] = values
        if name in shape_metric_names:
            k = shape_metric_names.index(name)
            shape_metrics_data[f"{name}_p05"] = shape_metric_band[0, :, k]
            shape_metrics_data[f"{name}_median"] = shape_metric_band[1, :, k]
            shape_metrics_data[f"{name}_p95"] = shape_metric_band[2, :, k]
    shape_metrics_df = pd.DataFrame(shape_metrics_data)
    shape_metrics_df.to_csv(TABLES / "publication_grade_surface_shape_metrics.csv", index=False)

    horizons = np.repeat(np.arange(H + 1), pvars)
    variables = np.tile(np.array(labels), H + 1)
    date_str = dates.dt.strftime("%Y-%m-%d").to_numpy()
    shape_alloc_df = pd.DataFrame(
        {
            "date": np.repeat(date_str, m_dim),
            "horizon_months": np.tile(horizons, len(dates)),
            "variable": np.tile(variables, len(dates)),
            "shape_variance": surface_shape["surface_diag"].ravel(),
            "baseline_leverage": np.tile(surface_shape["baseline_leverage"], len(dates)),
            "relative_shape_variance": surface_shape["relative_shape_variance"].ravel(),
            "log_relative_shape_variance": surface_shape["log_relative_shape_variance"].ravel(),
            "shape_cell_share": (surface_shape["surface_diag"] / HEADLINE_R).ravel(),
            "baseline_cell_share": np.tile(surface_shape["baseline_leverage"] / HEADLINE_R, len(dates)),
        }
    )
    shape_alloc_df.to_csv(TABLES / "publication_grade_surface_shape_allocations.csv", index=False)

    episodes = selected_shape_episodes(dates, headline.tau, headline.scale_log_tau, surface_shape["surface_shape_rms_log_relative"], march_idx)
    pd.DataFrame(episodes).drop(columns=["idx"]).to_csv(TABLES / "publication_grade_surface_shape_episodes.csv", index=False)
    shape_trace = np.trace(surface_shape["shape"], axis1=1, axis2=2) / HEADLINE_R
    neutral_surface = surface_shape_from_A(np.eye(HEADLINE_R)[None, :, :], headline.V, H, pvars, labels)
    variable_share_cols = [f"variable_share_{safe_name(label)}" for label in labels]
    horizon_share_cols = [f"horizon_{hh:02d}_share" for hh in range(H + 1)]
    invariant_checks = pd.DataFrame(
        [
            {"check": "trace_S_over_R_equals_one", "max_abs_error": float(np.max(np.abs(shape_trace - 1.0)))},
            {"check": "sum_diag_C_equals_R", "max_abs_error": float(np.max(np.abs(surface_shape["surface_diag"].sum(axis=1) - HEADLINE_R)))},
            {"check": "variable_shares_sum_to_one", "max_abs_error": float(np.max(np.abs(shape_metrics_df[variable_share_cols].sum(axis=1) - 1.0)))},
            {"check": "horizon_shares_sum_to_one", "max_abs_error": float(np.max(np.abs(shape_metrics_df[horizon_share_cols].sum(axis=1) - 1.0)))},
            {"check": "neutral_shape_log_relative_zero", "max_abs_error": float(np.max(np.abs(neutral_surface["log_relative_shape_variance"])))},
        ]
    )
    invariant_checks.to_csv(TABLES / "publication_grade_surface_shape_invariant_checks.csv", index=False)
    if invariant_checks["max_abs_error"].max() > 1e-8:
        raise RuntimeError("Surface-shape invariant check failed")

    path_df = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "tau": headline.tau,
            "tau_point_p05": tau_band["point_low"],
            "tau_point_median": tau_band["point_med"],
            "tau_point_p95": tau_band["point_high"],
            "tau_simul_p05": tau_band["sim_low"],
            "tau_simul_p95": tau_band["sim_high"],
            "tau_full_pipeline_p05": boot_band["point_low"],
            "tau_full_pipeline_p95": boot_band["point_high"],
            "tau_full_pipeline_simul_p05": boot_band["sim_low"],
            "tau_full_pipeline_simul_p95": boot_band["sim_high"],
            "scale_log_tau": headline.scale_log_tau,
            "scale_point_p05": scale_band["point_low"],
            "scale_point_p95": scale_band["point_high"],
            "shape_distance": headline.shape_distance,
            "shape_point_p05": shape_band["point_low"],
            "shape_point_p95": shape_band["point_high"],
            "robust_observation_weight": headline.fit.weights,
            "reference_objective_weight": headline.fit.objective_weights
            if headline.fit.objective_weights is not None
            else np.ones(len(dates)),
            "tau_total_second_moment": np.trace(headline.total_second_moment_whitened, axis1=1, axis2=2) / HEADLINE_R
            if headline.total_second_moment_whitened is not None
            else headline.tau,
            "tau_mean_component": np.trace(headline.mean_component_whitened, axis1=1, axis2=2) / HEADLINE_R
            if headline.mean_component_whitened is not None
            else np.zeros(len(dates)),
            "tau_within_covariance": np.trace(headline.within_covariance_whitened, axis1=1, axis2=2) / HEADLINE_R
            if headline.within_covariance_whitened is not None
            else headline.tau,
            "prob_top10_by_state_draw": top10_probs,
        }
    )
    for name in shape_metric_names:
        path_df[name] = shape_metrics_df[name]
    for r in range(HEADLINE_R):
        path_df[f"A{r+1}{r+1}"] = headline.A[:, r, r]
    path_df.to_csv(TABLES / "publication_grade_headline_state_path.csv", index=False)

    uncertainty = pd.DataFrame(
        [
            {"quantity": "state_draws", "value": B_STATE},
            {"quantity": "tau_log_simultaneous_90_sup_crit", "value": tau_band["sup_crit"]},
            {"quantity": "tau_simultaneous_band_scale", "value": "log_tau_exponentiated"},
            {"quantity": "headline_tau_max", "value": float(headline.tau.max())},
            {"quantity": "headline_tau_max_month", "value": dates.iloc[int(np.argmax(headline.tau))].strftime("%Y-%m")},
            {"quantity": "headline_surface_shape_max", "value": float(surface_shape["surface_shape_rms_log_relative"].max())},
            {"quantity": "headline_surface_shape_max_month", "value": dates.iloc[int(np.argmax(surface_shape["surface_shape_rms_log_relative"]))].strftime("%Y-%m")},
            {"quantity": "march_2020_rank1_probability_state_draws", "value": march_rank1_prob},
            {"quantity": "march_2020_top10_probability_state_draws", "value": march_top10_prob},
            {"quantity": "minimum_student_t_weight", "value": float(headline.fit.weights.min())},
            {"quantity": "student_t_degrees_of_freedom", "value": ROBUST_NU},
            {"quantity": "student_t_minimum_weight_floor", "value": MIN_STUDENT_WEIGHT},
        ]
    )
    uncertainty.to_csv(TABLES / "publication_grade_uncertainty_summary.csv", index=False)

    shock_env = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")})
    for key in ["base5_headline", "base5_mp_pm_only", "base5_event_manual", "policy_without_cbi", "cbi_with_policy", "cbi_without_policy", "sf_fed_raw", "sf_fed_orthogonalized"]:
        if (key, HEADLINE_R) not in results:
            continue
        res = results[(key, HEADLINE_R)]
        tau_df = pd.DataFrame({"date": pd.to_datetime(res.dates).dt.strftime("%Y-%m-%d"), f"tau_{key}": res.tau})
        shock_env = shock_env.merge(tau_df, on="date", how="left")
    tau_cols = [c for c in shock_env.columns if c.startswith("tau_")]
    shock_env["tau_spec_min"] = shock_env[tau_cols].min(axis=1)
    shock_env["tau_spec_max"] = shock_env[tau_cols].max(axis=1)
    shock_env["tau_spec_range"] = shock_env["tau_spec_max"] - shock_env["tau_spec_min"]
    shock_env.to_csv(TABLES / "publication_grade_shock_construction_envelope.csv", index=False)

    subspace_rows = []
    for rank in RANKS:
        if ("base5_headline", rank) not in results:
            continue
        res = results[("base5_headline", rank)]
        angles = principal_angles(headline.V, res.V)
        subspace_rows.append(
            {
                "variant": "base5_headline",
                "comparison": f"R{rank} vs headline R{HEADLINE_R}",
                "rank": rank,
                "max_angle_degrees": float(np.max(angles)),
                "mean_angle_degrees": float(np.mean(angles)),
                **{f"angle_{j+1}_degrees": float(angles[j]) for j in range(len(angles))},
            }
        )
    subspace_df = pd.DataFrame(subspace_rows)
    subspace_df.to_csv(TABLES / "publication_grade_rank_subspace_angles.csv", index=False)

    boot_summary_rows = []
    for key, out in bootstrap_outputs.items():
        boot_df = out["boot_df"]
        clean = boot_df[boot_df.get("error").isna()] if len(boot_df) and "error" in boot_df.columns else boot_df
        if len(clean):
            boot_summary_rows.append(
                {
                    "variant": key,
                    "bootstrap_draws_requested": B_BOOT,
                    "bootstrap_draws_valid": int(len(clean)),
                    "alpha_p05": float(clean["alpha_hat"].quantile(0.05)),
                    "alpha_p50": float(clean["alpha_hat"].quantile(0.50)),
                    "alpha_p95": float(clean["alpha_hat"].quantile(0.95)),
                    "trace_share_p05": float(clean["retained_trace_share"].quantile(0.05)),
                    "trace_share_p95": float(clean["retained_trace_share"].quantile(0.95)),
                    "max_tau_p05": float(clean["max_tau"].quantile(0.05)),
                    "max_tau_p95": float(clean["max_tau"].quantile(0.95)),
                    "march_2020_rank1_probability_full_pipeline": float((clean["march_2020_rank"] == 1).mean()) if march_idx is not None and "march_2020_rank" in clean else np.nan,
                    "march_2020_top10_probability_full_pipeline": float((clean["march_2020_rank"] <= 10).mean()) if march_idx is not None and "march_2020_rank" in clean else np.nan,
                    "top10_overlap_with_baseline_median": float(clean["top10_overlap_with_baseline"].median()) if "top10_overlap_with_baseline" in clean else np.nan,
                    "tau_full_pipeline_log_simultaneous_90_sup_crit": out["band"]["sup_crit"],
                    "tau_full_pipeline_simultaneous_band_scale": "log_tau_exponentiated",
                }
            )
    boot_summary = pd.DataFrame(boot_summary_rows)
    boot_summary.to_csv(TABLES / "publication_grade_full_pipeline_bootstrap_summary.csv", index=False)

    extra_tables = write_requested_robustness_tables(specs_all, score_data, results, bootstrap_outputs, sf_meta)
    extra_tables["sample_coverage"] = sample_coverage
    print("Estimating full-coordinate Section 3.1 ridge-soft temporal-kernel backend", flush=True)
    section31 = build_full_coordinate_section31_outputs(
        score_data["base5_headline"],
        variant="base5_headline",
        label="Original five outcomes, original sample",
        dates=dates,
        outcome_labels=labels,
        march_idx=march_idx,
        bootstrap_draws=B_BOOT,
    )
    full_headline = section31["result"]
    full_tau_band = section31["tau_band"]
    full_shape = section31["shape"]
    full_block_df = section31["block_df"]
    full_concentration_df = section31["concentration_df"]
    full_episodes = section31["episodes"]
    path_df = section31["path_df"]
    uncertainty = section31["uncertainty"]

    fig = plt.figure(figsize=(10.5, 5.5))
    plt.plot(dates, full_headline.tau_soft, label="tau_soft estimate", color="black")
    plt.fill_between(dates, full_tau_band["point_low"], full_tau_band["point_high"], alpha=0.24, label="90% pointwise moving-block bootstrap band")
    plt.plot(dates, full_tau_band["sim_low"], linestyle="--", linewidth=1.1, label="90% log-simultaneous bootstrap lower")
    plt.plot(dates, full_tau_band["sim_high"], linestyle="--", linewidth=1.1, label="90% log-simultaneous bootstrap upper")
    plt.axhline(1.0, linewidth=0.8)
    score_range_title = f"{dates.iloc[0].strftime('%Y:%m')}-{dates.iloc[-1].strftime('%Y:%m')}"
    plt.title(f"Full-coordinate soft response-score covariance amplification, {score_range_title}")
    plt.ylabel("tau_soft")
    plt.legend()
    tau_ffbs_chart = savefig(fig, "01_full_coordinate_tau_soft_block_bootstrap_bands")

    fig = plt.figure(figsize=(10.5, 5.5))
    plt.plot(dates, full_headline.tau_soft, label="tau_soft estimate", color="black")
    plt.fill_between(dates, full_tau_band["point_low"], full_tau_band["point_high"], alpha=0.24, label="90% pointwise moving-block bootstrap band")
    plt.plot(dates, full_tau_band["sim_low"], linestyle="--", linewidth=1.1, label="90% log-simultaneous bootstrap lower")
    plt.plot(dates, full_tau_band["sim_high"], linestyle="--", linewidth=1.1, label="90% log-simultaneous bootstrap upper")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Full-coordinate soft response-score covariance amplification: block-bootstrap uncertainty")
    plt.ylabel("tau_soft")
    plt.legend()
    tau_full_pipeline_chart = savefig(fig, "01b_full_coordinate_tau_soft_block_bootstrap_bands")

    shape_heatmap_chart = plot_shape_heatmap_atlas(full_shape["cell_shape"], dates, full_episodes, H, labels)
    shape_marginals_chart = plot_full_coordinate_block_shape(full_block_df, full_concentration_df)
    shape_directions_chart, shape_direction_df = plot_shape_directions(surface_shape["shape"], headline.V, full_episodes, H, labels)
    shape_direction_df.to_csv(TABLES / "publication_grade_surface_shape_directions.csv", index=False)

    fig = plt.figure(figsize=(8.8, 5.2))
    base_rank = rank_summary[rank_summary["variant"] == "base5_headline"].copy()
    plt.bar(base_rank["rank"].astype(str), base_rank["retained_trace_share"])
    plt.ylabel("Retained trace share")
    plt.xlabel("Rank")
    plt.title("Rank sensitivity of retained response-score covariance geometry")
    rank_chart = savefig(fig, "06_rank_sensitivity")

    fig = plt.figure(figsize=(10.5, 5.4))
    for c in tau_cols:
        plt.plot(pd.to_datetime(shock_env["date"]), shock_env[c], label=c.replace("tau_", ""))
    plt.fill_between(pd.to_datetime(shock_env["date"]), shock_env["tau_spec_min"], shock_env["tau_spec_max"], alpha=0.14, label="specification envelope")
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t")
    plt.title("Shock-construction uncertainty envelope")
    plt.legend()
    shock_chart = savefig(fig, "07_shock_construction_envelope")

    fig = plt.figure(figsize=(10.5, 4.8))
    plt.plot(dates, headline.fit.weights)
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Student-t robust observation weights")
    plt.ylabel("weight")
    weights_chart = savefig(fig, "08_student_t_weights")

    charts = {
        "tau_ffbs": tau_ffbs_chart,
        "tau_full_pipeline": tau_full_pipeline_chart,
        "shape_heatmap": shape_heatmap_chart,
        "shape_marginals": shape_marginals_chart,
        "shape_directions": shape_directions_chart,
        "rank": rank_chart,
        "shock": shock_chart,
        "weights": weights_chart,
    }
    build_report(rank_summary, path_df, uncertainty, shock_env, boot_summary, charts, extra_tables=extra_tables)
    if WRITE_LEGACY_TOP5_COMPAT:
        write_top5_compatible_outputs(
            panels,
            score_data,
            headline,
            rank_summary,
            path_df,
            tau_draws,
            diag_draws,
            beta_boot,
            share_boot,
            mode_boot,
            subspace_angle_boot,
        )
    else:
        pd.DataFrame(
            [
                {
                    "legacy_output": "top5_compatible_outputs",
                    "status": "skipped",
                    "enable_with": "OVK_WRITE_LEGACY_TOP5_COMPAT=1",
                    "reason": "Section 3.1 figures now use the full-coordinate ridge-soft backend; legacy rank-five outputs are comparison-only.",
                }
            ]
        ).to_csv(TABLES / "legacy_rank_five_outputs_skipped.csv", index=False)
    write_robustness_compatible_outputs(panels, results)

    metadata = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "headline_outcomes": HEADLINE_OUTCOMES,
        "headline_variant": "base5_headline",
        "headline_rank": HEADLINE_R,
        "section31_headline_backend": "full_coordinate_temporal_kernel",
        "section31_coordinate_dimension": int(full_headline.chi.shape[1]),
        "section31_ridge_rho": float(full_headline.rho),
        "section31_soft_effective_dimension_d_rho": float(full_headline.d_rho),
        "section31_kernel_eta": float(full_headline.kernel_eta),
        "outcome_labels": labels,
        "outcome_surface_dimension": m_dim,
        "ranks": RANKS,
        "state_draws": B_STATE,
        "full_pipeline_bootstrap_draws": B_BOOT,
        "bootstrap_variants": BOOTSTRAP_VARIANTS,
        "bootstrap_block_length": BOOT_BLOCK_LEN,
        "student_t_degrees_of_freedom": ROBUST_NU,
        "student_t_minimum_weight_floor": MIN_STUDENT_WEIGHT,
        "covariance_estimator_mode": headline.estimator_mode,
        "default_covariance_estimator_mode": ESTIMATOR_MODE,
        "arithmetic_reference_ridge_scale": ARITHMETIC_REFERENCE_RIDGE_SCALE,
        "alpha_grid": ALPHA_GRID.tolist(),
        "workers": PUBLICATION_WORKERS,
        "worker_benchmark": worker_benchmark,
        "sf_fed": sf_meta,
        "sample_coverage": dict(zip(sample_coverage["item"].astype(str), sample_coverage["value"].astype(str))),
        "placebo_seed": PLACEBO_SEED,
        "placebo_shift_months": PLACEBO_SHIFT_MONTHS,
        "cache_dir": str(CACHE_DIR),
        "cache_disabled": DISABLE_CACHE,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "estimand_language": "full-coordinate time-varying response-score covariance; dynamic covariance operator of LP surfaces; soft ridge-whitened covariance amplification of monetary-policy response surfaces",
        "estimation_upgrades": [
            "Figures 1-2 use the full 125-coordinate working grid, soft ridge whitening, and no spectral cutoff",
            "full-coordinate temporal-kernel backend K_hat[t]=sum_s w[t,s] chi_s chi_s' with normalized nonnegative average-preserving weights",
            "Figure 1 bands are moving-block bootstrap bands, not FFBS state-uncertainty bands",
            "base-five headline outcome set with same-sample eight-outcome expectations comparison",
            "raw and equal-outcome-trace standardized K and K_std outputs",
            "placebo shock, policy/CBI split, SF Fed/Bauer-Swanson appendix shock, and smooth-LP robustness variants",
            "episode-level spike uncertainty from full-pipeline bootstrap tau paths",
            "arithmetic outer-product total second-moment estimator is the default; log_spd_legacy is deprecated comparison mode",
            "reference weights are normalized, used in C_hat, and used in the state smoother objective",
            "alpha selected only by deprecated log_spd_legacy mode",
            "F, Q, and R estimated by EM-style state-space iterations",
            "Student-t robust observation weighting",
        ],
    }
    (OUT / "publication_grade_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), CODE / "run_publication_grade_ovk.py")
    helper = Path(__file__).with_name("ovk_data.py")
    if helper.exists():
        shutil.copy2(helper, CODE / "ovk_data.py")
    tmp_worker_arrays = OUT / "_worker_arrays"
    if tmp_worker_arrays.exists():
        rmtree_with_memmap_retry(tmp_worker_arrays)

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, arcname=f.relative_to(OUT))
        z.write(FINAL_PDF, arcname=FINAL_PDF.name)
        z.write(FINAL_HTML, arcname=FINAL_HTML.name)

    print("DONE")
    print("PDF", FINAL_PDF, FINAL_PDF.stat().st_size)
    print("HTML", FINAL_HTML, FINAL_HTML.stat().st_size)
    print("ZIP", FINAL_ZIP, FINAL_ZIP.stat().st_size)
    print(rank_summary[rank_summary["variant"].eq("base5_headline")].to_string(index=False))


if __name__ == "__main__":
    main()
