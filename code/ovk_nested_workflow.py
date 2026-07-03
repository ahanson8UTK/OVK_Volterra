#!/usr/bin/env python3
"""
Nested LP-score mean/covariance workflow for monthly monetary-policy OVK project.

Models:
M0: q_t = beta + u_t,               K_t = K
M1: q_t = beta + u_t,               K_t dynamic
M2: q_t = theta_hat + moving_center_t + u_t, K_t = K
M3: q_t = theta_hat + moving_center_t + u_t, K_t dynamic

The production implementation scores the complete LP response-score vector after
full-rank ridge whitening.  Predictive model comparison is based on one-step
Gaussian scores with causal arithmetic outer-product covariance updates.  The
legacy reduced-basis workflow is retained only behind OVK_NESTED_MODE=legacy-reduced.
"""
from __future__ import annotations

import os
import sys
import math
import json
import html
import zipfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from run_publication_grade_ovk import (
    ALPHA_GRID,
    EM_ITERS,
    HEADLINE_R,
    ROBUST_NU,
    estimate_alpha_and_state,
    matrix_series_from_state,
    scale_shape_from_A,
)
import ovk_full_coordinate_nested as full_nested
from ovk_data import BASE_OUTCOME_COLUMNS, OUTCOME_SPECS, build_outcome_frame

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        PageBreak, KeepTogether
    )
except Exception as e:
    raise RuntimeError("ReportLab is required to build the PDF report.") from e

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(os.environ.get("OVK_BASE_DIR", "/mnt/data"))
PANEL_PATH = Path(os.environ.get(
    "OVK_NESTED_PANEL_PATH",
    str(ROOT / "monthly_ovk_state_space_uncertainty" / "outputs" / "ovk_monetary_panel_monthly_fixed_full.csv"),
))
OUTDIR = Path(os.environ.get("OVK_NESTED_OUTDIR", str(ROOT / "ovk_nested_mean_cov_workflow")))
CHART_DIR = OUTDIR / "charts"
TABLE_DIR = OUTDIR / "tables"
CODE_DIR = OUTDIR / "code"
for d in [OUTDIR, CHART_DIR, TABLE_DIR, CODE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FINAL_PDF = Path(os.environ.get("OVK_NESTED_FINAL_PDF", str(ROOT / "nested_mean_covariance_ovk_report.pdf")))
FINAL_ZIP = Path(os.environ.get("OVK_NESTED_FINAL_ZIP", str(ROOT / "nested_mean_covariance_ovk_workflow_bundle.zip")))
REPORT_TITLE = os.environ.get("OVK_NESTED_REPORT_TITLE", "Nested Mean-Covariance LP/OVK Workflow")
REPORT_SUBTITLE = os.environ.get(
    "OVK_NESTED_REPORT_SUBTITLE",
    "Monthly monetary-policy score surfaces; four nested models for moving IRFs and dynamic uncertainty geometry",
)
NESTED_BOOTSTRAP_DRAWS = int(os.environ.get("OVK_NESTED_SCORE_BOOTSTRAP_DRAWS", "2000"))
NESTED_BOOTSTRAP_BLOCK_LEN = int(os.environ.get("OVK_NESTED_SCORE_BOOTSTRAP_BLOCK_LEN", "12"))
NESTED_MODE = os.environ.get("OVK_NESTED_MODE", "full-coordinate-arithmetic").strip().lower()
ALL_OUTCOME_COLUMNS = tuple(spec.column for spec in OUTCOME_SPECS)
for p in [FINAL_PDF, FINAL_ZIP]:
    p.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Numeric utilities
# -----------------------------
def regularize_cov(S: np.ndarray, ridge_scale: float = 1e-6) -> np.ndarray:
    """Symmetrize and ridge-regularize a covariance matrix."""
    S = np.asarray(S, float)
    S = 0.5 * (S + S.T)
    d = S.shape[0]
    tr = np.trace(S) / max(d, 1)
    if not np.isfinite(tr) or tr <= 0:
        tr = 1.0
    vals = np.linalg.eigvalsh(S)
    minv = vals.min()
    ridge = max(ridge_scale * tr, -minv + ridge_scale * tr, 1e-10)
    return S + ridge * np.eye(d)


def shrink_cov(S: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Shrink covariance to its diagonal target."""
    S = regularize_cov(S)
    D = np.diag(np.diag(S))
    return regularize_cov((1 - alpha) * S + alpha * D)


def logpdf_gaussian(e: np.ndarray, S: np.ndarray) -> float:
    """Multivariate Gaussian log-density for zero-mean residual e."""
    S = regularize_cov(S)
    d = len(e)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(logdet):
        S = regularize_cov(S, ridge_scale=1e-4)
        sign, logdet = np.linalg.slogdet(S)
    try:
        sol = np.linalg.solve(S, e)
    except np.linalg.LinAlgError:
        S = regularize_cov(S, ridge_scale=1e-4)
        sol = np.linalg.solve(S, e)
        sign, logdet = np.linalg.slogdet(S)
    quad = float(e @ sol)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def gram_schmidt_columns(cols: List[np.ndarray], tol: float = 1e-8, max_dim: Optional[int] = None) -> np.ndarray:
    basis = []
    for c in cols:
        v = np.asarray(c, float).copy()
        n0 = np.linalg.norm(v)
        if not np.isfinite(n0) or n0 < tol:
            continue
        for b in basis:
            v -= b * float(b @ v)
        n = np.linalg.norm(v)
        if n > tol:
            basis.append(v / n)
        if max_dim is not None and len(basis) >= max_dim:
            break
    if not basis:
        raise ValueError("No basis vectors survived Gram-Schmidt.")
    return np.column_stack(basis)


def block_bootstrap_ci(x: np.ndarray, block_len: int = 12, B: int = 2000, seed: int = 123) -> Tuple[float, float, float, float]:
    """CI for the mean of x using circular moving-block bootstrap. Returns mean, p05, p95, prob_mean_gt_0."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
        idx = np.concatenate([((s + np.arange(block_len)) % n) for s in starts])[:n]
        means[b] = x[idx].mean()
    return float(x.mean()), float(np.percentile(means, 5)), float(np.percentile(means, 95)), float(np.mean(means > 0))

# -----------------------------
# Data and LP score construction
# -----------------------------
@dataclass(frozen=True)
class NestedVariant:
    key: str
    label: str
    outcome_columns: tuple[str, ...]


def panel_with_outcomes(panel: pd.DataFrame, outcome_columns: tuple[str, ...]) -> pd.DataFrame:
    """Return panel with only the requested recognized outcome columns."""
    keep = set(outcome_columns)
    missing = [c for c in outcome_columns if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing requested outcome columns: {', '.join(missing)}")
    drop = [c for c in ALL_OUTCOME_COLUMNS if c in panel.columns and c not in keep]
    return panel.drop(columns=drop)


def nested_variant_specs(panel: pd.DataFrame) -> list[NestedVariant]:
    requested = [
        key.strip()
        for key in os.environ.get("OVK_NESTED_VARIANTS", "base5_headline,all8_expectation_overlap").split(",")
        if key.strip()
    ]
    available = set(panel.columns)
    specs: dict[str, NestedVariant] = {
        "base5_headline": NestedVariant(
            "base5_headline",
            "Headline original five outcomes",
            tuple(BASE_OUTCOME_COLUMNS),
        ),
        "all8_expectation_overlap": NestedVariant(
            "all8_expectation_overlap",
            "Eight outcomes with expectations",
            tuple(ALL_OUTCOME_COLUMNS),
        ),
    }
    out: list[NestedVariant] = []
    for key in requested:
        if key not in specs:
            raise ValueError(f"Unknown nested variant {key!r}. Valid choices: {', '.join(specs)}")
        spec = specs[key]
        missing = [c for c in spec.outcome_columns if c not in available]
        if missing:
            raise ValueError(
                f"Nested variant {key!r} requires missing outcome columns: {', '.join(missing)}. "
                "Use the pipeline-prepared panel with expectations or set OVK_NESTED_VARIANTS=base5_headline."
            )
        out.append(spec)
    return out


def build_lp_scores(
    panel: pd.DataFrame,
    H: int = 24,
    L: int = 12,
    outcome_columns: tuple[str, ...] | None = None,
) -> Dict[str, object]:
    panel = panel.sort_values("date").reset_index(drop=True).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel_with_outcomes(panel, tuple(outcome_columns or BASE_OUTCOME_COLUMNS))
    Ybase = build_outcome_frame(panel)
    outcome_labels = list(Ybase.columns)
    mvals = panel["MP_used"].to_numpy(float)
    cvals = panel["CBI_used"].to_numpy(float)
    mstd = mvals / np.nanstd(mvals)
    cstd = cvals / np.nanstd(cvals)

    Yarr = Ybase.to_numpy(float)
    dYarr = np.vstack([np.full((1, Yarr.shape[1]), np.nan), np.diff(Yarr, axis=0)])
    T, pvars = Yarr.shape
    valid = []
    for t in range(T):
        if t - L < 0 or t - 1 < 0 or t + H >= T:
            continue
        ok = (
            np.isfinite(mstd[t]) and np.isfinite(cstd[t]) and
            np.isfinite(mstd[t-L:t]).all() and np.isfinite(cstd[t-L:t]).all() and
            np.isfinite(Yarr[t-1:t+H+1, :]).all() and
            np.isfinite(Yarr[t-L:t, :]).all() and
            np.isfinite(dYarr[t-L:t, :]).all()
        )
        if ok:
            valid.append(t)
    valid = np.array(valid, dtype=int)
    controls = [np.ones(len(valid)), valid.astype(float), cstd[valid]]
    for lag in range(1, L + 1):
        controls += [mstd[valid - lag], cstd[valid - lag]]
    for lag in range(1, L + 1):
        controls += [Yarr[valid - lag, :], dYarr[valid - lag, :]]
    X = np.hstack([
        np.asarray(a)[:, None] if np.asarray(a).ndim == 1 else np.asarray(a)
        for a in controls
    ])
    Yresp = np.hstack([
        Yarr[valid + h, :] - Yarr[valid - 1, :]
        for h in range(H + 1)
    ])
    Xs = X.copy()
    mu = Xs[:, 1:].mean(axis=0)
    sd = Xs[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    Xs[:, 1:] = (Xs[:, 1:] - mu) / sd
    m = mstd[valid]
    B = np.linalg.lstsq(Xs, np.column_stack([m, Yresp]), rcond=None)[0]
    resid = np.column_stack([m, Yresp]) - Xs @ B
    m_res = resid[:, 0]
    Y_res = resid[:, 1:]
    sigma_m2 = float(np.mean(m_res ** 2))
    Q = (m_res[:, None] * Y_res) / sigma_m2
    dates = panel["date"].iloc[valid].reset_index(drop=True)
    return {
        "Q": Q,
        "dates": dates,
        "valid_idx": valid,
        "outcome_labels": outcome_labels,
        "H": H,
        "L": L,
        "pvars": pvars,
        "panel": panel,
    }

# -----------------------------
# Basis construction
# -----------------------------
def covariance_eigenbasis(Q: np.ndarray, train_end: Optional[int] = None) -> Dict[str, np.ndarray]:
    if train_end is None:
        Quse = Q
    else:
        Quse = Q[:train_end]
    beta = Quse.mean(axis=0)
    E = Quse - beta
    K = (E.T @ E) / len(E)
    evals, evecs = np.linalg.eigh(K)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    V = evecs.copy()
    for r in range(V.shape[1]):
        j = np.argmax(np.abs(V[:, r]))
        if V[j, r] < 0:
            V[:, r] *= -1
    return {"beta": beta, "K": K, "evals": evals, "V": V}


def build_mean_and_eval_basis(beta: np.ndarray, V: np.ndarray, d_eval: int = 10) -> Dict[str, np.ndarray]:
    beta_norm = np.linalg.norm(beta)
    if beta_norm < 1e-10:
        b1 = V[:, 0]
    else:
        b1 = beta / beta_norm
    # Mode 4 was empirically important for the average IRF projection in the prior run.
    v4 = V[:, 3]
    b2_raw = v4 - b1 * float(b1 @ v4)
    if np.linalg.norm(b2_raw) < 1e-8:
        v5 = V[:, 4]
        b2_raw = v5 - b1 * float(b1 @ v5)
    b2 = b2_raw / np.linalg.norm(b2_raw)
    Bbeta = np.column_stack([b1, b2])
    cols = [b1, b2]
    for r in range(20):
        cols.append(V[:, r])
    W = gram_schmidt_columns(cols, max_dim=d_eval)
    return {"Bbeta": Bbeta, "W": W}

# -----------------------------
# Model filtering and predictive scoring
# -----------------------------
@dataclass
class ModelParams:
    kmean: float = 0.05
    kcov: float = 0.05
    phi: float = 0.98
    cov_shrink: float = 0.15
    cov_target_weight: float = 0.10


def initial_dynamic_mean_residuals(y: np.ndarray, mu: np.ndarray, Bc: np.ndarray, params: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    """Run the mean filter through y and return forecast residuals and final gamma."""
    m = Bc.shape[1]
    gamma = np.zeros(m)
    residuals = []
    for t in range(len(y)):
        gamma_pred = params.phi * gamma
        mean_pred = mu + Bc @ gamma_pred
        e = y[t] - mean_pred
        residuals.append(e)
        gamma_obs = Bc.T @ (y[t] - mu)
        gamma = (1 - params.kmean) * gamma_pred + params.kmean * gamma_obs
    return np.asarray(residuals), gamma


def run_predictive_scores(
    y: np.ndarray,
    Bc: np.ndarray,
    model: str,
    start: int,
    end: int,
    params: ModelParams,
    init_start: int = 0,
) -> Dict[str, object]:
    """
    Compute one-step-ahead predictive log scores for y[start:end].
    Initial state is estimated from y[init_start:start].
    """
    y0 = y[init_start:start]
    d = y.shape[1]
    mu = y0.mean(axis=0)
    dynamic_mean = model in ["M2", "M3"]
    dynamic_cov = model in ["M1", "M3"]

    if dynamic_mean:
        init_resids, gamma = initial_dynamic_mean_residuals(y0, mu, Bc, params)
        Sbase = shrink_cov(np.cov(init_resids.T, bias=True), params.cov_shrink)
    else:
        gamma = np.zeros(Bc.shape[1])
        init_resids = y0 - mu
        Sbase = shrink_cov(np.cov(init_resids.T, bias=True), params.cov_shrink)
    S_target = Sbase.copy()
    S_filt = Sbase.copy()
    S_const = Sbase.copy()

    log_scores = []
    residuals = []
    means = []
    cov_diags = []
    gammas_pred = []

    for t in range(start, end):
        if dynamic_mean:
            gamma_pred = params.phi * gamma
            mean_pred = mu + Bc @ gamma_pred
        else:
            gamma_pred = gamma.copy()
            mean_pred = mu
        if dynamic_cov:
            S_pred = (1 - params.cov_target_weight) * S_filt + params.cov_target_weight * S_target
            S_pred = shrink_cov(S_pred, params.cov_shrink)
        else:
            S_pred = S_const
        e = y[t] - mean_pred
        lp = logpdf_gaussian(e, S_pred)
        log_scores.append(lp)
        residuals.append(e)
        means.append(mean_pred)
        cov_diags.append(np.diag(S_pred))
        gammas_pred.append(gamma_pred)

        # Update filters with observed y_t.
        if dynamic_mean:
            gamma_obs = Bc.T @ (y[t] - mu)
            gamma = (1 - params.kmean) * gamma_pred + params.kmean * gamma_obs
        if dynamic_cov:
            S_filt = (1 - params.kcov) * S_filt + params.kcov * np.outer(e, e)
            S_filt = regularize_cov(S_filt)

    return {
        "log_scores": np.asarray(log_scores),
        "residuals": np.asarray(residuals),
        "means": np.asarray(means),
        "cov_diags": np.asarray(cov_diags),
        "gammas_pred": np.asarray(gammas_pred),
        "mu": mu,
        "Sbase": Sbase,
        "params": params,
    }


def tune_model(y: np.ndarray, Bc: np.ndarray, model: str, train_end: int, val_start: int) -> ModelParams:
    kmeans = [0.015, 0.03, 0.05, 0.08, 0.12]
    kcovs = [0.015, 0.03, 0.05, 0.08, 0.12]
    best_score = -np.inf
    best = ModelParams()
    if model == "M0":
        return ModelParams(kmean=0.0, kcov=0.0)
    if model == "M1":
        for kc in kcovs:
            p = ModelParams(kmean=0.0, kcov=kc)
            out = run_predictive_scores(y, Bc, model, val_start, train_end, p)
            score = np.mean(out["log_scores"])
            if score > best_score:
                best_score = score
                best = p
    elif model == "M2":
        for km in kmeans:
            p = ModelParams(kmean=km, kcov=0.0)
            out = run_predictive_scores(y, Bc, model, val_start, train_end, p)
            score = np.mean(out["log_scores"])
            if score > best_score:
                best_score = score
                best = p
    elif model == "M3":
        for km in kmeans:
            for kc in kcovs:
                p = ModelParams(kmean=km, kcov=kc)
                out = run_predictive_scores(y, Bc, model, val_start, train_end, p)
                score = np.mean(out["log_scores"])
                if score > best_score:
                    best_score = score
                    best = p
    return best

# -----------------------------
# Dynamic A_t survival diagnostic in original OVK top-3 basis
# -----------------------------
def filtered_gamma_path(y_centered: np.ndarray, Bc: np.ndarray, kmean: float, phi: float = 0.98) -> np.ndarray:
    m = Bc.shape[1]
    gamma = np.zeros(m)
    path = np.zeros((len(y_centered), m))
    for t in range(len(y_centered)):
        gamma_pred = phi * gamma
        path[t] = gamma_pred
        gamma_obs = Bc.T @ y_centered[t]
        gamma = (1 - kmean) * gamma_pred + kmean * gamma_obs
    return path


def ewma_A_from_z(z: np.ndarray, kcov: float, normalize_mean: bool = True) -> np.ndarray:
    R = z.shape[1]
    A = np.zeros((len(z), R, R))
    cur = np.eye(R)
    for t in range(len(z)):
        cur = (1 - kcov) * cur + kcov * np.outer(z[t], z[t])
        cur = regularize_cov(cur, ridge_scale=1e-8)
        A[t] = cur
    if normalize_mean:
        Abar = A.mean(axis=0)
        vals, vecs = np.linalg.eigh(regularize_cov(Abar, ridge_scale=1e-8))
        invsqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T
        A = np.asarray([invsqrt @ a @ invsqrt for a in A])
    return A


def upgraded_state_space_A_from_z(z: np.ndarray, rank: int) -> Dict[str, object]:
    """Estimate A_t from standardized basis scores using the shared publication-grade core."""
    fit = estimate_alpha_and_state(z, rank=rank, em_iters=EM_ITERS, nu=ROBUST_NU, alpha_grid=ALPHA_GRID)
    A = matrix_series_from_state(fit.xs, rank)
    tau, scale_log_tau, shape_distance = scale_shape_from_A(A)
    return {
        "fit": fit,
        "A": A,
        "tau": tau,
        "scale_log_tau": scale_log_tau,
        "shape_distance": shape_distance,
    }

# -----------------------------
# Plot helpers
# -----------------------------
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_logscore_bars(model_summary: pd.DataFrame, path: Path):
    models = model_summary["model"].tolist()
    vals = model_summary["avg_log_score"].to_numpy()
    plt.figure(figsize=(8, 4.8))
    plt.bar(models, vals)
    plt.axhline(model_summary.loc[model_summary["model"] == "M0", "avg_log_score"].iloc[0], linestyle="--", linewidth=1)
    plt.title("Out-of-sample predictive quasi-log score")
    plt.xlabel("Model")
    plt.ylabel("Average one-step log score")
    save_fig(path)


def plot_logscore_diff_cumulative(dates: pd.Series, score_dict: Dict[str, np.ndarray], path: Path):
    plt.figure(figsize=(9, 5))
    base = score_dict["M0"]
    for m in ["M1", "M2", "M3"]:
        plt.plot(dates, np.cumsum(score_dict[m] - base), label=f"{m} - M0")
    plt.axhline(0, linewidth=0.8)
    plt.title("Cumulative out-of-sample log score difference vs M0")
    plt.xlabel("Date")
    plt.ylabel("Cumulative log score difference")
    plt.legend()
    save_fig(path)


def plot_gamma_path(dates: pd.Series, gammas: np.ndarray, path: Path):
    plt.figure(figsize=(9, 5))
    for j in range(gammas.shape[1]):
        plt.plot(dates, gammas[:, j], label=f"gamma {j+1}")
    plt.axhline(0, linewidth=0.8)
    plt.title("Predicted time-varying mean loadings in B_beta basis")
    plt.xlabel("Date")
    plt.ylabel("Loading")
    plt.legend()
    save_fig(path)


def plot_A_survival(dates: pd.Series, trace1: np.ndarray, trace3: np.ndarray, path: Path):
    plt.figure(figsize=(9, 5))
    plt.plot(dates, trace1, label="legacy audit M1 fixed-mean A_t")
    plt.plot(dates, trace3, label="legacy audit M3 mean-adjusted A_t")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Do response-score covariance spikes survive after mean adjustment?")
    plt.xlabel("Date")
    plt.ylabel("legacy rank-reduced trace(A_t)")
    plt.legend()
    save_fig(path)


def plot_mean_vs_cov_norm(dates: pd.Series, mean_norm: np.ndarray, traceA: np.ndarray, path: Path):
    plt.figure(figsize=(9, 5))
    mn = mean_norm / np.nanmedian(mean_norm[mean_norm > 0]) if np.any(mean_norm > 0) else mean_norm
    ta = traceA / np.nanmedian(traceA[traceA > 0]) if np.any(traceA > 0) else traceA
    plt.plot(dates, mn, label="Mean-drift norm, scaled")
    plt.plot(dates, ta, label="Covariance amplification, scaled")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Moving-center signal vs residual covariance amplification")
    plt.xlabel("Date")
    plt.ylabel("Relative scale")
    plt.legend()
    save_fig(path)


def plot_full_coordinate_mean_within(mean_within: pd.DataFrame, path: Path):
    x = pd.to_datetime(mean_within["date"])
    plt.figure(figsize=(9, 5))
    plt.plot(x, mean_within["tau_total"], label="total second moment")
    plt.plot(x, mean_within["tau_mean"], label="local moving-center component")
    plt.plot(x, mean_within["tau_within"], label="within component")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Full-coordinate mean/within decomposition")
    plt.xlabel("Date")
    plt.ylabel("ridge-soft tau")
    plt.legend()
    save_fig(path)


def plot_full_coordinate_mean_vs_within(mean_within: pd.DataFrame, path: Path):
    x = pd.to_datetime(mean_within["date"])
    local = mean_within["local_mean_norm"].to_numpy(float)
    within = mean_within["tau_within"].to_numpy(float)
    local_scaled = local / max(float(np.nanmedian(local[local > 0])) if np.any(local > 0) else 1.0, 1e-12)
    within_scaled = within / max(float(np.nanmedian(within[within > 0])) if np.any(within > 0) else 1.0, 1e-12)
    plt.figure(figsize=(9, 5))
    plt.plot(x, local_scaled, label="local moving-center norm")
    plt.plot(x, within_scaled, label="within covariance tau")
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Moving center versus within-coordinate covariance amplification")
    plt.xlabel("Date")
    plt.ylabel("Median-scaled value")
    plt.legend()
    save_fig(path)

# -----------------------------
# PDF report
# -----------------------------
def para(text, style):
    return Paragraph(text, style)


PDF_TABLE_WIDTH = 7.15 * inch


def _pdf_table_cell(value, float_fmt: str = "{:.3f}", header: bool = False) -> Paragraph:
    if isinstance(value, (float, np.floating)):
        text = "" if not np.isfinite(float(value)) else float_fmt.format(float(value))
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
        fontSize=5.8 if header else 5.5,
        leading=6.5 if header else 6.2,
        wordWrap="CJK",
    )
    return Paragraph(text, style)


def _pdf_table_widths(columns: list[object]) -> list[float]:
    weights = []
    for col in columns:
        name = str(col).lower()
        if any(token in name for token in ["interpretation", "covariance", "description"]):
            weights.append(1.8)
        elif any(token in name for token in ["model", "sample", "episode", "month", "date"]):
            weights.append(1.05)
        else:
            weights.append(0.85)
    scale = PDF_TABLE_WIDTH / max(sum(weights), 1.0)
    return [w * scale for w in weights]


def df_to_table(df: pd.DataFrame, max_rows: int = 20, float_fmt: str = "{:.3f}", max_cols: int = 9) -> Table:
    dfx = df.copy().head(max_rows)
    if len(dfx.columns) > max_cols:
        dfx = dfx.iloc[:, :max_cols].copy()
    if dfx.empty and len(dfx.columns) == 0:
        dfx = pd.DataFrame({"note": ["No rows available."]})
    data = [[_pdf_table_cell(c, float_fmt, header=True) for c in dfx.columns]]
    for _, row in dfx.iterrows():
        data.append([_pdf_table_cell(v, float_fmt) for v in row.tolist()])
    tbl = Table(data, repeatRows=1, colWidths=_pdf_table_widths(list(dfx.columns)), hAlign="LEFT", splitByRow=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
    ]))
    return tbl


def build_pdf_report(pdf_path: Path, results: Dict[str, object]):
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.2, leading=12))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=9))

    story = []
    story.append(para(REPORT_TITLE, styles["Title2"]))
    story.append(para(REPORT_SUBTITLE, styles["Bodyx"]))
    story.append(para("Full-coordinate nested predictive comparison", styles["H1x"]))
    story.append(para(
        "This nested comparison is the paper's fit anchor. It separates a moving LP-response center from a dynamic covariance operator of LP score surfaces, so the main claim can stay focused on response-score covariance amplification rather than unqualified state-dependent structural IRFs.",
        styles["Bodyx"],
    ))
    story.append(Spacer(1, 0.12 * inch))

    sample = results["sample_table"]
    story.append(para("1. Empirical setup", styles["H1x"]))
    story.append(para(
        "The analysis rebuilds the monthly LP score surfaces used in the previous OVK work. "
        "The score contribution q_t is a stacked response surface over horizons 0-24 months and the balanced outcome variables in the merged panel. "
        "The monetary shock is MP_used, standardized to one monthly standard deviation, with CBI_used included as a control shock. Controls include a trend and 12 lags of shocks, macro levels, and macro first differences.",
        styles["Bodyx"]
    ))
    story.append(df_to_table(sample, max_rows=20))

    story.append(para("2. The four nested models", styles["H1x"]))
    story.append(para(
        "All four models are estimated and scored in the complete LP response-score coordinate system after a full-rank ridge-whitening transform. "
        "No reduced evaluation basis or covariance eigenvectors enter the production Table I values.",
        styles["Bodyx"]
    ))
    story.append(para(
        "In the contrast table, avg_log_score_diff is an average over evaluation dates, not a cumulative sum. The sum_log_score_diff and avg_joint_log_score_diff_per_dimension columns are reported separately to make the scale explicit.",
        styles["Bodyx"]
    ))
    model_defs = pd.DataFrame({
        "Model": ["M0", "M1", "M2", "M3"],
        "Mean": ["fixed theta_hat", "fixed theta_hat", "full-coordinate moving center", "full-coordinate moving center"],
        "Covariance": ["constant K", "dynamic K_t", "constant K", "dynamic K_t"],
        "Interpretation": [
            "Standard average LP score model.",
            "Dynamic covariance operator: response-score covariance changes around a fixed center.",
            "Time-varying IRF surface: center moves, covariance fixed.",
            "Joint model: center moves and uncertainty geometry changes."
        ]
    })
    story.append(df_to_table(model_defs, max_rows=4))
    story.append(para(
        "Dynamic means are estimated by a common-persistence recursion in all coordinates. "
        "Dynamic covariances are estimated by a causal arithmetic outer-product recursion. "
        "The nested predictive comparison remains a one-step Gaussian predictive-score diagnostic, not a likelihood-ratio test.",
        styles["Bodyx"]
    ))

    story.append(para("3. Predictive log-score comparison", styles["H1x"]))
    story.append(para(
        "Higher log scores indicate better one-step predictive density for the full LP score surface. "
        "The key comparisons are M2-M0 for evidence of time-varying mean IRFs, M1-M0 for evidence of time-varying uncertainty, and M3 relative to both M1 and M2 for evidence that both moments move.",
        styles["Bodyx"]
    ))
    story.append(df_to_table(results["model_summary"], max_rows=4))
    story.append(Spacer(1, 0.05 * inch))
    img = Image(str(results["charts"]["logscore_bars"]), width=6.8 * inch, height=3.9 * inch)
    story.append(img)
    story.append(Spacer(1, 0.05 * inch))
    story.append(df_to_table(results["comparison_table"], max_rows=8))
    if "covariance_diagnostics" in results:
        cov_cols = [
            "model",
            "score_dimension",
            "covariance_estimator",
            "statistical_shrinkage",
            "training_residual_count",
            "training_residual_rank",
            "covariance_floor",
            "min_predicted_cov_eigenvalue",
            "jittered_periods",
        ]
        cov_cols = [c for c in cov_cols if c in results["covariance_diagnostics"].columns]
        story.append(para("Covariance numerical diagnostics", styles["H2x"]))
        story.append(df_to_table(results["covariance_diagnostics"][cov_cols], max_rows=4, max_cols=len(cov_cols)))
    story.append(PageBreak())

    story.append(para("4. Cumulative evidence over time", styles["H1x"]))
    story.append(para(
        "The cumulative log-score plot shows when each model gains or loses predictive likelihood relative to the fixed-mean, fixed-covariance benchmark M0. "
        "Persistent upward movement is evidence that the alternative model captures a recurring feature of the score process rather than a single isolated observation.",
        styles["Bodyx"]
    ))
    story.append(Image(str(results["charts"]["cumulative"]), width=6.8 * inch, height=3.9 * inch))
    story.append(para("5. Full-coordinate moving-center component", styles["H1x"]))
    story.append(para(
        "The moving-center component is a full-coordinate common-persistence recursion. "
        "This is the part of the workflow that tests whether a moving-center component improves prediction.",
        styles["Bodyx"]
    ))
    story.append(Image(str(results["charts"]["gamma"]), width=6.8 * inch, height=3.9 * inch))
    story.append(PageBreak())

    story.append(para("6. Mean/within covariance diagnostic", styles["H1x"]))
    story.append(para(
        "The separate diagnostic decomposes the full-coordinate descriptive second moment into local moving-center and within components using the same soft ridge geometry. "
        "This audit is descriptive and does not feed back into the causal predictive filters.",
        styles["Bodyx"]
    ))
    story.append(df_to_table(results["survival_summary"], max_rows=20))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Image(str(results["charts"]["survival"]), width=6.8 * inch, height=3.9 * inch))
    story.append(Spacer(1, 0.05 * inch))
    story.append(df_to_table(results["top_spikes"], max_rows=12))
    story.append(PageBreak())

    story.append(para("7. Moving-center signal versus within covariance", styles["H1x"]))
    story.append(para(
        "This plot compares the scaled descriptive moving-center norm with the within-coordinate covariance component. "
        "It is an audit artifact, not an input into M0-M3 scoring.",
        styles["Bodyx"]
    ))
    story.append(Image(str(results["charts"]["mean_cov"]), width=6.8 * inch, height=3.9 * inch))

    story.append(para("8. Economic interpretation", styles["H1x"]))
    econ_text = results["economic_interpretation"]
    for paragraph in econ_text.split("\n\n"):
        story.append(para(paragraph, styles["Bodyx"]))
        story.append(Spacer(1, 0.04 * inch))

    story.append(para("9. Implementation caveats", styles["H1x"]))
    caveats = [
        "The comparison is conditional on the constructed LP score surface. It is not a full-sample structural likelihood for the macroeconomy.",
        "The production nested path uses full-coordinate ridge-whitened Gaussian scores and causal arithmetic covariance updates; legacy reduced-rank state-space code is audit-only.",
        "The displayed contrast column is an average per evaluation date; the cumulative sum and per-coordinate average are separate columns.",
        "The ridge geometry and persistence values are fixed before final out-of-sample scoring from the pre-evaluation period.",
        "A positive log-score advantage for M1 or M3 is evidence of improved predictive density for LP score surfaces. It is not by itself proof that the structural IRF changes.",
        "If M3 dominates M1 and M2, the data favor a joint moving-mean and moving-covariance description. If M1 dominates and A_t spikes survive mean adjustment, the response-score covariance interpretation is stronger."
    ]
    for c in caveats:
        story.append(para("- " + c, styles["Bodyx"]))

    doc.build(story)

# -----------------------------
# Main workflow
# -----------------------------
def _comparison_row_value(comparison_table: pd.DataFrame, comparison: str, col: str) -> float:
    rows = comparison_table.loc[comparison_table["comparison"].eq(comparison)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][col])


def run_nested_variant_full_coordinate(panel: pd.DataFrame, spec: NestedVariant) -> Dict[str, object]:
    """Run the production full-coordinate arithmetic nested comparison."""
    variant_dir = OUTDIR / spec.key
    variant_table_dir = TABLE_DIR / spec.key
    variant_chart_dir = CHART_DIR / spec.key
    for d in [variant_dir, variant_table_dir, variant_chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    score_data = build_lp_scores(panel, H=24, L=12, outcome_columns=spec.outcome_columns)
    Q = np.asarray(score_data["Q"], dtype=float)
    dates = pd.Series(score_data["dates"]).reset_index(drop=True)
    N, M = Q.shape
    H = int(score_data["H"])
    pvars = int(score_data["pvars"])
    expected_dim = (H + 1) * pvars
    if M != expected_dim:
        raise RuntimeError(f"LP score dimension {M} does not match full horizon-outcome grid {expected_dim}.")
    if spec.key == "base5_headline" and H == 24 and pvars == 5 and M != 125:
        raise RuntimeError(f"Production headline nested comparison expected p=125, got p={M}.")

    config = full_nested.config_from_env(
        bootstrap_draws=NESTED_BOOTSTRAP_DRAWS,
        bootstrap_block_len=NESTED_BOOTSTRAP_BLOCK_LEN,
    )
    result = full_nested.run_full_coordinate_nested(Q, dates, config, include_sensitivity=True)
    full_paths = full_nested.write_full_coordinate_outputs(result, variant_table_dir)

    model_summary = result.model_summary
    comparison_table = result.comparisons
    covariance_diagnostics = result.covariance_diagnostics
    score_decomposition = result.score_decomposition
    pair_decomposition = result.pair_decomposition
    pair_summary = result.pair_summary
    influential_dates = result.influential_dates
    alternative_scores = result.alternative_scores
    structural_diagnostics = result.structural_diagnostics
    covariance_sensitivity = result.covariance_sensitivity
    eval_score_df = result.model_scores
    mean_within = result.mean_within
    m1m0_per_coordinate = _comparison_row_value(
        comparison_table,
        "M1 - M0",
        "avg_joint_log_score_diff_per_dimension",
    )
    min_cov_eig = float(covariance_diagnostics["min_predicted_cov_eigenvalue"].min())
    min_cov_floor = float(covariance_diagnostics["covariance_floor"].min())
    jittered_periods = int(covariance_diagnostics["jittered_periods"].sum())
    final_training_residuals = int(covariance_diagnostics["training_residual_count"].min())

    sample_table = pd.DataFrame(
        {
            "item": [
                "LP score observations",
                "Score-surface dimension",
                "Usable sample range",
                "Reference geometry observations",
                "Final score covariance/regularization training observations",
                "Validation observations used for half-life selection",
                "Evaluation observations",
                "Evaluation range",
                "Horizons",
                "Outcomes",
                "Evaluation basis dimension",
                "Mean model",
                "Covariance model",
                "Selected mean half-life",
                "Selected covariance half-life",
                "Soft effective dimension d_rho",
                "Displayed score-difference unit",
                "M1-M0 average joint score difference per dimension",
                "Minimum covariance floor across M0-M3",
                "Minimum predicted covariance eigenvalue across M0-M3",
                "Cholesky jittered periods across M0-M3",
                "Covariance estimator",
                "Predictive criterion",
                "Bootstrap for score-difference CIs",
            ],
            "value": [
                N,
                M,
                f"{dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}",
                result.eval_start,
                final_training_residuals,
                result.eval_start - result.validation_start,
                N - result.eval_start,
                f"{dates.iloc[result.eval_start].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}",
                f"0 to {H} months",
                ", ".join(score_data["outcome_labels"]),
                "full coordinate p; no reduced basis",
                "full-coordinate common-persistence recursive moving center",
                "causal arithmetic outer-product covariance recursion",
                "inf" if np.isinf(result.tuning.selected_mean_half_life) else result.tuning.selected_mean_half_life,
                "inf" if np.isinf(result.tuning.selected_cov_half_life) else result.tuning.selected_cov_half_life,
                result.geometry.d_rho,
                "avg_log_score_diff is an average per evaluation date; sum_log_score_diff is reported separately",
                m1m0_per_coordinate,
                min_cov_floor,
                min_cov_eig,
                jittered_periods,
                config.covariance_estimator,
                "Original-coordinate one-step Gaussian log score after full-rank ridge whitening and common Jacobian",
                f"Paired circular moving block, {NESTED_BOOTSTRAP_BLOCK_LEN}-month blocks, {NESTED_BOOTSTRAP_DRAWS} draws",
            ],
        }
    )
    sample_table.to_csv(variant_table_dir / "sample_table.csv", index=False)
    model_summary.to_csv(variant_table_dir / "model_summary.csv", index=False)
    comparison_table.to_csv(variant_table_dir / "model_comparisons_block_bootstrap.csv", index=False)
    covariance_diagnostics.to_csv(variant_table_dir / "covariance_numerical_diagnostics.csv", index=False)
    score_decomposition.to_csv(variant_table_dir / "gaussian_score_decomposition.csv", index=False)
    pair_decomposition.to_csv(variant_table_dir / "model_pair_score_decomposition.csv", index=False)
    pair_summary.to_csv(variant_table_dir / "model_pair_score_summary.csv", index=False)
    influential_dates.to_csv(variant_table_dir / "influential_evaluation_dates.csv", index=False)
    alternative_scores.to_csv(variant_table_dir / "alternative_score_diagnostics.csv", index=False)
    structural_diagnostics.to_csv(variant_table_dir / "structural_subspace_diagnostics.csv", index=False)
    covariance_sensitivity.to_csv(variant_table_dir / "covariance_regularization_sensitivity.csv", index=False)

    top_n = min(12, len(mean_within))
    top_total_idx = np.argsort(mean_within["tau_total"].to_numpy(float))[::-1][:top_n]
    top_within = set(np.argsort(mean_within["tau_within"].to_numpy(float))[::-1][:top_n])
    top_spikes = pd.DataFrame(
        {
            "date": mean_within.iloc[top_total_idx]["date"].to_numpy(),
            "rank_total_tau": np.arange(1, top_n + 1),
            "tau_total": mean_within.iloc[top_total_idx]["tau_total"].to_numpy(float),
            "tau_mean": mean_within.iloc[top_total_idx]["tau_mean"].to_numpy(float),
            "tau_within": mean_within.iloc[top_total_idx]["tau_within"].to_numpy(float),
            "within_share": mean_within.iloc[top_total_idx]["tau_within"].to_numpy(float)
            / np.maximum(mean_within.iloc[top_total_idx]["tau_total"].to_numpy(float), 1e-12),
            "in_top_within_tau": [int(idx) in top_within for idx in top_total_idx],
        }
    )
    top_spikes.to_csv(variant_table_dir / "top_A_t_spikes_M1_with_M3_adjustment.csv", index=False)

    survival_summary = pd.DataFrame(
        {
            "quantity": [
                "Mean tau_total",
                "Mean tau_mean",
                "Mean tau_within",
                "Max tau_total",
                "Max tau_within",
                "Date of max full-coordinate total tau",
                "Date of max full-coordinate within tau",
                "Max absolute tau decomposition error",
                "Mean smoothing ESS",
                "Max local moving-center norm",
            ],
            "value": [
                float(mean_within["tau_total"].mean()),
                float(mean_within["tau_mean"].mean()),
                float(mean_within["tau_within"].mean()),
                float(mean_within["tau_total"].max()),
                float(mean_within["tau_within"].max()),
                mean_within.loc[mean_within["tau_total"].idxmax(), "date"],
                mean_within.loc[mean_within["tau_within"].idxmax(), "date"],
                float(np.max(np.abs(mean_within["tau_decomposition_error"].to_numpy(float)))),
                float(mean_within["smoothing_ess"].mean()),
                float(mean_within["local_mean_norm"].max()),
            ],
        }
    )
    survival_summary.to_csv(variant_table_dir / "A_t_survival_summary.csv", index=False)

    eval_score_df.to_csv(variant_table_dir / "oos_log_scores_by_month.csv", index=False)
    gamma_df = mean_within[["date", "local_mean_norm", "tau_mean", "tau_within", "tau_total"]].copy()
    gamma_df["mean_drift_norm"] = gamma_df["local_mean_norm"]
    gamma_df.to_csv(variant_table_dir / "time_varying_mean_gamma_path_full_sample.csv", index=False)
    A_df = mean_within[
        ["date", "tau_total", "tau_mean", "tau_within", "local_mean_norm", "tau_decomposition_error"]
    ].copy()
    A_df.to_csv(variant_table_dir / "A_t_survival_paths.csv", index=False)

    score_dict = {m: eval_score_df[f"log_score_{m}"].to_numpy(float) for m in ["M0", "M1", "M2", "M3"]}
    eval_dates = pd.to_datetime(eval_score_df["date"])
    charts = {}
    charts["logscore_bars"] = variant_chart_dir / "01_oos_logscore_bars.png"
    plot_logscore_bars(model_summary, charts["logscore_bars"])
    charts["cumulative"] = variant_chart_dir / "02_cumulative_logscore_differences.png"
    plot_logscore_diff_cumulative(eval_dates, score_dict, charts["cumulative"])
    charts["gamma"] = variant_chart_dir / "03_full_coordinate_mean_within_decomposition.png"
    plot_full_coordinate_mean_within(mean_within, charts["gamma"])
    charts["survival"] = variant_chart_dir / "04_full_coordinate_total_within_tau.png"
    plot_full_coordinate_mean_within(mean_within, charts["survival"])
    charts["mean_cov"] = variant_chart_dir / "05_moving_center_vs_within_covariance.png"
    plot_full_coordinate_mean_vs_within(mean_within, charts["mean_cov"])

    def comp_value(comp: str):
        return comparison_table.loc[comparison_table["comparison"] == comp].iloc[0]

    m1m0 = comp_value("M1 - M0")
    m2m0 = comp_value("M2 - M0")
    m3m1 = comp_value("M3 - M1")
    m3m2 = comp_value("M3 - M2")
    m1m0_summary = pair_summary.loc[pair_summary["comparison"].eq("M1 - M0")].iloc[0]
    sensitivity_sign_changes = int(covariance_sensitivity["sign_differs_from_primary"].sum()) if not covariance_sensitivity.empty else 0
    econ = (
        f"Mean variation test. M2-M0 has an average full-coordinate log-score difference of {m2m0['avg_log_score_diff']:.3f} "
        f"with a paired moving-block 90 percent interval [{m2m0['p05']:.3f}, {m2m0['p95']:.3f}]. "
        "This tests whether a common-persistence moving center in all LP response-score coordinates improves prediction relative to the fixed-center benchmark.\n\n"
        f"Covariance variation test. M1-M0 has an average full-coordinate log-score difference of {m1m0['avg_log_score_diff']:.3f} "
        f"with interval [{m1m0['p05']:.3f}, {m1m0['p95']:.3f}]. "
        "This is the causal arithmetic covariance-recursion test, scored in the same complete coordinate system. "
        f"Scale audit: this contrast is an average per evaluation date, not a sum; divided by {M} score coordinates it is {m1m0_per_coordinate:.3f}. "
        f"The minimum covariance floor is {min_cov_floor:.6g}, the minimum predicted covariance eigenvalue is {min_cov_eig:.6g}, and Cholesky jitter is triggered in {jittered_periods} model-date cells. "
        f"Influence audit: the largest date contributes {m1m0_summary['top1_abs_date_fraction_of_total']:.3f} of the cumulative M1-M0 difference and the top five dates contribute {m1m0_summary['top5_abs_dates_fraction_of_total']:.3f}. "
        f"The mean Mahalanobis-per-dimension is {m1m0_summary['mean_right_mahalanobis_per_dimension']:.3f} for M0 versus {m1m0_summary['mean_left_mahalanobis_per_dimension']:.3f} for M1, so the corrected score advantage is best read as a severe fixed-covariance underdispersion diagnostic, not an unconditional structural ranking.\n\n"
        f"Joint model test. M3-M1 is {m3m1['avg_log_score_diff']:.3f} on average, while M3-M2 is {m3m2['avg_log_score_diff']:.3f}. "
        "The primary M3 result uses the separately selected M1 covariance and M2 mean half-lives, not a jointly retuned final-evaluation fit. "
        f"The sensitivity table reports {sensitivity_sign_changes} sign changes relative to the primary Ledoit-Wolf setting across all displayed contrasts/settings; any sign change should be read as instability for that particular contrast rather than as a headline result.\n\n"
        "Mean-versus-within audit. The separate descriptive decomposition writes tau_total, tau_mean, and tau_within using the same full-coordinate soft geometry. "
        "It is not used by the causal M0-M3 predictive filters."
    )

    metadata = {
        **result.metadata,
        "variant": spec.key,
        "variant_label": spec.label,
        "outcome_columns": list(spec.outcome_columns),
        "outcome_labels": list(score_data["outcome_labels"]),
        "panel_path": str(PANEL_PATH),
        "N": int(N),
        "M": int(M),
        "H": int(H),
        "pvars": int(pvars),
        "theta_hat_is_training_mean_of_raw_Q": True,
        "full_coordinate_artifacts": {k: str(v) for k, v in full_paths.items()},
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "spec": spec,
        "score_data": score_data,
        "sample_table": sample_table,
        "model_summary": model_summary,
        "comparison_table": comparison_table,
        "survival_summary": survival_summary,
        "covariance_diagnostics": covariance_diagnostics,
        "score_decomposition": score_decomposition,
        "pair_decomposition": pair_decomposition,
        "pair_summary": pair_summary,
        "influential_dates": influential_dates,
        "alternative_scores": alternative_scores,
        "structural_diagnostics": structural_diagnostics,
        "covariance_sensitivity": covariance_sensitivity,
        "top_spikes": top_spikes,
        "eval_scores": eval_score_df,
        "gamma_path": gamma_df,
        "survival_paths": A_df,
        "charts": charts,
        "economic_interpretation": econ,
        "metadata": metadata,
        "table_dir": variant_table_dir,
        "chart_dir": variant_chart_dir,
    }


def run_nested_variant(panel: pd.DataFrame, spec: NestedVariant) -> Dict[str, object]:
    if NESTED_MODE in {"legacy-reduced", "legacy", "reduced"}:
        legacy_root = OUTDIR / "legacy_reduced"
        legacy_root.mkdir(parents=True, exist_ok=True)
        return run_nested_variant_legacy(panel, spec)
    if NESTED_MODE not in {"full-coordinate-arithmetic", "full_coordinate_arithmetic", "full"}:
        raise ValueError("OVK_NESTED_MODE must be full-coordinate-arithmetic or legacy-reduced.")
    return run_nested_variant_full_coordinate(panel, spec)


def run_nested_variant_legacy(panel: pd.DataFrame, spec: NestedVariant) -> Dict[str, object]:
    variant_dir = OUTDIR / spec.key
    variant_table_dir = TABLE_DIR / spec.key
    variant_chart_dir = CHART_DIR / spec.key
    for d in [variant_dir, variant_table_dir, variant_chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    score_data = build_lp_scores(panel, H=24, L=12, outcome_columns=spec.outcome_columns)
    Q = score_data["Q"]
    dates = score_data["dates"]
    N, M = Q.shape
    H = score_data["H"]
    pvars = score_data["pvars"]

    # Pre-evaluation split: leave enough pre-GFC history to initialize/tune, evaluate from 2006 onward.
    train_end = 180
    val_start = 100
    if N <= train_end + 24:
        raise RuntimeError("Sample too short for chosen train/eval split.")

    eig_train = covariance_eigenbasis(Q, train_end=train_end)
    beta_train = eig_train["beta"]
    V_train = eig_train["V"]
    evals_train = eig_train["evals"]
    bases = build_mean_and_eval_basis(beta_train, V_train, d_eval=10)
    W = bases["W"]
    Bbeta = bases["Bbeta"]
    Bc = W.T @ Bbeta
    y = Q @ W
    d_eval = y.shape[1]

    # Tune and evaluate models.
    params = {}
    outputs = {}
    score_dict = {}
    models = ["M0", "M1", "M2", "M3"]
    for model in models:
        params[model] = tune_model(y, Bc, model, train_end=train_end, val_start=val_start)
        outputs[model] = run_predictive_scores(y, Bc, model, train_end, N, params[model])
        score_dict[model] = outputs[model]["log_scores"]

    eval_dates = dates.iloc[train_end:N].reset_index(drop=True)

    model_rows = []
    for m in models:
        p = params[m]
        model_rows.append({
            "model": m,
            "avg_log_score": float(np.mean(score_dict[m])),
            "sum_log_score": float(np.sum(score_dict[m])),
            "kmean": p.kmean,
            "kcov": p.kcov,
            "phi": p.phi,
            "cov_shrink": p.cov_shrink,
            "cov_target_weight": p.cov_target_weight,
        })
    model_summary = pd.DataFrame(model_rows)

    comparison_pairs = [
        ("M1", "M0", "Dynamic covariance vs fixed benchmark"),
        ("M2", "M0", "Dynamic mean vs fixed benchmark"),
        ("M3", "M0", "Joint mean-covariance vs fixed benchmark"),
        ("M3", "M1", "Does joint model beat dynamic covariance only?"),
        ("M3", "M2", "Does joint model beat dynamic mean only?"),
        ("M2", "M1", "Mean-only vs covariance-only"),
    ]
    comp_rows = []
    for a, b, desc in comparison_pairs:
        diff = score_dict[a] - score_dict[b]
        mean, lo, hi, prob = block_bootstrap_ci(
            diff,
            block_len=NESTED_BOOTSTRAP_BLOCK_LEN,
            B=NESTED_BOOTSTRAP_DRAWS,
            seed=1234,
        )
        comp_rows.append({
            "comparison": f"{a} - {b}",
            "meaning": desc,
            "avg_log_score_diff": mean,
            "p05": lo,
            "p95": hi,
            "prob_diff_gt_0": prob,
            "sum_log_score_diff": float(np.sum(diff)),
        })
    comparison_table = pd.DataFrame(comp_rows)

    # Full-sample survival diagnostic using full-sample covariance basis.
    eig_full = covariance_eigenbasis(Q, train_end=None)
    beta_full = eig_full["beta"]
    V_full = eig_full["V"]
    evals_full = eig_full["evals"]
    bases_full = build_mean_and_eval_basis(beta_full, V_full, d_eval=10)
    Bbeta_full = bases_full["Bbeta"]
    W_full = bases_full["W"]
    Bc_full = W_full.T @ Bbeta_full
    y_full = Q @ W_full
    mu_full = y_full.mean(axis=0)

    kmean_m3 = params["M3"].kmean if params["M3"].kmean > 0 else params["M2"].kmean
    gamma_full = filtered_gamma_path(y_full - mu_full, Bc_full, kmean=kmean_m3, phi=params["M3"].phi)
    mean_drift_full_y = gamma_full @ Bc_full.T  # N x d coordinates in W space
    # Convert mean drift back to original score-surface coordinates: W @ drift_y.
    mean_drift_full_q = mean_drift_full_y @ W_full.T

    R_cov = HEADLINE_R
    V_R = V_full[:, :R_cov]
    lam_R = evals_full[:R_cov]
    lam_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(lam_R, 1e-12)))
    z_m1 = (Q - beta_full) @ V_R @ lam_inv_sqrt
    z_m3 = (Q - beta_full - mean_drift_full_q) @ V_R @ lam_inv_sqrt
    upgraded_m1 = upgraded_state_space_A_from_z(z_m1, R_cov)
    upgraded_m3 = upgraded_state_space_A_from_z(z_m3, R_cov)
    A_m1 = upgraded_m1["A"]
    A_m3 = upgraded_m3["A"]
    trace_m1 = upgraded_m1["tau"]
    trace_m3 = upgraded_m3["tau"]
    corr_trace = float(np.corrcoef(trace_m1, trace_m3)[0, 1])
    top_n = 12
    top_idx_m1 = np.argsort(trace_m1)[::-1][:top_n]
    top_idx_m3 = np.argsort(trace_m3)[::-1][:top_n]
    overlap = len(set(top_idx_m1).intersection(set(top_idx_m3)))
    top_spikes = pd.DataFrame({
        "date": dates.iloc[top_idx_m1].dt.strftime("%Y-%m-%d").values,
        "rank_M1": np.arange(1, top_n + 1),
        "traceA_M1": trace_m1[top_idx_m1],
        "traceA_M3_after_mean_adjustment": trace_m3[top_idx_m1],
        "retention_ratio_M3_over_M1": trace_m3[top_idx_m1] / trace_m1[top_idx_m1],
        "in_M3_top12": [idx in set(top_idx_m3) for idx in top_idx_m1],
    })
    survival_summary = pd.DataFrame({
        "quantity": [
            "Correlation of trace(A_t) paths, M1 vs M3",
            "Top-12 spike overlap count",
            "Top-12 spike overlap share",
            "Mean trace(A_t), M1",
            "Mean trace(A_t), M3",
            "SD trace(A_t), M1",
            "SD trace(A_t), M3",
            "Max trace(A_t), M1",
            "Max trace(A_t), M3",
            "Date of max M1 amplification",
            "Date of max M3 amplification",
        ],
        "value": [
            corr_trace,
            overlap,
            overlap / top_n,
            trace_m1.mean(),
            trace_m3.mean(),
            trace_m1.std(ddof=0),
            trace_m3.std(ddof=0),
            trace_m1.max(),
            trace_m3.max(),
            dates.iloc[int(np.argmax(trace_m1))].strftime("%Y-%m-%d"),
            dates.iloc[int(np.argmax(trace_m3))].strftime("%Y-%m-%d"),
        ]
    })

    # Mean drift norm vs covariance amplification.
    mean_norm = np.linalg.norm(mean_drift_full_q, axis=1)

    # Save tables.
    sample_table = pd.DataFrame({
        "item": [
            "LP score observations", "Score-surface dimension", "Usable sample range",
            "Pre-evaluation training observations", "Evaluation observations", "Evaluation range",
            "Horizons", "Outcomes", "Evaluation basis dimension", "Mean basis dimension",
            "A_t survival estimator", "A_t survival rank", "Selected alpha, M1", "Selected alpha, M3",
            "Predictive criterion", "Bootstrap for score-difference CIs"
        ],
        "value": [
            N, M,
            f"{dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}",
            train_end, N - train_end,
            f"{dates.iloc[train_end].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}",
            "0 to 24 months", ", ".join(score_data["outcome_labels"]),
            d_eval, Bbeta.shape[1],
            "Legacy audit reduced-rank log-SPD state-space core",
            R_cov,
            upgraded_m1["fit"].alpha,
            upgraded_m3["fit"].alpha,
            "Legacy audit one-step Gaussian quasi-log score in reduced coordinates",
            f"Circular moving block, {NESTED_BOOTSTRAP_BLOCK_LEN}-month blocks, {NESTED_BOOTSTRAP_DRAWS} draws",
        ]
    })
    sample_table.to_csv(variant_table_dir / "sample_table.csv", index=False)
    model_summary.to_csv(variant_table_dir / "model_summary.csv", index=False)
    comparison_table.to_csv(variant_table_dir / "model_comparisons_block_bootstrap.csv", index=False)
    survival_summary.to_csv(variant_table_dir / "A_t_survival_summary.csv", index=False)
    top_spikes.to_csv(variant_table_dir / "top_A_t_spikes_M1_with_M3_adjustment.csv", index=False)

    # Full paths for states/scores.
    eval_score_df = pd.DataFrame({"date": eval_dates.dt.strftime("%Y-%m-%d")})
    for m in models:
        eval_score_df[f"log_score_{m}"] = score_dict[m]
    eval_score_df.to_csv(variant_table_dir / "oos_log_scores_by_month.csv", index=False)
    gamma_df = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")})
    for j in range(gamma_full.shape[1]):
        gamma_df[f"gamma_{j+1}"] = gamma_full[:, j]
    gamma_df["mean_drift_norm"] = mean_norm
    gamma_df.to_csv(variant_table_dir / "time_varying_mean_gamma_path_full_sample.csv", index=False)
    A_df = pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d"),
        "traceA_M1_fixed_mean": trace_m1,
        "traceA_M3_after_mean_adjustment": trace_m3,
        "mean_drift_norm": mean_norm,
    })
    A_df.to_csv(variant_table_dir / "A_t_survival_paths.csv", index=False)

    # Charts.
    charts = {}
    charts["logscore_bars"] = variant_chart_dir / "01_oos_logscore_bars.png"
    plot_logscore_bars(model_summary, charts["logscore_bars"])
    charts["cumulative"] = variant_chart_dir / "02_cumulative_logscore_differences.png"
    plot_logscore_diff_cumulative(eval_dates, score_dict, charts["cumulative"])
    charts["gamma"] = variant_chart_dir / "03_gamma_path_full_sample.png"
    plot_gamma_path(dates, gamma_full, charts["gamma"])
    charts["survival"] = variant_chart_dir / "04_A_t_survival_M1_vs_M3.png"
    plot_A_survival(dates, trace_m1, trace_m3, charts["survival"])
    charts["mean_cov"] = variant_chart_dir / "05_mean_drift_vs_cov_amplification.png"
    plot_mean_vs_cov_norm(dates, mean_norm, trace_m3, charts["mean_cov"])

    # Economic interpretation strings based on results.
    def comp_value(comp: str):
        row = comparison_table.loc[comparison_table["comparison"] == comp].iloc[0]
        return row
    m1m0 = comp_value("M1 - M0")
    m2m0 = comp_value("M2 - M0")
    m3m1 = comp_value("M3 - M1")
    m3m2 = comp_value("M3 - M2")
    m3m0 = comp_value("M3 - M0")
    # interpret signs with cautious labels.
    def evidence_phrase(row):
        if row["p05"] > 0:
            return "strong positive"
        if row["avg_log_score_diff"] > 0 and row["prob_diff_gt_0"] > 0.75:
            return "positive but statistically fragile"
        if row["avg_log_score_diff"] < 0 and row["p95"] < 0:
            return "strong negative"
        if row["avg_log_score_diff"] < 0:
            return "negative or weak"
        return "ambiguous"
    econ = (
        f"Mean variation test. M2-M0 has an average log-score difference of {m2m0['avg_log_score_diff']:.3f} "
        f"with a 90 percent block-bootstrap interval [{m2m0['p05']:.3f}, {m2m0['p95']:.3f}]. "
        f"This is {evidence_phrase(m2m0)} evidence on whether a moving IRF center improves predictive fit relative to the standard fixed average LP model. "
        "Economically, this comparison asks whether the response surface itself shifts persistently through time, rather than merely becoming more or less uncertain."
        "\n\n"
        f"Uncertainty variation test. M1-M0 has an average log-score difference of {m1m0['avg_log_score_diff']:.3f} "
        f"with a 90 percent interval [{m1m0['p05']:.3f}, {m1m0['p95']:.3f}]. "
        f"This is {evidence_phrase(m1m0)} evidence for time-varying covariance around a fixed average IRF. "
        "This is the dynamic response-score covariance channel: the center of the LP response need not move, but the covariance envelope around it can widen, shrink, and rotate across horizon-variable co-movement directions."
        "\n\n"
        f"Joint model test. M3-M1 is {m3m1['avg_log_score_diff']:.3f} on average, with interval [{m3m1['p05']:.3f}, {m3m1['p95']:.3f}], while M3-M2 is {m3m2['avg_log_score_diff']:.3f} with interval [{m3m2['p05']:.3f}, {m3m2['p95']:.3f}]. "
        "These comparisons ask whether the data prefer both a moving center and a moving covariance envelope. "
        "If the joint model beats both single-channel alternatives, it is evidence that monetary-policy transmission is changing in two ways: average response shapes move and precision around those shapes also changes."
        "\n\n"
        f"A_t survival. The correlation between the M1 amplification path and the M3 residual amplification path is {corr_trace:.3f}. "
        f"The top-12 spike overlap is {overlap} out of {top_n}. "
        "This diagnostic directly addresses the concern that covariance-amplification spikes are just unmodeled moving LP-response centers. "
        "High correlation and high top-spike overlap mean that the covariance-amplification episodes survive after subtracting the estimated moving-center component. Low overlap would imply that the original A_t path was partly a proxy for mean drift."
        "\n\n"
        "Economic reading. A moving IRF center corresponds to state-dependent mean transmission: the expected response to a monetary-policy shock changes. "
        "A dynamic response-score covariance corresponds to state-dependent detectability or precision: the average response is more or less tightly pinned down in particular response-shape directions. "
        "The nested workflow therefore separates two claims that are often conflated in rolling LP work: whether the impulse response changed, and whether uncertainty around the average response changed."
    )

    # Save metadata/results JSON.
    metadata = {
        "variant": spec.key,
        "variant_label": spec.label,
        "outcome_columns": list(spec.outcome_columns),
        "outcome_labels": list(score_data["outcome_labels"]),
        "panel_path": str(PANEL_PATH),
        "N": int(N),
        "M": int(M),
        "train_end": int(train_end),
        "val_start": int(val_start),
        "eval_start_date": dates.iloc[train_end].strftime("%Y-%m-%d"),
        "eval_end_date": dates.iloc[-1].strftime("%Y-%m-%d"),
        "params": {k: vars(v) for k, v in params.items()},
        "upgraded_state_space_core": {
            "enabled_for_A_t_survival": True,
            "rank": int(R_cov),
            "alpha_grid": ALPHA_GRID.tolist(),
            "student_t_degrees_of_freedom": ROBUST_NU,
            "M1_alpha_hat": upgraded_m1["fit"].alpha,
            "M3_alpha_hat": upgraded_m3["fit"].alpha,
            "M1_spectral_radius": upgraded_m1["fit"].spectral_radius,
            "M3_spectral_radius": upgraded_m3["fit"].spectral_radius,
        },
        "survival_corr_trace": corr_trace,
        "top12_overlap": int(overlap),
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "spec": spec,
        "score_data": score_data,
        "sample_table": sample_table,
        "model_summary": model_summary,
        "comparison_table": comparison_table,
        "survival_summary": survival_summary,
        "top_spikes": top_spikes,
        "eval_scores": eval_score_df,
        "gamma_path": gamma_df,
        "survival_paths": A_df,
        "charts": charts,
        "economic_interpretation": econ,
        "metadata": metadata,
        "table_dir": variant_table_dir,
        "chart_dir": variant_chart_dir,
    }


def _variant_prefixed(result: Dict[str, object], table_key: str) -> pd.DataFrame:
    spec: NestedVariant = result["spec"]  # type: ignore[assignment]
    score_data: dict = result["score_data"]  # type: ignore[assignment]
    df = result[table_key].copy()  # type: ignore[index, union-attr]
    df.insert(0, "outcomes", ", ".join(score_data["outcome_labels"]))
    df.insert(0, "outcome_count", len(score_data["outcome_labels"]))
    df.insert(0, "variant_label", spec.label)
    df.insert(0, "variant", spec.key)
    return df


def _comparison_row(comparison_table: pd.DataFrame, comparison: str) -> pd.Series:
    rows = comparison_table.loc[comparison_table["comparison"] == comparison]
    if rows.empty:
        raise KeyError(f"Comparison not found: {comparison}")
    return rows.iloc[0]


def write_combined_variant_outputs(results: list[Dict[str, object]]) -> pd.DataFrame:
    table_map = {
        "sample_table_by_variant.csv": "sample_table",
        "model_summary_by_variant.csv": "model_summary",
        "model_comparisons_block_bootstrap_by_variant.csv": "comparison_table",
        "covariance_numerical_diagnostics_by_variant.csv": "covariance_diagnostics",
        "gaussian_score_decomposition_by_variant.csv": "score_decomposition",
        "model_pair_score_decomposition_by_variant.csv": "pair_decomposition",
        "model_pair_score_summary_by_variant.csv": "pair_summary",
        "influential_evaluation_dates_by_variant.csv": "influential_dates",
        "alternative_score_diagnostics_by_variant.csv": "alternative_scores",
        "structural_subspace_diagnostics_by_variant.csv": "structural_diagnostics",
        "covariance_regularization_sensitivity_by_variant.csv": "covariance_sensitivity",
        "A_t_survival_summary_by_variant.csv": "survival_summary",
        "top_A_t_spikes_M1_with_M3_adjustment_by_variant.csv": "top_spikes",
        "oos_log_scores_by_month_by_variant.csv": "eval_scores",
        "time_varying_mean_gamma_path_full_sample_by_variant.csv": "gamma_path",
        "A_t_survival_paths_by_variant.csv": "survival_paths",
    }
    for filename, table_key in table_map.items():
        combined = pd.concat([_variant_prefixed(res, table_key) for res in results], ignore_index=True)
        combined.to_csv(TABLE_DIR / filename, index=False)

    rows = []
    for res in results:
        spec: NestedVariant = res["spec"]  # type: ignore[assignment]
        score_data: dict = res["score_data"]  # type: ignore[assignment]
        metadata: dict = res["metadata"]  # type: ignore[assignment]
        comparisons: pd.DataFrame = res["comparison_table"]  # type: ignore[assignment]
        survival: pd.DataFrame = res["survival_summary"]  # type: ignore[assignment]
        dates = pd.to_datetime(score_data["dates"])
        row = {
            "variant": spec.key,
            "variant_label": spec.label,
            "outcome_count": len(score_data["outcome_labels"]),
            "outcomes": ", ".join(score_data["outcome_labels"]),
            "n_valid": int(metadata["N"]),
            "score_surface_dimension": int(metadata["M"]),
            "sample_start": dates.iloc[0].strftime("%Y-%m"),
            "sample_end": dates.iloc[-1].strftime("%Y-%m"),
            "eval_start": metadata["eval_start_date"],
            "eval_end": metadata["eval_end_date"],
        }
        for comp_name, out_col in [
            ("M1 - M0", "M1_minus_M0"),
            ("M2 - M0", "M2_minus_M0"),
            ("M3 - M1", "M3_minus_M1"),
            ("M3 - M2", "M3_minus_M2"),
        ]:
            comp_row = _comparison_row(comparisons, comp_name)
            row[out_col] = float(comp_row["avg_log_score_diff"])
            if "avg_joint_log_score_diff_per_dimension" in comp_row:
                row[f"{out_col}_per_dimension"] = float(comp_row["avg_joint_log_score_diff_per_dimension"])
        covariance_diagnostics: pd.DataFrame = res.get("covariance_diagnostics", pd.DataFrame())  # type: ignore[assignment]
        if not covariance_diagnostics.empty:
            row["min_covariance_floor"] = float(covariance_diagnostics["covariance_floor"].min())
            row["min_predicted_cov_eigenvalue"] = float(covariance_diagnostics["min_predicted_cov_eigenvalue"].min())
            row["jittered_periods_total"] = int(covariance_diagnostics["jittered_periods"].sum())
        row["survival_corr_trace_M1_M3"] = float(metadata.get("survival_corr_trace", np.nan))
        row["top12_overlap"] = int(metadata.get("top12_overlap", 0))
        max_m1 = survival.loc[survival["quantity"].eq("Date of max M1 amplification"), "value"]
        max_full = survival.loc[survival["quantity"].eq("Date of max full-coordinate total tau"), "value"]
        if len(max_m1):
            row["date_of_max_M1_amplification"] = max_m1.iloc[0]
        elif len(max_full):
            row["date_of_max_M1_amplification"] = max_full.iloc[0]
        else:
            row["date_of_max_M1_amplification"] = ""
        rows.append(row)
    overview = pd.DataFrame(rows)
    overview.to_csv(TABLE_DIR / "nested_mean_covariance_variant_summary.csv", index=False)
    return overview


def copy_legacy_headline_outputs(result: Dict[str, object]) -> None:
    """Keep old root-level nested filenames pointed at the headline five-outcome run."""
    table_dir: Path = result["table_dir"]  # type: ignore[assignment]
    for name in [
        "sample_table.csv",
        "model_summary.csv",
        "model_comparisons_block_bootstrap.csv",
        "covariance_numerical_diagnostics.csv",
        "gaussian_score_decomposition.csv",
        "model_pair_score_decomposition.csv",
        "model_pair_score_summary.csv",
        "influential_evaluation_dates.csv",
        "alternative_score_diagnostics.csv",
        "structural_subspace_diagnostics.csv",
        "covariance_regularization_sensitivity.csv",
        "nested_full_coordinate_comparisons.csv",
        "nested_full_coordinate_covariance_diagnostics.csv",
        "nested_full_coordinate_gaussian_score_decomposition.csv",
        "nested_full_coordinate_model_pair_score_decomposition.csv",
        "nested_full_coordinate_model_pair_score_summary.csv",
        "nested_full_coordinate_influential_evaluation_dates.csv",
        "nested_full_coordinate_alternative_score_diagnostics.csv",
        "nested_full_coordinate_structural_subspace_diagnostics.csv",
        "nested_full_coordinate_covariance_sensitivity.csv",
        "A_t_survival_summary.csv",
        "top_A_t_spikes_M1_with_M3_adjustment.csv",
        "oos_log_scores_by_month.csv",
        "time_varying_mean_gamma_path_full_sample.csv",
        "A_t_survival_paths.csv",
    ]:
        src = table_dir / name
        if src.exists():
            shutil.copy2(src, TABLE_DIR / name)
    charts: dict[str, Path] = result["charts"]  # type: ignore[assignment]
    for path in charts.values():
        shutil.copy2(path, CHART_DIR / path.name)


def build_multi_variant_pdf_report(pdf_path: Path, results: list[Dict[str, object]], overview: pd.DataFrame) -> None:
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.2, leading=12))

    story = []
    story.append(para(REPORT_TITLE, styles["Title2"]))
    story.append(para(REPORT_SUBTITLE, styles["Bodyx"]))
    story.append(para(
        "This report runs the nested mean/covariance comparison for each requested outcome surface using the production full-coordinate arithmetic workflow. All M0-M3 predictive scores use the complete LP response-score vector, a common ridge-whitened geometry, a full-coordinate moving center, and a causal arithmetic outer-product covariance recursion.",
        styles["Bodyx"],
    ))
    story.append(para(
        "Log-score contrasts labeled avg_log_score_diff are averages over evaluation dates, not sums. The cumulative sum is reported separately as sum_log_score_diff, and the table also reports avg_joint_log_score_diff_per_dimension to put the Gaussian-score scale on the score-surface dimension.",
        styles["Bodyx"],
    ))
    story.append(para("Variant overview", styles["H1x"]))
    overview_cols = [
        "variant",
        "outcome_count",
        "sample_start",
        "sample_end",
        "M1_minus_M0",
        "M1_minus_M0_per_dimension",
        "M2_minus_M0",
        "M3_minus_M1",
        "M3_minus_M2",
        "min_covariance_floor",
        "min_predicted_cov_eigenvalue",
        "jittered_periods_total",
        "survival_corr_trace_M1_M3",
    ]
    overview_view_cols = [col for col in overview_cols if col in overview.columns]
    story.append(df_to_table(overview[overview_view_cols], max_rows=10, max_cols=len(overview_view_cols)))

    anchor_rows = []
    for res in results:
        spec: NestedVariant = res["spec"]  # type: ignore[assignment]
        score_data: dict = res["score_data"]  # type: ignore[assignment]
        comparisons: pd.DataFrame = res["comparison_table"]  # type: ignore[assignment]
        for comp in ["M1 - M0", "M2 - M0", "M3 - M1", "M3 - M2"]:
            row = _comparison_row(comparisons, comp)
            anchor_rows.append(
                {
                    "variant": spec.key,
                    "outcome_count": len(score_data["outcome_labels"]),
                    "comparison": comp,
                    "score_dimension": row.get("score_dimension", np.nan),
                    "avg_log_score_diff": row["avg_log_score_diff"],
                    "avg_joint_log_score_diff_per_dimension": row.get(
                        "avg_joint_log_score_diff_per_dimension",
                        np.nan,
                    ),
                    "p05": row["p05"],
                    "p95": row["p95"],
                    "prob_diff_gt_0": row["prob_diff_gt_0"],
                }
            )
    story.append(para("Nested evidence anchor", styles["H1x"]))
    story.append(df_to_table(pd.DataFrame(anchor_rows), max_rows=12, max_cols=9))

    for idx, res in enumerate(results):
        spec: NestedVariant = res["spec"]  # type: ignore[assignment]
        score_data: dict = res["score_data"]  # type: ignore[assignment]
        story.append(PageBreak())
        story.append(para(spec.label, styles["H1x"]))
        story.append(para(f"Outcomes: {', '.join(score_data['outcome_labels'])}", styles["Bodyx"]))
        story.append(para("Empirical setup", styles["H2x"]))
        story.append(df_to_table(res["sample_table"], max_rows=20))  # type: ignore[arg-type]
        story.append(para("Predictive log-score comparison", styles["H2x"]))
        story.append(df_to_table(res["model_summary"], max_rows=4))  # type: ignore[arg-type]
        story.append(Spacer(1, 0.04 * inch))
        story.append(Image(str(res["charts"]["logscore_bars"]), width=6.8 * inch, height=3.7 * inch))  # type: ignore[index]
        story.append(df_to_table(res["comparison_table"], max_rows=8))  # type: ignore[arg-type]
        cov_diag = res["covariance_diagnostics"]  # type: ignore[index]
        cov_cols = [
            "model",
            "score_dimension",
            "covariance_estimator",
            "statistical_shrinkage",
            "training_residual_count",
            "training_residual_rank",
            "covariance_floor",
            "min_predicted_cov_eigenvalue",
            "jittered_periods",
        ]
        story.append(para("Covariance numerical diagnostics", styles["H2x"]))
        story.append(df_to_table(cov_diag[cov_cols], max_rows=4, max_cols=len(cov_cols)))  # type: ignore[index]
        story.append(para("Score decomposition summary", styles["H2x"]))
        pair_summary = res["pair_summary"]  # type: ignore[index]
        pair_cols = [
            "comparison",
            "mean",
            "median",
            "top1_abs_date_fraction_of_total",
            "top5_abs_dates_fraction_of_total",
            "pct_dates_with_any_clipped_eigenvalues",
        ]
        story.append(df_to_table(pair_summary[pair_cols], max_rows=6, max_cols=len(pair_cols)))  # type: ignore[index]
        story.append(para("Influential evaluation dates", styles["H2x"]))
        influential = res["influential_dates"]  # type: ignore[index]
        infl_cols = [
            "abs_rank_within_comparison",
            "evaluation_date",
            "comparison",
            "score_diff",
            "logdet_contribution",
            "mahalanobis_contribution",
            "fraction_of_total_difference",
        ]
        story.append(df_to_table(influential[infl_cols], max_rows=12, max_cols=len(infl_cols)))  # type: ignore[index]
        story.append(para("Regularization sensitivity", styles["H2x"]))
        sens = res["covariance_sensitivity"]  # type: ignore[index]
        sens_cols = [
            "setting",
            "comparison",
            "avg_date_level_joint_log_score_diff",
            "median_date_level_diff",
            "p05",
            "p95",
            "proportion_dates_won",
            "sign_differs_from_primary",
        ]
        story.append(df_to_table(sens[sens_cols], max_rows=12, max_cols=len(sens_cols)))  # type: ignore[index]
        story.append(para("Alternative and structural diagnostics", styles["H2x"]))
        alt = res["alternative_scores"]  # type: ignore[index]
        story.append(df_to_table(alt, max_rows=4, max_cols=5))  # type: ignore[arg-type]
        structural = res["structural_diagnostics"]  # type: ignore[index]
        story.append(df_to_table(structural, max_rows=4, max_cols=8))  # type: ignore[arg-type]
        story.append(PageBreak())
        story.append(para("Full-coordinate mean/within diagnostic", styles["H2x"]))
        story.append(df_to_table(res["survival_summary"], max_rows=20))  # type: ignore[arg-type]
        story.append(Spacer(1, 0.04 * inch))
        story.append(Image(str(res["charts"]["survival"]), width=6.8 * inch, height=3.7 * inch))  # type: ignore[index]
        story.append(df_to_table(res["top_spikes"], max_rows=12))  # type: ignore[arg-type]
        if idx == len(results) - 1:
            story.append(para("Implementation caveats", styles["H1x"]))
            caveats = [
                "Each variant builds its own LP score surface and full-rank ridge geometry before nested scoring.",
                "The displayed contrast column avg_log_score_diff is a date average; the sum and per-coordinate average are included to avoid scale ambiguity.",
                "The eight-outcome route has a larger score-surface dimension and includes expectations, so log scores should be read within variant rather than as direct cross-variant likelihood levels.",
                "A positive M1-M0 advantage is evidence for time-varying response-score covariance; it is not an unqualified structural-IRF claim.",
            ]
            for c in caveats:
                story.append(para("- " + c, styles["Bodyx"]))
    doc.build(story)


def main():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Panel not found: {PANEL_PATH}")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    specs = nested_variant_specs(panel)
    print("Running nested mean/covariance variants:", ", ".join(spec.key for spec in specs), flush=True)
    results: list[Dict[str, object]] = []
    for spec in specs:
        print(f"  nested variant {spec.key}", flush=True)
        results.append(run_nested_variant(panel, spec))

    overview = write_combined_variant_outputs(results)
    headline = next((res for res in results if res["spec"].key == "base5_headline"), results[0])  # type: ignore[union-attr]
    copy_legacy_headline_outputs(headline)

    metadata = {
        "panel_path": str(PANEL_PATH),
        "variants": [res["metadata"] for res in results],
        "headline_legacy_variant": headline["spec"].key,  # type: ignore[index, union-attr]
        "score_difference_bootstrap_draws": NESTED_BOOTSTRAP_DRAWS,
        "score_difference_bootstrap_block_length_months": NESTED_BOOTSTRAP_BLOCK_LEN,
    }
    (OUTDIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    try:
        shutil.copy(__file__, CODE_DIR / "ovk_nested_workflow.py")
        helper = Path(__file__).with_name("ovk_data.py")
        if helper.exists():
            shutil.copy(helper, CODE_DIR / "ovk_data.py")
    except Exception:
        pass

    (CODE_DIR / "README_CODE.txt").write_text(
        "Run ovk_nested_workflow.py in an environment with numpy, pandas, matplotlib, and reportlab.\n"
        "Input expected: a pipeline-prepared monthly panel with MP_used, CBI_used, and recognized outcome columns.\n"
        "Outputs: variant-specific tables/charts, combined by-variant tables, PDF report, and bundle zip.\n"
    )

    pdf_path = OUTDIR / "nested_mean_covariance_ovk_report.pdf"
    build_multi_variant_pdf_report(pdf_path, results, overview)

    zip_path = FINAL_ZIP
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in OUTDIR.rglob("*"):
            if f.is_file():
                if f.resolve() == zip_path.resolve():
                    continue
                z.write(f, arcname=str(f.relative_to(OUTDIR)))
    final_pdf = FINAL_PDF
    if pdf_path.resolve() != final_pdf.resolve():
        shutil.copy(pdf_path, final_pdf)

    print("DONE")
    print(f"PDF: {final_pdf}")
    print(f"ZIP: {zip_path}")
    print(overview.to_string(index=False))

if __name__ == "__main__":
    main()
