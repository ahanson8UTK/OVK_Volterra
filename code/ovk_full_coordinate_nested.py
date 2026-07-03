#!/usr/bin/env python3
"""Full-coordinate arithmetic nested predictive comparison.

This module implements the production nested M0/M1/M2/M3 comparison in the
complete LP response-score coordinate system.  It deliberately avoids the
legacy reduced evaluation basis, PCA covariance paths, log-SPD maps, FFBS, and
state-space covariance estimators used by older audit workflows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, eigh, solve_triangular
from sklearn.covariance import LedoitWolf

from run_publication_grade_ovk import (
    ARITHMETIC_REFERENCE_RIDGE_SCALE,
    FULL_COORDINATE_KERNEL_ETA,
    circular_block_indices,
    full_coordinate_K_from_weights,
    full_coordinate_ridge,
    full_coordinate_temporal_weights,
    sym,
)


MODEL_ORDER = ("M0", "M1", "M2", "M3")
COMPARISON_PAIRS = (
    ("M1", "M0", "Dynamic covariance vs fixed benchmark"),
    ("M2", "M0", "Dynamic mean vs fixed benchmark"),
    ("M3", "M0", "Joint mean-covariance vs fixed benchmark"),
    ("M3", "M1", "Does joint model beat dynamic covariance only?"),
    ("M3", "M2", "Does joint model beat dynamic mean only?"),
    ("M2", "M1", "Mean-only vs covariance-only"),
)


@dataclass(frozen=True)
class FullCoordinateNestedConfig:
    """Numerical and split configuration for the full-coordinate comparison."""

    eval_start_index: int = 180
    validation_months: int = 80
    min_initial_observations: int = 60
    ridge_scale: float = ARITHMETIC_REFERENCE_RIDGE_SCALE
    covariance_estimator: str = "ledoit_wolf"
    covariance_shrinkage: float = 0.10
    covariance_floor_rel: float = 1.0e-8
    max_cholesky_jitter_rel: float = 1.0e-4
    mean_half_lives: tuple[float, ...] = (6.0, 12.0, 24.0, 48.0, 96.0, math.inf)
    cov_half_lives: tuple[float, ...] = (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, math.inf)
    bootstrap_draws: int = 2000
    bootstrap_block_len: int = 12
    bootstrap_seed: int = 1234
    kernel_eta: float = FULL_COORDINATE_KERNEL_ETA
    input_already_centered: bool = False
    expected_production_p: int | None = 125


@dataclass(frozen=True)
class FullCoordinateGeometry:
    """Common ridge-whitened full-rank geometry built from pre-evaluation data."""

    p: int
    theta_hat: np.ndarray
    x: np.ndarray
    y: np.ndarray
    C_ref: np.ndarray
    D_ref: np.ndarray
    L_ref: np.ndarray
    rho_rel: float
    rho: float
    trace_C_ref: float
    d_rho: float
    logdet_L_ref: float
    ref_end: int
    centering_convention: str
    cholesky_min_pivot: float
    cholesky_max_pivot: float
    cholesky_diag_condition: float


@dataclass(frozen=True)
class ModelSpec:
    """A nested model defined only by mean and covariance persistence."""

    name: str
    mean_half_life: float
    cov_half_life: float
    lambda_mean: float
    lambda_cov: float


@dataclass(frozen=True)
class CovarianceRegularizationState:
    """Statistical covariance regularization parameters learned before scoring."""

    estimator: str
    shrinkage: float
    target_scale: float
    covariance_floor_rel: float
    max_cholesky_jitter_rel: float
    training_residual_count: int
    training_residual_rank: int


@dataclass(frozen=True)
class GaussianScoreComponents:
    """Gaussian log-score components computed from one SPD covariance."""

    logpdf: float
    constant_term: float
    logdet_term: float
    mahalanobis_term: float
    mahalanobis_squared: float
    logdet: float
    min_pivot: float
    jitter: float
    covariance_used: np.ndarray


@dataclass
class ModelScoreResult:
    """Period-level causal predictive scores and diagnostics for one model."""

    spec: ModelSpec
    dates: pd.Series
    log_scores_y: np.ndarray
    log_scores_x: np.ndarray
    constant_terms: np.ndarray
    logdet_terms: np.ndarray
    mahalanobis_terms: np.ndarray
    mahalanobis_squared: np.ndarray
    mahalanobis_per_dimension: np.ndarray
    marginal_log_scores: np.ndarray
    average_marginal_log_scores: np.ndarray
    residual_norms: np.ndarray
    mean_norms: np.ndarray
    covariance_traces: np.ndarray
    covariance_logdets: np.ndarray
    min_cholesky_pivots: np.ndarray
    jitters: np.ndarray
    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    K_const: np.ndarray
    K_const_raw: np.ndarray
    regularization: CovarianceRegularizationState
    covariance_floor: float
    covariance_floor_values: np.ndarray
    covariance_scale_used_for_regularization: np.ndarray
    raw_min_eigenvalues: np.ndarray
    raw_max_eigenvalues: np.ndarray
    regularized_min_eigenvalues: np.ndarray
    regularized_max_eigenvalues: np.ndarray
    raw_condition_numbers: np.ndarray
    regularized_condition_numbers: np.ndarray
    raw_effective_ranks: np.ndarray
    regularized_effective_ranks: np.ndarray
    number_eigenvalues_below_floor: np.ndarray
    number_eigenvalues_clipped: np.ndarray
    covariance_rank_before_regularization: np.ndarray


@dataclass(frozen=True)
class TuningResult:
    """Selected half-lives and all validation candidates."""

    selected_mean_half_life: float
    selected_cov_half_life: float
    selected_lambda_mean: float
    selected_lambda_cov: float
    candidates: pd.DataFrame


@dataclass
class FullCoordinateNestedResult:
    """Complete output bundle for the full-coordinate nested comparison."""

    config: FullCoordinateNestedConfig
    geometry: FullCoordinateGeometry
    validation_geometry: FullCoordinateGeometry
    tuning: TuningResult
    model_results: dict[str, ModelScoreResult]
    model_scores: pd.DataFrame
    model_summary: pd.DataFrame
    comparisons: pd.DataFrame
    covariance_diagnostics: pd.DataFrame
    score_decomposition: pd.DataFrame
    pair_decomposition: pd.DataFrame
    pair_summary: pd.DataFrame
    influential_dates: pd.DataFrame
    alternative_scores: pd.DataFrame
    structural_diagnostics: pd.DataFrame
    covariance_sensitivity: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    bootstrap_indices: np.ndarray
    mean_within: pd.DataFrame
    metadata: dict[str, object]
    validation_start: int
    eval_start: int


def config_from_env(
    *,
    bootstrap_draws: int | None = None,
    bootstrap_block_len: int | None = None,
) -> FullCoordinateNestedConfig:
    """Build config from environment variables used by the nested workflow."""

    def _grid(name: str, default: Iterable[float]) -> tuple[float, ...]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return tuple(default)
        out: list[float] = []
        for token in raw.split(","):
            x = token.strip().lower()
            if x in {"inf", "infinity", "infinite"}:
                out.append(math.inf)
            elif x:
                out.append(float(x))
        if not out:
            raise ValueError(f"{name} did not contain any valid half-lives.")
        return tuple(out)

    return FullCoordinateNestedConfig(
        eval_start_index=int(os.environ.get("OVK_NESTED_EVAL_START_INDEX", "180")),
        validation_months=int(os.environ.get("OVK_NESTED_VALIDATION_MONTHS", "80")),
        min_initial_observations=int(os.environ.get("OVK_NESTED_MIN_INITIAL_OBSERVATIONS", "60")),
        ridge_scale=float(os.environ.get("OVK_ARITHMETIC_REFERENCE_RIDGE_SCALE", str(ARITHMETIC_REFERENCE_RIDGE_SCALE))),
        covariance_estimator=os.environ.get("OVK_NESTED_COVARIANCE_ESTIMATOR", "ledoit_wolf").strip().lower(),
        covariance_shrinkage=float(os.environ.get("OVK_NESTED_COVARIANCE_SHRINKAGE", "0.10")),
        covariance_floor_rel=float(os.environ.get("OVK_NESTED_COVARIANCE_FLOOR_REL", "1e-8")),
        max_cholesky_jitter_rel=float(os.environ.get("OVK_NESTED_MAX_CHOLESKY_JITTER_REL", "1e-4")),
        mean_half_lives=_grid("OVK_NESTED_MEAN_HALF_LIVES", (6.0, 12.0, 24.0, 48.0, 96.0, math.inf)),
        cov_half_lives=_grid("OVK_NESTED_COV_HALF_LIVES", (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, math.inf)),
        bootstrap_draws=int(bootstrap_draws if bootstrap_draws is not None else os.environ.get("OVK_NESTED_SCORE_BOOTSTRAP_DRAWS", "2000")),
        bootstrap_block_len=int(bootstrap_block_len if bootstrap_block_len is not None else os.environ.get("OVK_NESTED_SCORE_BOOTSTRAP_BLOCK_LEN", "12")),
        bootstrap_seed=int(os.environ.get("OVK_NESTED_BOOTSTRAP_SEED", "1234")),
        kernel_eta=float(os.environ.get("OVK_FULL_COORDINATE_KERNEL_ETA", str(FULL_COORDINATE_KERNEL_ETA))),
        input_already_centered=os.environ.get("OVK_NESTED_INPUT_ALREADY_CENTERED", "0").lower() in {"1", "true", "yes", "on"},
        expected_production_p=None
        if os.environ.get("OVK_NESTED_EXPECTED_P", "125").strip().lower() in {"none", "any", ""}
        else int(os.environ.get("OVK_NESTED_EXPECTED_P", "125")),
    )


def half_life_to_lambda(half_life: float) -> float:
    """Convert a positive half-life to EWMA persistence."""
    hl = float(half_life)
    if math.isinf(hl):
        return 1.0
    if hl <= 0.0 or not np.isfinite(hl):
        raise ValueError("Finite half-lives must be positive.")
    return float(0.5 ** (1.0 / hl))


def _half_life_label(half_life: float) -> str:
    return "inf" if math.isinf(float(half_life)) else f"{float(half_life):.12g}"


def determine_splits(n: int, config: FullCoordinateNestedConfig) -> tuple[int, int]:
    """Return validation and final-evaluation start indices."""
    eval_start = int(config.eval_start_index)
    if eval_start <= 0 or eval_start >= n:
        raise ValueError(f"eval_start_index={eval_start} must lie inside the sample of length {n}.")
    validation_start = eval_start - int(config.validation_months)
    if validation_start < int(config.min_initial_observations):
        raise ValueError(
            "The validation split leaves too few initial observations: "
            f"validation_start={validation_start}, minimum={config.min_initial_observations}."
        )
    return validation_start, eval_start


def build_reference_geometry(
    q: np.ndarray,
    ref_end: int,
    config: FullCoordinateNestedConfig,
) -> FullCoordinateGeometry:
    """Build theta, ridge reference covariance, and full-rank whitened vectors."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim != 2:
        raise ValueError("q must be an observation-by-coordinate matrix.")
    n, p = q_arr.shape
    if ref_end <= 0 or ref_end > n:
        raise ValueError("ref_end must select a nonempty prefix of q.")
    if config.input_already_centered:
        theta_hat = np.zeros(p, dtype=float)
        centering = "input_already_centered_x_t_equals_q_t"
    else:
        theta_hat = q_arr[:ref_end].mean(axis=0)
        centering = "raw_lp_score_contributions_centered_by_pre_evaluation_theta_hat"
    x = q_arr - theta_hat[None, :]
    x_ref = x[:ref_end]
    C_ref = sym((x_ref.T @ x_ref) / max(ref_end, 1))
    rho = full_coordinate_ridge(C_ref, ridge_scale=config.ridge_scale)
    D_ref = sym(C_ref + rho * np.eye(p))
    L_ref = cholesky(D_ref, lower=True, check_finite=False)
    y = solve_triangular(L_ref, x.T, lower=True, check_finite=False).T
    solved_C = cho_solve((L_ref, True), C_ref, check_finite=False)
    d_rho = float(np.trace(solved_C))
    if not np.isfinite(d_rho) or d_rho <= 0.0:
        raise ValueError(f"d_rho must be positive and finite; got {d_rho}.")
    pivots = np.diag(L_ref)
    trace_C = float(np.trace(C_ref))
    logdet_L = float(np.sum(np.log(pivots)))
    return FullCoordinateGeometry(
        p=p,
        theta_hat=theta_hat,
        x=x,
        y=y,
        C_ref=C_ref,
        D_ref=D_ref,
        L_ref=L_ref,
        rho_rel=float(config.ridge_scale),
        rho=float(rho),
        trace_C_ref=trace_C,
        d_rho=d_rho,
        logdet_L_ref=logdet_L,
        ref_end=int(ref_end),
        centering_convention=centering,
        cholesky_min_pivot=float(np.min(pivots)),
        cholesky_max_pivot=float(np.max(pivots)),
        cholesky_diag_condition=float(np.max(pivots) / max(float(np.min(pivots)), 1e-300)),
    )


def _mean_training_residuals(y: np.ndarray, lambda_mean: float, fit_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = y.shape[1]
    residuals = np.empty((fit_end, p), dtype=float)
    predicted = np.empty((fit_end, p), dtype=float)
    m = np.zeros(p, dtype=float)
    dynamic_mean = float(lambda_mean) < 1.0
    for t in range(fit_end):
        predicted[t] = m
        residuals[t] = y[t] - m
        if dynamic_mean:
            m = float(lambda_mean) * m + (1.0 - float(lambda_mean)) * y[t]
    return residuals, predicted, m


def _k_const_from_residuals(residuals: np.ndarray) -> np.ndarray:
    return sym((residuals.T @ residuals) / max(len(residuals), 1))


def _safe_covariance_scale(K: np.ndarray) -> float:
    p = K.shape[0]
    scale = float(np.trace(sym(K))) / max(p, 1)
    if not np.isfinite(scale) or scale <= 0.0:
        vals = np.linalg.eigvalsh(sym(K))
        positive = vals[vals > 0.0]
        scale = float(np.mean(positive)) if len(positive) else 1.0
    return max(scale, 1.0e-12)


def _condition_number_from_eigs(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vmax = float(np.max(vals)) if len(vals) else float("nan")
    positive = vals[vals > 0.0]
    if not len(positive) or vmax <= 0.0:
        return float("inf")
    vmin = float(np.min(positive))
    return float(vmax / max(vmin, 1.0e-300))


def _rank_from_eigs(vals: np.ndarray, scale: float, tol: float = 1.0e-10) -> int:
    vals = np.asarray(vals, dtype=float)
    threshold = max(float(scale), 1.0) * float(tol)
    return int(np.sum(vals > threshold))


def _effective_rank_from_eigs(vals: np.ndarray) -> float:
    vals = np.maximum(np.asarray(vals, dtype=float), 0.0)
    total = float(np.sum(vals))
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    probs = vals[vals > 0.0] / total
    return float(np.exp(-np.sum(probs * np.log(probs))))


def _fit_covariance_regularization(
    residuals: np.ndarray,
    config: FullCoordinateNestedConfig,
) -> CovarianceRegularizationState:
    """Fit statistical regularization parameters using only pre-score residuals."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise ValueError("residuals must be a nonempty observation-by-coordinate matrix.")
    raw = _k_const_from_residuals(residuals)
    p = raw.shape[0]
    scale = _safe_covariance_scale(raw)
    rank = int(np.linalg.matrix_rank(residuals, tol=max(scale, 1.0) * 1.0e-10))
    estimator = str(config.covariance_estimator).strip().lower().replace("-", "_")
    if estimator in {"ledoit_wolf", "lw", "primary"}:
        lw = LedoitWolf(store_precision=False, assume_centered=True).fit(residuals)
        shrinkage = float(lw.shrinkage_)
        estimator = "ledoit_wolf"
    elif estimator in {"diagonal", "diag"}:
        shrinkage = 1.0
        estimator = "diagonal"
    elif estimator in {"legacy", "legacy_floor", "legacy_floor_add", "raw_floor"}:
        shrink = float(config.covariance_shrinkage)
        if shrink < 0.0 or shrink > 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1].")
        shrinkage = shrink
        estimator = "legacy_floor_add"
    else:
        raise ValueError(
            "covariance_estimator must be one of ledoit_wolf, diagonal, or legacy_floor_add; "
            f"got {config.covariance_estimator!r}."
        )
    return CovarianceRegularizationState(
        estimator=estimator,
        shrinkage=shrinkage,
        target_scale=scale,
        covariance_floor_rel=float(config.covariance_floor_rel),
        max_cholesky_jitter_rel=float(config.max_cholesky_jitter_rel),
        training_residual_count=int(residuals.shape[0]),
        training_residual_rank=rank,
    )


def _statistically_regularized_covariance(
    K_raw: np.ndarray,
    K_const_raw: np.ndarray,
    regularization: CovarianceRegularizationState,
) -> np.ndarray:
    """Apply the configured statistical covariance estimator before numerical flooring."""
    K_raw = sym(K_raw)
    K_const_raw = sym(K_const_raw)
    p = K_raw.shape[0]
    estimator = regularization.estimator
    scale = _safe_covariance_scale(K_raw)
    if estimator == "ledoit_wolf":
        alpha = float(regularization.shrinkage)
        return sym((1.0 - alpha) * K_raw + alpha * scale * np.eye(p))
    if estimator == "diagonal":
        return sym(np.diag(np.maximum(np.diag(K_raw), 0.0)))
    if estimator == "legacy_floor_add":
        alpha = float(regularization.shrinkage)
        return sym((1.0 - alpha) * K_raw + alpha * K_const_raw)
    raise ValueError(f"Unknown covariance estimator: {estimator}")


def _regularize_forecast_covariance(
    K_raw: np.ndarray,
    K_const_raw: np.ndarray,
    regularization: CovarianceRegularizationState,
) -> dict[str, object]:
    """Return final SPD covariance and detailed rank/floor diagnostics."""
    K_raw = sym(K_raw)
    stat = _statistically_regularized_covariance(K_raw, K_const_raw, regularization)
    p = K_raw.shape[0]
    raw_vals = np.linalg.eigvalsh(K_raw)
    stat_vals, stat_vecs = np.linalg.eigh(sym(stat))
    scale = _safe_covariance_scale(stat)
    floor_value = max(float(regularization.covariance_floor_rel) * scale, 0.0)
    if regularization.estimator == "legacy_floor_add":
        regularized = sym(stat + floor_value * np.eye(p))
        clipped = np.zeros(p, dtype=bool)
    else:
        clipped = stat_vals < floor_value
        final_vals = np.maximum(stat_vals, floor_value)
        regularized = sym((stat_vecs * final_vals) @ stat_vecs.T)
    reg_vals = np.linalg.eigvalsh(regularized)
    raw_scale = _safe_covariance_scale(K_raw)
    return {
        "raw": K_raw,
        "statistical": stat,
        "regularized": regularized,
        "covariance_scale": scale,
        "floor_value": floor_value,
        "raw_min_eigenvalue": float(np.min(raw_vals)),
        "raw_max_eigenvalue": float(np.max(raw_vals)),
        "regularized_min_eigenvalue": float(np.min(reg_vals)),
        "regularized_max_eigenvalue": float(np.max(reg_vals)),
        "raw_condition_number": _condition_number_from_eigs(raw_vals),
        "regularized_condition_number": _condition_number_from_eigs(reg_vals),
        "raw_effective_rank": _effective_rank_from_eigs(raw_vals),
        "regularized_effective_rank": _effective_rank_from_eigs(reg_vals),
        "number_eigenvalues_below_floor": int(np.sum(raw_vals < floor_value)),
        "number_eigenvalues_clipped": int(np.sum(clipped)),
        "covariance_rank_before_regularization": _rank_from_eigs(raw_vals, raw_scale),
    }


def cholesky_logpdf(
    e: np.ndarray,
    V: np.ndarray,
    *,
    max_jitter: float,
) -> GaussianScoreComponents:
    """Gaussian log density via Cholesky, using one final covariance for all terms."""
    p = len(e)
    jitter = 0.0
    eye = np.eye(p)
    attempts = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)
    last_error: Exception | None = None
    for multiplier in attempts:
        jitter = 0.0 if multiplier == 0.0 else multiplier * max_jitter
        if jitter > max_jitter:
            break
        try:
            used = sym(V + jitter * eye)
            L = cholesky(used, lower=True, check_finite=False)
            z = solve_triangular(L, e, lower=True, check_finite=False)
            quad = float(z @ z)
            logdet = float(2.0 * np.sum(np.log(np.diag(L))))
            constant = -0.5 * p * math.log(2.0 * math.pi)
            logdet_term = -0.5 * logdet
            mahalanobis_term = -0.5 * quad
            logpdf = constant + logdet_term + mahalanobis_term
            return GaussianScoreComponents(
                logpdf=float(logpdf),
                constant_term=float(constant),
                logdet_term=float(logdet_term),
                mahalanobis_term=float(mahalanobis_term),
                mahalanobis_squared=quad,
                logdet=logdet,
                min_pivot=float(np.min(np.diag(L))),
                jitter=float(jitter),
                covariance_used=used,
            )
        except Exception as exc:  # pragma: no cover - exercised only for pathological matrices
            last_error = exc
    raise np.linalg.LinAlgError(f"Cholesky failed before max_jitter={max_jitter:.6g}: {last_error}")


def score_causal_model(
    y: np.ndarray,
    dates: pd.Series,
    geometry: FullCoordinateGeometry,
    config: FullCoordinateNestedConfig,
    spec: ModelSpec,
    *,
    k_const_fit_end: int,
    score_start: int,
    score_end: int | None = None,
) -> ModelScoreResult:
    """Score one nested model with strictly causal mean and covariance updates."""
    y_arr = np.asarray(y, dtype=float)
    n, p = y_arr.shape
    stop = n if score_end is None else int(score_end)
    if not (0 < k_const_fit_end <= score_start <= stop <= n):
        raise ValueError("Invalid fit/score split.")

    lambda_mean = float(spec.lambda_mean)
    lambda_cov = float(spec.lambda_cov)
    dynamic_cov = lambda_cov < 1.0
    residuals_fit, _, mean_state = _mean_training_residuals(y_arr, lambda_mean, k_const_fit_end)
    K_const_raw = _k_const_from_residuals(residuals_fit)
    regularization = _fit_covariance_regularization(residuals_fit, config)
    const_reg = _regularize_forecast_covariance(K_const_raw, K_const_raw, regularization)
    K_const = np.asarray(const_reg["regularized"], dtype=float)
    covariance_floor = float(const_reg["floor_value"])
    max_jitter = float(config.max_cholesky_jitter_rel) * _safe_covariance_scale(K_const)
    K_state = K_const_raw.copy()

    for t in range(k_const_fit_end, score_start):
        e = y_arr[t] - mean_state
        if lambda_mean < 1.0:
            mean_state = lambda_mean * mean_state + (1.0 - lambda_mean) * y_arr[t]
        if dynamic_cov:
            K_state = sym(lambda_cov * K_state + (1.0 - lambda_cov) * np.outer(e, e))

    m = stop - score_start
    log_scores_y = np.empty(m, dtype=float)
    log_scores_x = np.empty(m, dtype=float)
    constant_terms = np.empty(m, dtype=float)
    logdet_terms = np.empty(m, dtype=float)
    mahalanobis_terms = np.empty(m, dtype=float)
    mahalanobis_squared = np.empty(m, dtype=float)
    mahalanobis_per_dimension = np.empty(m, dtype=float)
    marginal_log_scores = np.empty(m, dtype=float)
    average_marginal_log_scores = np.empty(m, dtype=float)
    residual_norms = np.empty(m, dtype=float)
    mean_norms = np.empty(m, dtype=float)
    covariance_traces = np.empty(m, dtype=float)
    covariance_logdets = np.empty(m, dtype=float)
    min_pivots = np.empty(m, dtype=float)
    jitters = np.empty(m, dtype=float)
    predicted_means = np.empty((m, p), dtype=float)
    predicted_covariances = np.empty((m, p, p), dtype=float)
    covariance_floor_values = np.empty(m, dtype=float)
    covariance_scales = np.empty(m, dtype=float)
    raw_min_eigs = np.empty(m, dtype=float)
    raw_max_eigs = np.empty(m, dtype=float)
    regularized_min_eigs = np.empty(m, dtype=float)
    regularized_max_eigs = np.empty(m, dtype=float)
    raw_conditions = np.empty(m, dtype=float)
    regularized_conditions = np.empty(m, dtype=float)
    raw_effective_ranks = np.empty(m, dtype=float)
    regularized_effective_ranks = np.empty(m, dtype=float)
    below_floor_counts = np.empty(m, dtype=int)
    clipped_counts = np.empty(m, dtype=int)
    ranks_before_regularization = np.empty(m, dtype=int)

    for out_i, t in enumerate(range(score_start, stop)):
        mean_pred = mean_state.copy()
        K_pred_raw = K_state if dynamic_cov else K_const_raw
        cov_reg = _regularize_forecast_covariance(K_pred_raw, K_const_raw, regularization)
        V = np.asarray(cov_reg["regularized"], dtype=float)
        e = y_arr[t] - mean_pred
        components = cholesky_logpdf(e, V, max_jitter=max_jitter)
        final_cov = components.covariance_used
        diag = np.maximum(np.diag(final_cov), 1.0e-300)
        marginal_terms = -0.5 * (np.log(2.0 * np.pi) + np.log(diag) + (e ** 2) / diag)
        log_scores_y[out_i] = components.logpdf
        log_scores_x[out_i] = components.logpdf - geometry.logdet_L_ref
        constant_terms[out_i] = components.constant_term
        logdet_terms[out_i] = components.logdet_term
        mahalanobis_terms[out_i] = components.mahalanobis_term
        mahalanobis_squared[out_i] = components.mahalanobis_squared
        mahalanobis_per_dimension[out_i] = components.mahalanobis_squared / max(p, 1)
        marginal_log_scores[out_i] = float(np.sum(marginal_terms))
        average_marginal_log_scores[out_i] = float(np.mean(marginal_terms))
        residual_norms[out_i] = float(np.linalg.norm(e))
        mean_norms[out_i] = float(np.linalg.norm(mean_pred))
        covariance_traces[out_i] = float(np.trace(final_cov))
        covariance_logdets[out_i] = components.logdet
        min_pivots[out_i] = components.min_pivot
        jitters[out_i] = components.jitter
        predicted_means[out_i] = mean_pred
        predicted_covariances[out_i] = final_cov
        covariance_floor_values[out_i] = float(cov_reg["floor_value"])
        covariance_scales[out_i] = float(cov_reg["covariance_scale"])
        raw_min_eigs[out_i] = float(cov_reg["raw_min_eigenvalue"])
        raw_max_eigs[out_i] = float(cov_reg["raw_max_eigenvalue"])
        regularized_min_eigs[out_i] = float(cov_reg["regularized_min_eigenvalue"])
        regularized_max_eigs[out_i] = float(cov_reg["regularized_max_eigenvalue"])
        raw_conditions[out_i] = float(cov_reg["raw_condition_number"])
        regularized_conditions[out_i] = float(cov_reg["regularized_condition_number"])
        raw_effective_ranks[out_i] = float(cov_reg["raw_effective_rank"])
        regularized_effective_ranks[out_i] = float(cov_reg["regularized_effective_rank"])
        below_floor_counts[out_i] = int(cov_reg["number_eigenvalues_below_floor"])
        clipped_counts[out_i] = int(cov_reg["number_eigenvalues_clipped"])
        ranks_before_regularization[out_i] = int(cov_reg["covariance_rank_before_regularization"])

        if lambda_mean < 1.0:
            mean_state = lambda_mean * mean_state + (1.0 - lambda_mean) * y_arr[t]
        if dynamic_cov:
            K_state = sym(lambda_cov * K_state + (1.0 - lambda_cov) * np.outer(e, e))

    return ModelScoreResult(
        spec=spec,
        dates=pd.Series(dates).iloc[score_start:stop].reset_index(drop=True),
        log_scores_y=log_scores_y,
        log_scores_x=log_scores_x,
        constant_terms=constant_terms,
        logdet_terms=logdet_terms,
        mahalanobis_terms=mahalanobis_terms,
        mahalanobis_squared=mahalanobis_squared,
        mahalanobis_per_dimension=mahalanobis_per_dimension,
        marginal_log_scores=marginal_log_scores,
        average_marginal_log_scores=average_marginal_log_scores,
        residual_norms=residual_norms,
        mean_norms=mean_norms,
        covariance_traces=covariance_traces,
        covariance_logdets=covariance_logdets,
        min_cholesky_pivots=min_pivots,
        jitters=jitters,
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        K_const=K_const,
        K_const_raw=K_const_raw,
        regularization=regularization,
        covariance_floor=covariance_floor,
        covariance_floor_values=covariance_floor_values,
        covariance_scale_used_for_regularization=covariance_scales,
        raw_min_eigenvalues=raw_min_eigs,
        raw_max_eigenvalues=raw_max_eigs,
        regularized_min_eigenvalues=regularized_min_eigs,
        regularized_max_eigenvalues=regularized_max_eigs,
        raw_condition_numbers=raw_conditions,
        regularized_condition_numbers=regularized_conditions,
        raw_effective_ranks=raw_effective_ranks,
        regularized_effective_ranks=regularized_effective_ranks,
        number_eigenvalues_below_floor=below_floor_counts,
        number_eigenvalues_clipped=clipped_counts,
        covariance_rank_before_regularization=ranks_before_regularization,
    )


def _spec(name: str, mean_half_life: float, cov_half_life: float) -> ModelSpec:
    return ModelSpec(
        name=name,
        mean_half_life=float(mean_half_life),
        cov_half_life=float(cov_half_life),
        lambda_mean=half_life_to_lambda(float(mean_half_life)),
        lambda_cov=half_life_to_lambda(float(cov_half_life)),
    )


def _validation_score(
    y: np.ndarray,
    dates: pd.Series,
    geometry: FullCoordinateGeometry,
    config: FullCoordinateNestedConfig,
    spec: ModelSpec,
    validation_start: int,
    eval_start: int,
) -> float:
    result = score_causal_model(
        y,
        dates,
        geometry,
        config,
        spec,
        k_const_fit_end=validation_start,
        score_start=validation_start,
        score_end=eval_start,
    )
    return float(np.mean(result.log_scores_x))


def tune_half_lives(
    q: np.ndarray,
    dates: pd.Series,
    config: FullCoordinateNestedConfig,
    validation_start: int,
    eval_start: int,
) -> tuple[TuningResult, FullCoordinateGeometry]:
    """Select M1 covariance and M2 mean half-lives on pre-evaluation validation data."""
    validation_geometry = build_reference_geometry(q, validation_start, config)
    y_val = validation_geometry.y
    rows: list[dict[str, object]] = []

    best_cov = None
    best_cov_score = -np.inf
    for half_life in config.cov_half_lives:
        spec = _spec("M1", math.inf, float(half_life))
        score = _validation_score(y_val, dates, validation_geometry, config, spec, validation_start, eval_start)
        rows.append(
            {
                "selection_target": "M1_cov_half_life",
                "candidate_half_life": _half_life_label(float(half_life)),
                "lambda_mean": 1.0,
                "lambda_cov": spec.lambda_cov,
                "validation_avg_log_score": score,
            }
        )
        if score > best_cov_score:
            best_cov_score = score
            best_cov = float(half_life)

    best_mean = None
    best_mean_score = -np.inf
    for half_life in config.mean_half_lives:
        spec = _spec("M2", float(half_life), math.inf)
        score = _validation_score(y_val, dates, validation_geometry, config, spec, validation_start, eval_start)
        rows.append(
            {
                "selection_target": "M2_mean_half_life",
                "candidate_half_life": _half_life_label(float(half_life)),
                "lambda_mean": spec.lambda_mean,
                "lambda_cov": 1.0,
                "validation_avg_log_score": score,
            }
        )
        if score > best_mean_score:
            best_mean_score = score
            best_mean = float(half_life)

    if best_cov is None or best_mean is None:
        raise RuntimeError("Half-life tuning did not evaluate any candidates.")
    candidates = pd.DataFrame(rows)
    candidates["selected"] = False
    candidates.loc[
        (candidates["selection_target"].eq("M1_cov_half_life"))
        & (candidates["candidate_half_life"].eq(_half_life_label(best_cov))),
        "selected",
    ] = True
    candidates.loc[
        (candidates["selection_target"].eq("M2_mean_half_life"))
        & (candidates["candidate_half_life"].eq(_half_life_label(best_mean))),
        "selected",
    ] = True
    tuning = TuningResult(
        selected_mean_half_life=best_mean,
        selected_cov_half_life=best_cov,
        selected_lambda_mean=half_life_to_lambda(best_mean),
        selected_lambda_cov=half_life_to_lambda(best_cov),
        candidates=candidates,
    )
    return tuning, validation_geometry


def model_specs_from_tuning(tuning: TuningResult) -> dict[str, ModelSpec]:
    return {
        "M0": _spec("M0", math.inf, math.inf),
        "M1": _spec("M1", math.inf, tuning.selected_cov_half_life),
        "M2": _spec("M2", tuning.selected_mean_half_life, math.inf),
        "M3": _spec("M3", tuning.selected_mean_half_life, tuning.selected_cov_half_life),
    }


def model_scores_frame(model_results: dict[str, ModelScoreResult]) -> pd.DataFrame:
    dates = next(iter(model_results.values())).dates
    out = pd.DataFrame({"date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d")})
    for name in MODEL_ORDER:
        result = model_results[name]
        out[f"log_score_{name}"] = result.log_scores_x
        out[f"log_score_y_{name}"] = result.log_scores_y
        out[f"constant_term_{name}"] = result.constant_terms
        out[f"logdet_term_{name}"] = result.logdet_terms
        out[f"mahalanobis_term_{name}"] = result.mahalanobis_terms
        out[f"mahalanobis_squared_{name}"] = result.mahalanobis_squared
        out[f"mahalanobis_per_dimension_{name}"] = result.mahalanobis_per_dimension
        out[f"marginal_log_score_{name}"] = result.marginal_log_scores
        out[f"avg_marginal_log_score_{name}"] = result.average_marginal_log_scores
        out[f"residual_norm_{name}"] = result.residual_norms
        out[f"predictive_mean_norm_{name}"] = result.mean_norms
        out[f"predictive_cov_trace_{name}"] = result.covariance_traces
        out[f"predictive_cov_logdet_{name}"] = result.covariance_logdets
        out[f"min_cholesky_pivot_{name}"] = result.min_cholesky_pivots
        out[f"cholesky_jitter_{name}"] = result.jitters
        out[f"number_eigenvalues_clipped_{name}"] = result.number_eigenvalues_clipped
        out[f"mean_half_life_{name}"] = _half_life_label(result.spec.mean_half_life)
        out[f"cov_half_life_{name}"] = _half_life_label(result.spec.cov_half_life)
        out[f"lambda_mean_{name}"] = result.spec.lambda_mean
        out[f"lambda_cov_{name}"] = result.spec.lambda_cov
    return out


def paired_moving_block_bootstrap(
    score_matrix: np.ndarray,
    *,
    block_len: int,
    draws: int,
    seed: int,
    return_indices: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap model means and all contrasts using one paired index path per draw."""
    scores = np.asarray(score_matrix, dtype=float)
    n, k = scores.shape
    if k != len(MODEL_ORDER):
        raise ValueError("score_matrix columns must be ordered as M0, M1, M2, M3.")
    rng = np.random.default_rng(seed)
    B = max(int(draws), 1)
    draw_rows: list[dict[str, float | int]] = []
    all_indices = np.empty((B, n), dtype=int) if return_indices else np.empty((0, 0), dtype=int)
    model_pos = {name: i for i, name in enumerate(MODEL_ORDER)}
    for b in range(B):
        ix = circular_block_indices(n, int(block_len), rng)
        if return_indices:
            all_indices[b] = ix
        means = scores[ix].mean(axis=0)
        row: dict[str, float | int] = {"draw": b}
        for name, pos in model_pos.items():
            row[f"avg_log_score_{name}"] = float(means[pos])
        for a, c, _ in COMPARISON_PAIRS:
            row[f"{a}_minus_{c}"] = float(means[model_pos[a]] - means[model_pos[c]])
        draw_rows.append(row)
    return pd.DataFrame(draw_rows), all_indices


def comparison_summary(
    model_scores: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    d_rho: float,
    score_dimension: int | None = None,
) -> pd.DataFrame:
    """Summarize raw and per-effective-coordinate model contrasts."""
    rows: list[dict[str, object]] = []
    n_eval = int(len(model_scores))
    p = int(score_dimension) if score_dimension is not None else None
    per_coord_denominator = float(p) if p and p > 0 else np.nan
    for a, b, desc in COMPARISON_PAIRS:
        diff = model_scores[f"log_score_{a}"].to_numpy(float) - model_scores[f"log_score_{b}"].to_numpy(float)
        draw_col = f"{a}_minus_{b}"
        draws = bootstrap_draws[draw_col].to_numpy(float)
        mean = float(diff.mean())
        lo = float(np.percentile(draws, 5))
        hi = float(np.percentile(draws, 95))
        total = float(diff.sum())
        rows.append(
            {
                "comparison": f"{a} - {b}",
                "meaning": desc,
                "score_difference_unit": "average per evaluation date",
                "n_evaluation_dates": n_eval,
                "score_dimension": p if p is not None else np.nan,
                "avg_log_score_diff": mean,
                "p05": lo,
                "p95": hi,
                "prob_diff_gt_0": float(np.mean(draws > 0.0)),
                "sum_log_score_diff": total,
                "avg_joint_log_score_diff_per_dimension": mean / per_coord_denominator,
                "avg_log_score_diff_per_d_rho": mean / float(d_rho),
                "p05_per_d_rho": lo / float(d_rho),
                "p95_per_d_rho": hi / float(d_rho),
                "bootstrap_type": "paired moving-block bootstrap of fitted score differentials",
            }
        )
    return pd.DataFrame(rows)


def model_summary_frame(model_results: dict[str, ModelScoreResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        result = model_results[name]
        p = int(result.predicted_means.shape[1])
        rows.append(
            {
                "model": name,
                "score_difference_unit": "average per evaluation date",
                "score_dimension": p,
                "avg_log_score": float(np.mean(result.log_scores_x)),
                "sum_log_score": float(np.sum(result.log_scores_x)),
                "avg_joint_log_score_per_dimension": float(np.mean(result.log_scores_x)) / float(p),
                "avg_marginal_log_score": float(np.mean(result.average_marginal_log_scores)),
                "mean_half_life": _half_life_label(result.spec.mean_half_life),
                "cov_half_life": _half_life_label(result.spec.cov_half_life),
                "lambda_mean": result.spec.lambda_mean,
                "lambda_cov": result.spec.lambda_cov,
                "covariance_estimator": result.regularization.estimator,
                "statistical_shrinkage": result.regularization.shrinkage,
                "training_residual_count": result.regularization.training_residual_count,
                "training_residual_rank": result.regularization.training_residual_rank,
                "min_covariance_floor": float(np.min(result.covariance_floor_values)),
                "max_covariance_floor": float(np.max(result.covariance_floor_values)),
                "max_cholesky_jitter": float(np.max(result.jitters)),
                "jittered_periods": int(np.sum(result.jitters > 0.0)),
                "min_cholesky_pivot": float(np.min(result.min_cholesky_pivots)),
                "mean_predictive_cov_trace": float(np.mean(result.covariance_traces)),
            }
        )
    return pd.DataFrame(rows)


def _smallest_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    """Return the smallest eigenvalue of a symmetric dense matrix."""
    vals = eigh(sym(matrix), eigvals_only=True, subset_by_index=[0, 0], check_finite=False)
    return float(vals[0])


def covariance_diagnostics_frame(model_results: dict[str, ModelScoreResult]) -> pd.DataFrame:
    """Summarize covariance floors and exact predictive-covariance eigenvalue minima."""
    rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        result = model_results[name]
        p = int(result.predicted_covariances.shape[1])
        clipped_dates = result.number_eigenvalues_clipped > 0
        rows.append(
            {
                "model": name,
                "score_dimension": p,
                "n_evaluation_dates": int(len(result.regularized_min_eigenvalues)),
                "covariance_estimator": result.regularization.estimator,
                "statistical_shrinkage": float(result.regularization.shrinkage),
                "training_residual_count": int(result.regularization.training_residual_count),
                "training_residual_rank": int(result.regularization.training_residual_rank),
                "covariance_floor": float(np.min(result.covariance_floor_values)),
                "max_covariance_floor": float(np.max(result.covariance_floor_values)),
                "min_raw_cov_eigenvalue": float(np.min(result.raw_min_eigenvalues)),
                "min_predicted_cov_eigenvalue": float(np.min(result.regularized_min_eigenvalues)),
                "p05_predicted_cov_min_eigenvalue": float(np.percentile(result.regularized_min_eigenvalues, 5)),
                "median_predicted_cov_min_eigenvalue": float(np.median(result.regularized_min_eigenvalues)),
                "min_K_const_eigenvalue_before_regularization": _smallest_symmetric_eigenvalue(result.K_const_raw),
                "min_eigenvalue_to_floor_ratio": float(
                    np.min(result.regularized_min_eigenvalues) / max(result.covariance_floor, 1.0e-300)
                ),
                "dates_with_any_clipped_eigenvalues": int(np.sum(clipped_dates)),
                "pct_dates_with_any_clipped_eigenvalues": float(np.mean(clipped_dates)),
                "total_eigenvalues_clipped": int(np.sum(result.number_eigenvalues_clipped)),
                "max_fraction_eigenvalues_clipped": float(np.max(result.number_eigenvalues_clipped) / max(p, 1)),
                "min_cholesky_pivot": float(np.min(result.min_cholesky_pivots)),
                "min_cholesky_pivot_squared": float(np.min(result.min_cholesky_pivots) ** 2),
                "max_cholesky_jitter": float(np.max(result.jitters)),
                "jittered_periods": int(np.sum(result.jitters > 0.0)),
                "mean_predictive_cov_trace": float(np.mean(result.covariance_traces)),
                "min_predictive_cov_trace": float(np.min(result.covariance_traces)),
            }
        )
    return pd.DataFrame(rows)


def gaussian_score_decomposition_frame(model_results: dict[str, ModelScoreResult]) -> pd.DataFrame:
    """Long date-model Gaussian score decomposition with covariance diagnostics."""
    rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        result = model_results[name]
        dates = pd.to_datetime(result.dates).dt.strftime("%Y-%m-%d")
        p = int(result.predicted_means.shape[1])
        for i, date in enumerate(dates):
            clipped = int(result.number_eigenvalues_clipped[i])
            rows.append(
                {
                    "evaluation_date": date,
                    "model": name,
                    "dimension": p,
                    "log_score": float(result.log_scores_x[i]),
                    "log_score_y": float(result.log_scores_y[i]),
                    "constant_term": float(result.constant_terms[i]),
                    "logdet_term": float(result.logdet_terms[i]),
                    "mahalanobis_term": float(result.mahalanobis_terms[i]),
                    "mahalanobis_squared": float(result.mahalanobis_squared[i]),
                    "mahalanobis_per_dimension": float(result.mahalanobis_per_dimension[i]),
                    "raw_min_eigenvalue": float(result.raw_min_eigenvalues[i]),
                    "regularized_min_eigenvalue": float(result.regularized_min_eigenvalues[i]),
                    "raw_max_eigenvalue": float(result.raw_max_eigenvalues[i]),
                    "regularized_max_eigenvalue": float(result.regularized_max_eigenvalues[i]),
                    "raw_condition_number": float(result.raw_condition_numbers[i]),
                    "regularized_condition_number": float(result.regularized_condition_numbers[i]),
                    "raw_effective_rank": float(result.raw_effective_ranks[i]),
                    "regularized_effective_rank": float(result.regularized_effective_ranks[i]),
                    "number_eigenvalues_below_floor": int(result.number_eigenvalues_below_floor[i]),
                    "number_eigenvalues_clipped": clipped,
                    "fraction_eigenvalues_clipped": float(clipped / max(p, 1)),
                    "covariance_scale_used_for_regularization": float(result.covariance_scale_used_for_regularization[i]),
                    "covariance_floor": float(result.covariance_floor_values[i]),
                    "covariance_estimator": result.regularization.estimator,
                    "statistical_shrinkage": float(result.regularization.shrinkage),
                    "jitter_added": float(result.jitters[i]),
                    "training_residual_count": int(result.regularization.training_residual_count),
                    "covariance_rank_before_regularization": int(result.covariance_rank_before_regularization[i]),
                    "marginal_log_score_sum": float(result.marginal_log_scores[i]),
                    "avg_marginal_log_score": float(result.average_marginal_log_scores[i]),
                }
            )
    return pd.DataFrame(rows)


def model_pair_score_decomposition_frame(score_decomposition: pd.DataFrame) -> pd.DataFrame:
    """Pairwise date-level score differences split into logdet and Mahalanobis pieces."""
    rows: list[dict[str, object]] = []
    by_model = {name: df.set_index("evaluation_date") for name, df in score_decomposition.groupby("model")}
    for a, b, desc in COMPARISON_PAIRS:
        left = by_model[a]
        right = by_model[b]
        common_dates = left.index.intersection(right.index)
        for date in common_dates:
            la = left.loc[date]
            rb = right.loc[date]
            const = float(la["constant_term"] - rb["constant_term"])
            logdet = float(la["logdet_term"] - rb["logdet_term"])
            maha = float(la["mahalanobis_term"] - rb["mahalanobis_term"])
            total = float(la["log_score"] - rb["log_score"])
            rows.append(
                {
                    "evaluation_date": date,
                    "comparison": f"{a} - {b}",
                    "meaning": desc,
                    "score_diff": total,
                    "constant_contribution": const,
                    "logdet_contribution": logdet,
                    "mahalanobis_contribution": maha,
                    "reconstruction_error": float(total - const - logdet - maha),
                    "left_mahalanobis_per_dimension": float(la["mahalanobis_per_dimension"]),
                    "right_mahalanobis_per_dimension": float(rb["mahalanobis_per_dimension"]),
                    "left_clipped_eigenvalues": int(la["number_eigenvalues_clipped"]),
                    "right_clipped_eigenvalues": int(rb["number_eigenvalues_clipped"]),
                    "either_model_clipped": bool(
                        int(la["number_eigenvalues_clipped"]) > 0 or int(rb["number_eigenvalues_clipped"]) > 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _top_abs_concentration(values: np.ndarray, k: int) -> float:
    values = np.asarray(values, dtype=float)
    denom = float(np.sum(values))
    if abs(denom) < 1.0e-300:
        return float("nan")
    order = np.argsort(np.abs(values))[::-1][: min(k, len(values))]
    return float(np.sum(values[order]) / denom)


def pair_score_summary_frame(pair_decomposition: pd.DataFrame) -> pd.DataFrame:
    """Summaries and top-date concentration for score differences."""
    rows: list[dict[str, object]] = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]
    for comparison, df in pair_decomposition.groupby("comparison", sort=False):
        vals = df["score_diff"].to_numpy(float)
        q = np.quantile(vals, quantiles)
        rows.append(
            {
                "comparison": comparison,
                "n_dates": int(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "q01": float(q[0]),
                "q05": float(q[1]),
                "q10": float(q[2]),
                "q25": float(q[3]),
                "q75": float(q[4]),
                "q90": float(q[5]),
                "q95": float(q[6]),
                "q99": float(q[7]),
                "top1_abs_date_fraction_of_total": _top_abs_concentration(vals, 1),
                "top5_abs_dates_fraction_of_total": _top_abs_concentration(vals, 5),
                "top10_abs_dates_fraction_of_total": _top_abs_concentration(vals, 10),
                "mean_left_mahalanobis_per_dimension": float(np.mean(df["left_mahalanobis_per_dimension"])),
                "mean_right_mahalanobis_per_dimension": float(np.mean(df["right_mahalanobis_per_dimension"])),
                "q95_left_mahalanobis_per_dimension": float(np.quantile(df["left_mahalanobis_per_dimension"], 0.95)),
                "q95_right_mahalanobis_per_dimension": float(np.quantile(df["right_mahalanobis_per_dimension"], 0.95)),
                "dates_with_any_clipped_eigenvalues": int(np.sum(df["either_model_clipped"])),
                "pct_dates_with_any_clipped_eigenvalues": float(np.mean(df["either_model_clipped"])),
                "total_clipped_eigenvalues": int(
                    np.sum(df["left_clipped_eigenvalues"]) + np.sum(df["right_clipped_eigenvalues"])
                ),
            }
        )
    return pd.DataFrame(rows)


def influential_dates_frame(pair_decomposition: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top dates by absolute model-pair score difference."""
    rows: list[pd.DataFrame] = []
    for comparison, df in pair_decomposition.groupby("comparison", sort=False):
        vals = df["score_diff"].to_numpy(float)
        total = float(np.sum(vals))
        view = df.copy()
        view["abs_score_diff"] = np.abs(vals)
        view["fraction_of_total_difference"] = vals / total if abs(total) > 1.0e-300 else np.nan
        view = view.sort_values("abs_score_diff", ascending=False).head(top_n)
        view.insert(0, "abs_rank_within_comparison", np.arange(1, len(view) + 1))
        rows.append(view)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def alternative_score_summary_frame(model_results: dict[str, ModelScoreResult]) -> pd.DataFrame:
    """Model-level marginal score diagnostics that avoid full-covariance inversion."""
    rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        result = model_results[name]
        rows.append(
            {
                "model": name,
                "joint_log_score_mean_per_date": float(np.mean(result.log_scores_x)),
                "joint_log_score_mean_per_dimension": float(
                    np.mean(result.log_scores_x) / max(result.predicted_means.shape[1], 1)
                ),
                "marginal_log_score_sum_mean_per_date": float(np.mean(result.marginal_log_scores)),
                "average_coordinate_marginal_log_score": float(np.mean(result.average_marginal_log_scores)),
            }
        )
    return pd.DataFrame(rows)


def structural_subspace_diagnostics(
    q: np.ndarray,
    *,
    residuals: np.ndarray | None = None,
    tol: float = 1.0e-10,
) -> pd.DataFrame:
    """Audit exact or near-exact linear constraints in score and residual matrices."""
    rows: list[dict[str, object]] = []
    for label, matrix in [("score_matrix", q), ("training_residual_matrix", residuals)]:
        if matrix is None:
            continue
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            continue
        svals = np.linalg.svd(arr, compute_uv=False)
        largest = float(svals[0]) if len(svals) else 0.0
        threshold = max(largest, 1.0) * tol
        rank = int(np.sum(svals > threshold))
        rows.append(
            {
                "matrix": label,
                "n_observations": int(arr.shape[0]),
                "dimension": int(arr.shape[1]),
                "numerical_rank": rank,
                "rank_deficiency": int(arr.shape[1] - rank),
                "smallest_singular_value": float(svals[-1]) if len(svals) else float("nan"),
                "largest_singular_value": largest,
                "condition_number": float(largest / max(float(svals[-1]), 1.0e-300)) if len(svals) else float("nan"),
                "tolerance": float(threshold),
                "structural_constraint_detected": bool(arr.shape[0] > arr.shape[1] and rank < arr.shape[1]),
                "rank_limited_by_observation_count": bool(arr.shape[0] <= arr.shape[1] and rank <= arr.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def covariance_sensitivity_analysis(
    q: np.ndarray,
    dates: pd.Series,
    base_config: FullCoordinateNestedConfig,
) -> pd.DataFrame:
    """Run reproducible covariance-estimator and numerical-floor sensitivity checks."""
    settings: list[tuple[str, FullCoordinateNestedConfig]] = [
        (
            "primary_ledoit_wolf",
            replace(base_config, covariance_estimator="ledoit_wolf"),
        ),
        (
            "diagonal_covariance",
            replace(base_config, covariance_estimator="diagonal"),
        ),
        (
            "previous_legacy_floor_add",
            replace(base_config, covariance_estimator="legacy_floor_add", covariance_floor_rel=1.0e-4),
        ),
        (
            "ledoit_wolf_floor_1e-10",
            replace(base_config, covariance_estimator="ledoit_wolf", covariance_floor_rel=1.0e-10),
        ),
        (
            "ledoit_wolf_floor_1e-8",
            replace(base_config, covariance_estimator="ledoit_wolf", covariance_floor_rel=1.0e-8),
        ),
        (
            "ledoit_wolf_floor_1e-6",
            replace(base_config, covariance_estimator="ledoit_wolf", covariance_floor_rel=1.0e-6),
        ),
    ]
    rows: list[dict[str, object]] = []
    primary_signs: dict[str, float] = {}
    for setting_name, cfg in settings:
        result = run_full_coordinate_nested(
            q,
            dates,
            cfg,
            include_mean_within=False,
            include_sensitivity=False,
        )
        summary_by_comp = result.pair_summary.set_index("comparison")
        for comp_row in result.comparisons.itertuples(index=False):
            comparison = str(comp_row.comparison)
            pair_stats = summary_by_comp.loc[comparison]
            diff = result.pair_decomposition.loc[
                result.pair_decomposition["comparison"].eq(comparison),
                "score_diff",
            ].to_numpy(float)
            sign = float(np.sign(float(comp_row.avg_log_score_diff)))
            if setting_name == "primary_ledoit_wolf":
                primary_signs[comparison] = sign
            rows.append(
                {
                    "setting": setting_name,
                    "covariance_estimator": cfg.covariance_estimator,
                    "covariance_floor_rel": float(cfg.covariance_floor_rel),
                    "comparison": comparison,
                    "avg_date_level_joint_log_score_diff": float(comp_row.avg_log_score_diff),
                    "median_date_level_diff": float(pair_stats["median"]),
                    "p05": float(comp_row.p05),
                    "p95": float(comp_row.p95),
                    "prob_bootstrap_diff_gt_0": float(comp_row.prob_diff_gt_0),
                    "proportion_dates_won": float(np.mean(diff > 0.0)),
                    "top1_abs_date_fraction_of_total": float(pair_stats["top1_abs_date_fraction_of_total"]),
                    "top5_abs_dates_fraction_of_total": float(pair_stats["top5_abs_dates_fraction_of_total"]),
                    "top10_abs_dates_fraction_of_total": float(pair_stats["top10_abs_dates_fraction_of_total"]),
                    "pct_dates_with_any_clipped_eigenvalues": float(
                        pair_stats["pct_dates_with_any_clipped_eigenvalues"]
                    ),
                    "total_clipped_eigenvalues": int(pair_stats["total_clipped_eigenvalues"]),
                    "sign": sign,
                    "sign_differs_from_primary": bool(
                        comparison in primary_signs and sign != 0.0 and primary_signs[comparison] != 0.0 and sign != primary_signs[comparison]
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_mean_within_diagnostic(
    dates: pd.Series,
    geometry: FullCoordinateGeometry,
    config: FullCoordinateNestedConfig,
) -> pd.DataFrame:
    """Two-sided descriptive mean/within decomposition, separate from prediction."""
    weights = full_coordinate_temporal_weights(dates, eta=config.kernel_eta)
    x = np.asarray(geometry.x, dtype=float)
    K_total = full_coordinate_K_from_weights(x, weights)
    local_mean = weights @ x
    K_mean = np.einsum("ti,tj->tij", local_mean, local_mean, optimize=True)
    denominator = float(geometry.d_rho)
    rows: list[dict[str, object]] = []
    for t in range(len(x)):
        kt = sym(K_total[t])
        km = sym(K_mean[t])
        kw_raw = sym(kt - km)
        min_eig = float(np.linalg.eigvalsh(kw_raw).min())
        scale = max(float(np.trace(kt)) / max(geometry.p, 1), 1.0e-12)
        correction_norm = 0.0
        if min_eig < -1.0e-7 * scale:
            raise ValueError(f"K_within has material negative eigenvalue at t={t}: {min_eig:.6g}.")
        if min_eig < 0.0:
            correction_norm = float(np.linalg.norm((-min_eig + 1.0e-12 * scale) * np.eye(geometry.p), ord="fro"))
        kw = kw_raw
        total_solved = cho_solve((geometry.L_ref, True), kt, check_finite=False)
        mean_solved = cho_solve((geometry.L_ref, True), km, check_finite=False)
        within_solved = cho_solve((geometry.L_ref, True), kw, check_finite=False)
        tau_total = float(np.trace(total_solved) / denominator)
        tau_mean = float(np.trace(mean_solved) / denominator)
        tau_within = float(np.trace(within_solved) / denominator)
        rows.append(
            {
                "date": pd.to_datetime(pd.Series(dates).iloc[t]).strftime("%Y-%m-%d"),
                "tau_total": tau_total,
                "tau_mean": tau_mean,
                "tau_within": tau_within,
                "tau_decomposition_error": float(tau_total - tau_mean - tau_within),
                "local_mean_norm": float(np.linalg.norm(local_mean[t])),
                "smoothing_ess": float(1.0 / np.sum(weights[t] ** 2)),
                "smoothing_support": int(np.sum(weights[t] > 1.0e-12)),
                "within_min_eigenvalue_before_cleanup": min_eig,
                "within_psd_cleanup_frobenius_norm": correction_norm,
            }
        )
    out = pd.DataFrame(rows)
    max_error = float(np.max(np.abs(out["tau_decomposition_error"].to_numpy(float))))
    if max_error > 1.0e-6:
        raise AssertionError(f"Mean/within tau decomposition error is too large: {max_error:.6g}.")
    return out


def _metadata_dict(
    config: FullCoordinateNestedConfig,
    geometry: FullCoordinateGeometry,
    tuning: TuningResult,
    dates: pd.Series,
    validation_start: int,
    eval_start: int,
    final_fit_end: int,
    model_summary: pd.DataFrame,
    covariance_diagnostics: pd.DataFrame,
) -> dict[str, object]:
    max_jitter = float(model_summary["max_cholesky_jitter"].max()) if len(model_summary) else 0.0
    jittered = int(model_summary["jittered_periods"].sum()) if len(model_summary) else 0
    min_cov_eig = (
        float(covariance_diagnostics["min_predicted_cov_eigenvalue"].min())
        if len(covariance_diagnostics)
        else float("nan")
    )
    min_cov_floor = (
        float(covariance_diagnostics["covariance_floor"].min())
        if len(covariance_diagnostics)
        else float("nan")
    )
    return {
        "estimator_mode": "full_coordinate_arithmetic_causal_predictive",
        "p": int(geometry.p),
        "centering_convention": geometry.centering_convention,
        "theta_hat_source": "prefix ending immediately before final evaluation start",
        "input_already_centered": bool(config.input_already_centered),
        "validation_start_index": int(validation_start),
        "eval_start_index": int(eval_start),
        "final_scoring_fit_end_index": int(final_fit_end),
        "final_scoring_training_residual_count": int(final_fit_end),
        "validation_observations_for_tuning": int(eval_start - validation_start),
        "sample_start": pd.to_datetime(pd.Series(dates).iloc[0]).strftime("%Y-%m-%d"),
        "validation_start_date": pd.to_datetime(pd.Series(dates).iloc[validation_start]).strftime("%Y-%m-%d"),
        "eval_start_date": pd.to_datetime(pd.Series(dates).iloc[eval_start]).strftime("%Y-%m-%d"),
        "eval_end_date": pd.to_datetime(pd.Series(dates).iloc[-1]).strftime("%Y-%m-%d"),
        "rho_rel": float(geometry.rho_rel),
        "rho": float(geometry.rho),
        "trace_C_ref": float(geometry.trace_C_ref),
        "d_rho": float(geometry.d_rho),
        "cholesky_diagnostics": {
            "min_pivot": float(geometry.cholesky_min_pivot),
            "max_pivot": float(geometry.cholesky_max_pivot),
            "diag_condition": float(geometry.cholesky_diag_condition),
        },
        "selected_half_lives": {
            "mean_half_life": _half_life_label(tuning.selected_mean_half_life),
            "cov_half_life": _half_life_label(tuning.selected_cov_half_life),
            "lambda_mean": float(tuning.selected_lambda_mean),
            "lambda_cov": float(tuning.selected_lambda_cov),
        },
        "covariance_stabilization": {
            "covariance_estimator": str(config.covariance_estimator),
            "covariance_shrinkage": float(config.covariance_shrinkage),
            "covariance_floor_rel": float(config.covariance_floor_rel),
            "min_covariance_floor": min_cov_floor,
            "min_predicted_cov_eigenvalue": min_cov_eig,
            "max_cholesky_jitter_rel": float(config.max_cholesky_jitter_rel),
            "max_cholesky_jitter_triggered": max_jitter,
            "jittered_periods_total": jittered,
        },
        "bootstrap": {
            "draws": int(config.bootstrap_draws),
            "block_length_months": int(config.bootstrap_block_len),
            "seed": int(config.bootstrap_seed),
            "type": "paired moving-block bootstrap of fitted score differentials",
        },
        "reused_arithmetic_helpers": [
            "run_publication_grade_ovk.full_coordinate_ridge",
            "run_publication_grade_ovk.full_coordinate_temporal_weights",
            "run_publication_grade_ovk.full_coordinate_K_from_weights",
            "run_publication_grade_ovk.circular_block_indices",
            "run_publication_grade_ovk.sym",
        ],
        "forbidden_legacy_components_used": [],
    }


def run_full_coordinate_nested(
    q: np.ndarray,
    dates: pd.Series,
    config: FullCoordinateNestedConfig,
    *,
    include_mean_within: bool = True,
    include_sensitivity: bool = False,
) -> FullCoordinateNestedResult:
    """Run tuning, causal scoring, paired bootstrap, and descriptive diagnostics."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim != 2:
        raise ValueError("q must be an observation-by-coordinate matrix.")
    validation_start, eval_start = determine_splits(len(q_arr), config)
    tuning, validation_geometry = tune_half_lives(q_arr, dates, config, validation_start, eval_start)
    geometry = build_reference_geometry(q_arr, eval_start, config)
    specs = model_specs_from_tuning(tuning)
    final_fit_end = eval_start
    model_results = {
        name: score_causal_model(
            geometry.y,
            dates,
            geometry,
            config,
            specs[name],
            k_const_fit_end=final_fit_end,
            score_start=eval_start,
            score_end=len(q_arr),
        )
        for name in MODEL_ORDER
    }
    scores = model_scores_frame(model_results)
    score_matrix = scores[[f"log_score_{name}" for name in MODEL_ORDER]].to_numpy(float)
    boot_draws, boot_indices = paired_moving_block_bootstrap(
        score_matrix,
        block_len=config.bootstrap_block_len,
        draws=config.bootstrap_draws,
        seed=config.bootstrap_seed,
        return_indices=True,
    )
    comparisons = comparison_summary(scores, boot_draws, geometry.d_rho, score_dimension=geometry.p)
    model_summary = model_summary_frame(model_results)
    covariance_diagnostics = covariance_diagnostics_frame(model_results)
    score_decomposition = gaussian_score_decomposition_frame(model_results)
    pair_decomposition = model_pair_score_decomposition_frame(score_decomposition)
    pair_summary = pair_score_summary_frame(pair_decomposition)
    influential_dates = influential_dates_frame(pair_decomposition)
    alternative_scores = alternative_score_summary_frame(model_results)
    m0_residuals, _, _ = _mean_training_residuals(geometry.y, specs["M0"].lambda_mean, validation_start)
    structural_diagnostics = structural_subspace_diagnostics(geometry.y, residuals=m0_residuals)
    mean_within = compute_mean_within_diagnostic(dates, geometry, config) if include_mean_within else pd.DataFrame()
    covariance_sensitivity = (
        covariance_sensitivity_analysis(q_arr, dates, config) if include_sensitivity else pd.DataFrame()
    )
    metadata = _metadata_dict(
        config,
        geometry,
        tuning,
        dates,
        validation_start,
        eval_start,
        final_fit_end,
        model_summary,
        covariance_diagnostics,
    )
    return FullCoordinateNestedResult(
        config=config,
        geometry=geometry,
        validation_geometry=validation_geometry,
        tuning=tuning,
        model_results=model_results,
        model_scores=scores,
        model_summary=model_summary,
        comparisons=comparisons,
        covariance_diagnostics=covariance_diagnostics,
        score_decomposition=score_decomposition,
        pair_decomposition=pair_decomposition,
        pair_summary=pair_summary,
        influential_dates=influential_dates,
        alternative_scores=alternative_scores,
        structural_diagnostics=structural_diagnostics,
        covariance_sensitivity=covariance_sensitivity,
        bootstrap_draws=boot_draws,
        bootstrap_indices=boot_indices,
        mean_within=mean_within,
        metadata=metadata,
        validation_start=validation_start,
        eval_start=eval_start,
    )


def write_full_coordinate_outputs(result: FullCoordinateNestedResult, table_dir: Path) -> dict[str, Path]:
    """Write the explicit full-coordinate audit artifacts."""
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": table_dir / "nested_full_coordinate_model_scores.csv",
        "comparisons": table_dir / "nested_full_coordinate_comparisons.csv",
        "covariance_diagnostics": table_dir / "nested_full_coordinate_covariance_diagnostics.csv",
        "score_decomposition": table_dir / "nested_full_coordinate_gaussian_score_decomposition.csv",
        "pair_decomposition": table_dir / "nested_full_coordinate_model_pair_score_decomposition.csv",
        "pair_summary": table_dir / "nested_full_coordinate_model_pair_score_summary.csv",
        "influential_dates": table_dir / "nested_full_coordinate_influential_evaluation_dates.csv",
        "alternative_scores": table_dir / "nested_full_coordinate_alternative_score_diagnostics.csv",
        "structural_diagnostics": table_dir / "nested_full_coordinate_structural_subspace_diagnostics.csv",
        "covariance_sensitivity": table_dir / "nested_full_coordinate_covariance_sensitivity.csv",
        "tuning": table_dir / "nested_full_coordinate_tuning.csv",
        "mean_within": table_dir / "nested_full_coordinate_mean_within.csv",
        "metadata": table_dir / "nested_full_coordinate_metadata.json",
        "bootstrap_draws": table_dir / "nested_full_coordinate_bootstrap_draws.csv",
    }
    result.model_scores.to_csv(paths["scores"], index=False)
    result.comparisons.to_csv(paths["comparisons"], index=False)
    result.covariance_diagnostics.to_csv(paths["covariance_diagnostics"], index=False)
    result.score_decomposition.to_csv(paths["score_decomposition"], index=False)
    result.pair_decomposition.to_csv(paths["pair_decomposition"], index=False)
    result.pair_summary.to_csv(paths["pair_summary"], index=False)
    result.influential_dates.to_csv(paths["influential_dates"], index=False)
    result.alternative_scores.to_csv(paths["alternative_scores"], index=False)
    result.structural_diagnostics.to_csv(paths["structural_diagnostics"], index=False)
    result.covariance_sensitivity.to_csv(paths["covariance_sensitivity"], index=False)
    result.tuning.candidates.to_csv(paths["tuning"], index=False)
    result.mean_within.to_csv(paths["mean_within"], index=False)
    result.bootstrap_draws.to_csv(paths["bootstrap_draws"], index=False)
    paths["metadata"].write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    return paths


def dataclass_to_jsonable(obj: object) -> object:
    """Convert dataclasses with arrays omitted to metadata-friendly dictionaries."""
    if hasattr(obj, "__dataclass_fields__"):
        raw = asdict(obj)
        return {
            k: ("array omitted" if isinstance(v, np.ndarray) else v)
            for k, v in raw.items()
            if k not in {"x", "y", "C_ref", "D_ref", "L_ref", "theta_hat"}
        }
    return obj
