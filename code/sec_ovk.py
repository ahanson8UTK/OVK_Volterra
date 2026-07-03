"""SEC-smoothed mean-kernel and state-conditioned OVK estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from ovk_data import DEFAULT_OUTCOME_LABELS


RANDOM_SEED = 20260602
OUTCOME_LABELS = list(DEFAULT_OUTCOME_LABELS)


@dataclass
class RidgeCVResult:
    """Result from blocked cross-validated ridge regression."""

    lambda_value: float
    coefficients: np.ndarray
    fitted: np.ndarray
    cv_loss: float
    records: pd.DataFrame
    effective_df: float
    smoothness_penalty: float


@dataclass
class SECLevelFit:
    """Fitted SEC log-kernel level model."""

    yhat: np.ndarray
    A: np.ndarray
    tau: np.ndarray
    ridge: RidgeCVResult
    min_eig: float
    max_trace_over_rank: float
    mean_A_error: float


def sym(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a matrix."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def spd_eigh(A: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetric positive-definite matrix with clipping."""
    vals, vecs = np.linalg.eigh(sym(np.asarray(A, float)))
    vals = np.maximum(vals, eps)
    return vals, vecs


def spd_invsqrt(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the inverse square root of an SPD matrix."""
    vals, vecs = spd_eigh(A, eps=eps)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def spd_log(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the matrix logarithm of an SPD matrix."""
    vals, vecs = spd_eigh(A, eps=eps)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def spd_exp(S: np.ndarray) -> np.ndarray:
    """Return the matrix exponential of a symmetric matrix."""
    vals, vecs = np.linalg.eigh(sym(np.asarray(S, float)))
    vals = np.clip(vals, -50.0, 50.0)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T


def svec(S: np.ndarray) -> np.ndarray:
    """Half-vectorize a symmetric matrix with diagonal entries first."""
    R = S.shape[0]
    vals = [S[i, i] for i in range(R)]
    vals.extend(S[i, j] for i in range(R) for j in range(i + 1, R))
    return np.asarray(vals, dtype=float)


def smat(v: np.ndarray, R: int) -> np.ndarray:
    """Invert :func:`svec` for an ``R`` by ``R`` symmetric matrix."""
    v = np.asarray(v, dtype=float)
    S = np.zeros((R, R), dtype=float)
    k = 0
    for i in range(R):
        S[i, i] = v[k]
        k += 1
    for i in range(R):
        for j in range(i + 1, R):
            S[i, j] = S[j, i] = v[k]
            k += 1
    return S


def svec_batch(mats: np.ndarray) -> np.ndarray:
    """Half-vectorize a batch of symmetric matrices."""
    return np.vstack([svec(m) for m in np.asarray(mats)])


def smat_batch(values: np.ndarray, R: int) -> np.ndarray:
    """Invert :func:`svec_batch`."""
    return np.stack([smat(v, R) for v in np.asarray(values)], axis=0)


def matrix_log_batch(mats: np.ndarray) -> np.ndarray:
    """Apply the SPD matrix logarithm to a batch."""
    return np.stack([spd_log(m) for m in np.asarray(mats)], axis=0)


def matrix_exp_batch(mats: np.ndarray) -> np.ndarray:
    """Apply the symmetric matrix exponential to a batch."""
    return np.stack([spd_exp(m) for m in np.asarray(mats)], axis=0)


def centered_kernel(Q_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``K_bar``, mean score surface, and centered score surfaces."""
    Q = np.asarray(Q_scores, dtype=float)
    beta = Q.mean(axis=0)
    E = Q - beta
    K = sym((E.T @ E) / len(E))
    return K, beta, E


def top_eigendecomposition(K: np.ndarray, rank: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted eigenvalues, leading eigenvectors, and trace shares."""
    vals, vecs = eigh(sym(K))
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    for j in range(vecs.shape[1]):
        idx = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[idx, j] < 0:
            vecs[:, j] *= -1.0
    trace = max(float(vals.sum()), 1e-12)
    return vals, vecs[:, :rank], vals / trace


def whitened_scores(E: np.ndarray, V: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    """Project centered score surfaces into whitened retained factors."""
    lam = np.maximum(np.asarray(eigvals[: V.shape[1]], dtype=float), 1e-12)
    return np.asarray(E) @ V @ np.diag(1.0 / np.sqrt(lam))


def gtilde_and_log_observations(Z: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct SPD proxies, mean-normalized proxies, and log observations."""
    Z = np.asarray(Z, dtype=float)
    T, R = Z.shape
    I = np.eye(R)
    G = alpha * I[None, :, :] + (1.0 - alpha) * np.einsum("ti,tj->tij", Z, Z, optimize=True)
    Gbar_inv = spd_invsqrt(G.mean(axis=0), eps=1e-10)
    Gtilde = np.einsum("ij,tjk,kl->til", Gbar_inv, G, Gbar_inv, optimize=True)
    Gtilde = sym(Gtilde)
    y = svec_batch(matrix_log_batch(Gtilde))
    return G, Gtilde, y


def blocked_folds(n_obs: int, n_folds: int | None = None) -> list[np.ndarray]:
    """Return contiguous time-block folds."""
    if n_folds is None:
        n_folds = 5 if n_obs >= 100 else 4
    n_folds = int(min(max(2, n_folds), n_obs))
    edges = np.linspace(0, n_obs, n_folds + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_folds) if edges[i] < edges[i + 1]]


def penalty_vector(n_features: int, eigenvalues: Iterable[float] | None = None) -> np.ndarray:
    """Build a ridge penalty vector that does not penalize the intercept."""
    pen = np.ones(n_features, dtype=float)
    pen[0] = 0.0
    if eigenvalues is not None:
        vals = np.asarray(list(eigenvalues), dtype=float)
        m = min(len(vals), n_features - 1)
        if m:
            pen[1 : m + 1] = np.maximum(vals[:m], 1e-8)
    return pen


def ridge_fit(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_value: float,
    penalties: np.ndarray | None = None,
) -> np.ndarray:
    """Fit multi-output ridge regression and return coefficients."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if penalties is None:
        penalties = penalty_vector(X.shape[1])
    reg = float(lambda_value) * np.diag(np.asarray(penalties, dtype=float))
    lhs = X.T @ X + reg
    rhs = X.T @ Y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def effective_degrees_of_freedom(X: np.ndarray, lambda_value: float, penalties: np.ndarray) -> float:
    """Return ridge effective degrees of freedom for a design matrix."""
    lhs = X.T @ X + float(lambda_value) * np.diag(penalties)
    try:
        H = X @ np.linalg.solve(lhs, X.T)
    except np.linalg.LinAlgError:
        H = X @ np.linalg.pinv(lhs) @ X.T
    return float(np.trace(H))


def ridge_smoothness(B: np.ndarray, eigenvalues: Iterable[float] | None = None) -> float:
    """Compute ``trace(B' Lambda_SEC B)`` excluding the intercept."""
    if B.shape[0] <= 1:
        return 0.0
    vals = np.asarray(list(eigenvalues) if eigenvalues is not None else [], dtype=float)
    if len(vals) < B.shape[0] - 1:
        vals = np.pad(vals, (0, B.shape[0] - 1 - len(vals)), constant_values=1.0)
    vals = np.maximum(vals[: B.shape[0] - 1], 0.0)
    return float(np.sum(vals[:, None] * (B[1:] ** 2)))


def evaluate_ridge_grid(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_grid: Iterable[float],
    folds: list[np.ndarray] | None = None,
    eigenvalues: Iterable[float] | None = None,
    label: str = "level",
) -> pd.DataFrame:
    """Evaluate blocked-CV ridge losses over a lambda grid."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if folds is None:
        folds = blocked_folds(len(X))
    penalties = penalty_vector(X.shape[1], eigenvalues)
    rows = []
    all_idx = np.arange(len(X))
    for lam in lambda_grid:
        fold_losses = []
        singular = False
        for fold_id, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
            Xtr, Ytr = X[train_idx], Y[train_idx]
            Xva, Yva = X[val_idx], Y[val_idx]
            cond = np.linalg.cond(Xtr.T @ Xtr + float(lam) * np.diag(penalties))
            if not np.isfinite(cond):
                singular = True
            B = ridge_fit(Xtr, Ytr, float(lam), penalties)
            pred = Xva @ B
            fold_losses.append(float(np.mean((Yva - pred) ** 2)))
        B_all = ridge_fit(X, Y, float(lam), penalties)
        rows.append(
            {
                "model": label,
                "ridge_lambda": float(lam),
                "cv_loss": float(np.mean(fold_losses)),
                "cv_loss_sd": float(np.std(fold_losses, ddof=0)),
                "smoothness_penalty": ridge_smoothness(B_all, eigenvalues),
                "effective_degrees_of_freedom": effective_degrees_of_freedom(X, float(lam), penalties),
                "singular_design": bool(singular),
                "max_design_condition": float(np.linalg.cond(X.T @ X + float(lam) * np.diag(penalties))),
            }
        )
    return pd.DataFrame(rows)


def fit_ridge_cv(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_grid: Iterable[float],
    folds: list[np.ndarray] | None = None,
    eigenvalues: Iterable[float] | None = None,
    label: str = "level",
) -> RidgeCVResult:
    """Select and fit a blocked-CV ridge model."""
    records = evaluate_ridge_grid(X, Y, lambda_grid, folds=folds, eigenvalues=eigenvalues, label=label)
    valid = records[~records["singular_design"]].copy()
    if valid.empty:
        valid = records.copy()
    best_row = valid.sort_values(["cv_loss", "ridge_lambda"]).iloc[0]
    penalties = penalty_vector(np.asarray(X).shape[1], eigenvalues)
    B = ridge_fit(X, Y, float(best_row["ridge_lambda"]), penalties)
    fitted = np.asarray(X) @ B
    return RidgeCVResult(
        lambda_value=float(best_row["ridge_lambda"]),
        coefficients=B,
        fitted=fitted,
        cv_loss=float(best_row["cv_loss"]),
        records=records,
        effective_df=float(best_row["effective_degrees_of_freedom"]),
        smoothness_penalty=float(best_row["smoothness_penalty"]),
    )


def filtered_score_kernel(
    E: np.ndarray,
    Phi: np.ndarray,
    lambda_grid: Iterable[float],
    eigenvalues: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, RidgeCVResult]:
    """Estimate the SEC-filtered score-surface mean kernel."""
    fit = fit_ridge_cv(Phi, E, lambda_grid, eigenvalues=eigenvalues, label="score_kernel")
    Ehat = fit.fitted
    K_sec = sym((Ehat.T @ Ehat) / len(Ehat))
    return K_sec, Ehat, fit


def A_from_log_predictions(yhat: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Map fitted log-state vectors to normalized SPD ``A_SEC(S_t)``."""
    mats = matrix_exp_batch(smat_batch(yhat, rank))
    C_inv = spd_invsqrt(mats.mean(axis=0), eps=1e-10)
    A = np.einsum("ij,tjk,kl->til", C_inv, mats, C_inv, optimize=True)
    A = sym(A)
    tau = np.trace(A, axis1=1, axis2=2) / rank
    return A, tau


def fit_sec_level_model(
    y: np.ndarray,
    Phi: np.ndarray,
    rank: int,
    lambda_grid: Iterable[float],
    eigenvalues: Iterable[float],
    folds: list[np.ndarray] | None = None,
) -> SECLevelFit:
    """Fit the primary SEC state-conditioned log-kernel level model."""
    ridge = fit_ridge_cv(Phi, y, lambda_grid, folds=folds, eigenvalues=eigenvalues, label="sec_level")
    A, tau = A_from_log_predictions(ridge.fitted, rank)
    eigs = np.linalg.eigvalsh(A)
    mean_A_error = float(np.linalg.norm(A.mean(axis=0) - np.eye(rank), ord="fro"))
    return SECLevelFit(
        yhat=ridge.fitted,
        A=A,
        tau=tau,
        ridge=ridge,
        min_eig=float(np.min(eigs)),
        max_trace_over_rank=float(np.max(tau)),
        mean_A_error=mean_A_error,
    )


def construct_rank5_kernel_mean(V: np.ndarray, eigvals: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return the sample mean of ``V Lambda^(1/2) A_t Lambda^(1/2) V'``."""
    lam_half = np.diag(np.sqrt(np.maximum(np.asarray(eigvals[: V.shape[1]], dtype=float), 1e-12)))
    core = np.einsum("ij,tjk,kl->til", lam_half, A, lam_half, optimize=True)
    Kt = np.einsum("mi,tij,nj->tmn", V, core, V, optimize=True)
    return sym(Kt.mean(axis=0))


def covariance_proxy_loss(Gtilde: np.ndarray, A: np.ndarray) -> float:
    """Return mean Frobenius loss between normalized proxies and fitted A."""
    diff = np.asarray(Gtilde) - np.asarray(A)
    return float(np.mean(np.sum(diff * diff, axis=(1, 2))))


def log_proxy_loss(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return mean squared log-proxy prediction loss."""
    diff = np.asarray(y) - np.asarray(yhat)
    return float(np.mean(diff * diff))


def fit_directional_drift_model(
    y: np.ndarray,
    Xi: np.ndarray,
    lambda_grid: Iterable[float],
    folds: list[np.ndarray] | None = None,
) -> tuple[RidgeCVResult, np.ndarray]:
    """Fit the SEC directional drift diagnostic ``Delta y_t = C'Xi_t + u_t``."""
    dy = np.asarray(y[1:] - y[:-1], dtype=float)
    X = np.column_stack([np.ones(len(Xi)), np.asarray(Xi, dtype=float)])
    if folds is None:
        folds = blocked_folds(len(X))
    fit = fit_ridge_cv(X, dy, lambda_grid, folds=folds, label="sec_directional_drift")
    return fit, dy


def var1_oos_loss(y: np.ndarray, lambda_grid: Iterable[float] | None = None) -> float:
    """Blocked-CV loss for a simple log-Euclidean VAR(1) baseline proxy."""
    Y = np.asarray(y, dtype=float)
    if lambda_grid is None:
        lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    X = np.column_stack([np.ones(len(Y) - 1), Y[:-1]])
    target = Y[1:]
    folds = blocked_folds(len(target))
    return fit_ridge_cv(X, target, lambda_grid, folds=folds, label="publication_grade_log_var_proxy").cv_loss


def build_macro_state_matrix(
    panel: pd.DataFrame,
    valid_idx: np.ndarray,
    shock_definition: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build predetermined macro-financial states using only t-1 or earlier data."""
    p = panel.sort_values("date").reset_index(drop=True).copy()
    p["date"] = pd.to_datetime(p["date"])
    ip = p["ip"].astype(float).to_numpy()
    cpi = p["cpi"].astype(float).to_numpy()
    unrate = p["unrate"].astype(float).to_numpy()
    gs2 = p["gs2"].astype(float).to_numpy()
    baa10y = p["baa10y"].astype(float).to_numpy()
    cpi10 = p["cpi10"].astype(float).to_numpy() if "cpi10" in p.columns else None
    expinf5yr = p["expinf5yr"].astype(float).to_numpy() if "expinf5yr" in p.columns else None
    mich = p["mich"].astype(float).to_numpy() if "mich" in p.columns else None
    state_variables = state_variables_for_panel(p)

    rows = []
    for idx in np.asarray(valid_idx, dtype=int):
        src = idx - 1
        row = {
            "date": p.loc[idx, "date"],
            "shock_definition": shock_definition,
            "state_source_date": p.loc[src, "date"] if src >= 0 else pd.NaT,
            "valid_sample": True,
        }
        try:
            if src < 12:
                raise IndexError("Need at least 12 lagged months for SEC states")
            row["ip_growth_12m_lag1"] = 100.0 * np.log(ip[src] / ip[src - 12])
            row["cpi_inflation_12m_lag1"] = 100.0 * np.log(cpi[src] / cpi[src - 12])
            row["unrate_lag1"] = unrate[src]
            row["gs2_lag1"] = gs2[src]
            row["baa10y_lag1"] = baa10y[src]
            row["gs2_change_3m_lag1"] = gs2[src] - gs2[src - 3]
            row["baa10y_change_3m_lag1"] = baa10y[src] - baa10y[src - 3]
            if cpi10 is not None:
                row["cpi10_lag1"] = cpi10[src]
                row["cpi10_change_3m_lag1"] = cpi10[src] - cpi10[src - 3]
            if expinf5yr is not None:
                row["expinf5yr_lag1"] = expinf5yr[src]
                row["expinf5yr_change_3m_lag1"] = expinf5yr[src] - expinf5yr[src - 3]
            if mich is not None:
                row["mich_lag1"] = mich[src]
                row["mich_change_3m_lag1"] = mich[src] - mich[src - 3]
        except Exception:
            for name in state_variables:
                row[name] = np.nan
            row["valid_sample"] = False
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in state_variables:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["valid_sample"] = df["valid_sample"] & df[state_variables].notna().all(axis=1)
    if not bool(df["valid_sample"].all()):
        bad = int((~df["valid_sample"]).sum())
        raise ValueError(f"{bad} SEC state rows are invalid; cannot preserve LP alignment")
    raw = df[state_variables].to_numpy(float)
    mu = raw.mean(axis=0)
    sd = raw.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    S = (raw - mu) / sd
    for i, col in enumerate(state_variables):
        df[f"{col}_standardized"] = S[:, i]
    if not np.isfinite(S).all():
        raise FloatingPointError("Non-finite standardized SEC state matrix")
    if not (pd.to_datetime(df["state_source_date"]) < pd.to_datetime(df["date"])).all():
        raise AssertionError("SEC state matrix violates the t-1 no-look-ahead rule")
    return df, S


BASE_STATE_VARIABLES = [
    "ip_growth_12m_lag1",
    "cpi_inflation_12m_lag1",
    "unrate_lag1",
    "gs2_lag1",
    "baa10y_lag1",
    "gs2_change_3m_lag1",
    "baa10y_change_3m_lag1",
]


def state_variables_for_panel(panel: pd.DataFrame) -> list[str]:
    variables = list(BASE_STATE_VARIABLES)
    if "cpi10" in panel.columns:
        variables.extend(["cpi10_lag1", "cpi10_change_3m_lag1"])
    if "expinf5yr" in panel.columns:
        variables.extend(["expinf5yr_lag1", "expinf5yr_change_3m_lag1"])
    if "mich" in panel.columns:
        variables.extend(["mich_lag1", "mich_change_3m_lag1"])
    return variables


STATE_VARIABLES = list(BASE_STATE_VARIABLES)
