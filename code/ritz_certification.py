"""Ridge-relative Ritz certification utilities.

The functions in this module are intentionally small and explicit.  They
implement the ridge-relative operator

    A_rho(s) = (C + rho I)^(-1/2) K(s) (C + rho I)^(-1/2)

and the associated finite-rank Ritz compression used by the empirical
certification outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import eigh as scipy_eigh


PSD_ATOL = 1.0e-10
PSD_RTOL = 1.0e-8
ASSERT_TOL = 5.0e-8
RITZ_IDENTITY_TOL = 5.0e-8
CERTIFICATE_TOL = 5.0e-8
GENERALIZED_EIGENPAIR_RESIDUAL_TOL = 1.0e-7


@dataclass(frozen=True)
class SymmetricEigendecomposition:
    values: np.ndarray
    vectors: np.ndarray
    min_raw_eigenvalue: float
    clipped_count: int


@dataclass(frozen=True)
class RidgeOperator:
    C: np.ndarray
    K: np.ndarray
    rho: float
    eigen: SymmetricEigendecomposition
    D: np.ndarray
    D_sqrt: np.ndarray
    D_invsqrt: np.ndarray
    A: np.ndarray


@dataclass(frozen=True)
class RitzCompression:
    V_R: np.ndarray
    lambda_R: np.ndarray
    theta: np.ndarray
    theta_from_K: np.ndarray
    identity_error: float


@dataclass(frozen=True)
class RitzCertificate:
    omitted_mass: float
    retained_mass: float
    residual_norm: float
    residual_term: float
    uncapped_bound: float
    capped_bound: float
    bound_utilization: float

    @property
    def bound(self) -> float:
        """Backward-compatible alias for the capped certificate."""
        return self.capped_bound


@dataclass(frozen=True)
class RitzResidual:
    matrix: np.ndarray
    spectral_norm: float


@dataclass(frozen=True)
class StateApproximationSummary:
    exact_error: float
    exact_stress: float
    compressed_stress: float
    retained_mass: float
    omitted_mass: float
    residual_norm: float
    residual_term: float
    uncapped_bound: float
    capped_bound: float
    bound_utilization: float


@dataclass(frozen=True)
class CertifiedRankResult:
    epsilon: float
    rank: int
    max_bound: float


@dataclass(frozen=True)
class IntervalSpectrumResult:
    eigenvalues: np.ndarray
    eigenvectors_D_normalized: np.ndarray
    whitened_eigenvectors: np.ndarray
    residuals: np.ndarray
    backend: str
    interval: tuple[float, float]


def sym(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a matrix or stack of matrices."""
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def _material_negative_tolerance(values: np.ndarray, atol: float = PSD_ATOL, rtol: float = PSD_RTOL) -> float:
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    return max(float(atol), float(rtol) * scale)


def symmetric_eigendecomposition_psd(
    matrix: np.ndarray,
    *,
    name: str = "matrix",
    atol: float = PSD_ATOL,
    rtol: float = PSD_RTOL,
) -> SymmetricEigendecomposition:
    """Eigen-decompose a symmetric PSD matrix with informative checks.

    Tiny negative eigenvalues are clipped to zero.  Material negative
    eigenvalues raise ``ValueError`` rather than being silently repaired.
    Eigenvalues are returned in descending order.
    """
    mat = sym(np.asarray(matrix, dtype=float))
    vals, vecs = np.linalg.eigh(mat)
    min_raw = float(vals.min()) if vals.size else 0.0
    tol = _material_negative_tolerance(vals, atol=atol, rtol=rtol)
    if min_raw < -tol:
        raise ValueError(f"{name} has a material negative eigenvalue {min_raw:.6g} below tolerance {tol:.6g}.")
    clipped = (vals < 0.0) & (vals >= -tol)
    vals = np.where(clipped, 0.0, vals)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    for j in range(vecs.shape[1]):
        pivot = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[pivot, j] < 0:
            vecs[:, j] *= -1.0
    return SymmetricEigendecomposition(vals, vecs, min_raw, int(clipped.sum()))


def psd_min_eigenvalue(matrix: np.ndarray, *, name: str = "matrix") -> float:
    eig = symmetric_eigendecomposition_psd(matrix, name=name)
    return eig.min_raw_eigenvalue


def matrix_sqrt_from_eigen(eigen: SymmetricEigendecomposition) -> np.ndarray:
    vals = np.sqrt(np.maximum(eigen.values, 0.0))
    return sym((eigen.vectors * vals[None, :]) @ eigen.vectors.T)


def build_ridge_relative_operator(C: np.ndarray, K: np.ndarray, rho: float) -> RidgeOperator:
    """Build ``A_rho = D^(-1/2) K D^(-1/2)`` with ``D = C + rho I``."""
    rho = float(rho)
    if rho <= 0 or not np.isfinite(rho):
        raise ValueError("rho must be positive and finite.")
    C_sym = sym(C)
    K_sym = sym(K)
    C_eig = symmetric_eigendecomposition_psd(C_sym, name="C_hat")
    symmetric_eigendecomposition_psd(K_sym, name="K_hat(s)")
    d_vals = C_eig.values + rho
    if np.min(d_vals) <= 0:
        raise ValueError("C_hat + rho I is not positive definite.")
    D = C_sym + rho * np.eye(C_sym.shape[0])
    D_sqrt = sym((C_eig.vectors * np.sqrt(d_vals)[None, :]) @ C_eig.vectors.T)
    D_invsqrt = sym((C_eig.vectors * (1.0 / np.sqrt(d_vals))[None, :]) @ C_eig.vectors.T)
    A = sym(D_invsqrt @ K_sym @ D_invsqrt)
    symmetric_eigendecomposition_psd(A, name="A_rho(s)")
    return RidgeOperator(C_sym, K_sym, rho, C_eig, D, D_sqrt, D_invsqrt, A)


def build_probe_soft(C: np.ndarray, rho: float) -> np.ndarray:
    """Return the whole-surface soft probe ``C(C+rho I)^(-1)`` normalized to trace one."""
    eig = symmetric_eigendecomposition_psd(C, name="C_hat")
    weights = eig.values / (eig.values + float(rho))
    denom = float(weights.sum())
    if denom <= 0:
        raise ValueError("Soft probe has zero trace; C_hat appears to be zero.")
    Q = (eig.vectors * (weights / denom)[None, :]) @ eig.vectors.T
    return validate_probe(Q, name="Q_soft")


def build_probe_direction(C: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Return ``C^(1/2) g g' C^(1/2)`` normalized to trace one."""
    eig = symmetric_eigendecomposition_psd(C, name="C_hat")
    C_sqrt = matrix_sqrt_from_eigen(eig)
    direction = np.asarray(g, dtype=float).reshape(-1)
    if direction.shape[0] != C_sqrt.shape[0]:
        raise ValueError("Direction length does not match C_hat dimension.")
    x = C_sqrt @ direction
    denom = float(x @ x)
    if denom <= 0:
        raise ValueError("Directional probe has zero C_hat norm.")
    return validate_probe(np.outer(x, x) / denom, name="Q_direction")


def build_probe_block(C: np.ndarray, selector: np.ndarray | Iterable[int] | Iterable[bool]) -> np.ndarray:
    """Return a trace-one block probe for selected coordinates.

    ``selector`` may be a boolean mask, an integer index list, or a dense
    selector matrix with shape ``(n_selected, p)``.
    """
    C_sym = sym(C)
    p = C_sym.shape[0]
    sel = np.asarray(selector)
    if sel.ndim == 1:
        if sel.dtype == bool:
            if sel.shape[0] != p:
                raise ValueError("Boolean selector length does not match C_hat dimension.")
            idx = np.flatnonzero(sel)
        else:
            idx = sel.astype(int)
        if len(idx) == 0:
            raise ValueError("Block selector is empty.")
        mask = np.zeros(p, dtype=float)
        mask[idx] = 1.0
        S_t_S = np.diag(mask)
        denom = float(np.trace(C_sym[np.ix_(idx, idx)]))
    elif sel.ndim == 2:
        if sel.shape[1] != p:
            raise ValueError("Selector matrix must have p columns.")
        S_t_S = sel.T @ sel
        denom = float(np.trace(sel @ C_sym @ sel.T))
    else:
        raise ValueError("selector must be one- or two-dimensional.")
    if denom <= 0:
        raise ValueError("Block probe denominator is non-positive.")
    C_sqrt = matrix_sqrt_from_eigen(symmetric_eigendecomposition_psd(C_sym, name="C_hat"))
    Q = C_sqrt @ S_t_S @ C_sqrt / denom
    return validate_probe(Q, name="Q_block")


def validate_probe(Q: np.ndarray, *, name: str = "Q", tol: float = 5.0e-8) -> np.ndarray:
    Q_sym = sym(Q)
    eig = symmetric_eigendecomposition_psd(Q_sym, name=name, atol=tol, rtol=tol)
    tr = float(np.trace(Q_sym))
    if abs(tr - 1.0) > tol:
        raise ValueError(f"{name} must have trace one; got {tr:.12g}.")
    if eig.values[0] < -tol:
        raise ValueError(f"{name} is not positive semidefinite.")
    return Q_sym


def ritz_compression(
    C_eigenvectors: np.ndarray,
    C_eigenvalues: np.ndarray,
    K: np.ndarray,
    rho: float,
    R: int,
) -> RitzCompression:
    """Compress the same ridge-relative operator into the first ``R`` C-eigenvectors."""
    V = np.asarray(C_eigenvectors, dtype=float)
    lam = np.asarray(C_eigenvalues, dtype=float)
    p = V.shape[0]
    R = int(min(max(R, 1), p))
    V_R = V[:, :R]
    lambda_R = lam[:R]
    scale = 1.0 / np.sqrt(lambda_R + float(rho))
    K_R = V_R.T @ sym(K) @ V_R
    theta_from_K = sym((scale[:, None] * K_R) * scale[None, :])
    D_invsqrt = sym((V * (1.0 / np.sqrt(lam + float(rho)))[None, :]) @ V.T)
    A = sym(D_invsqrt @ sym(K) @ D_invsqrt)
    theta = sym(V_R.T @ A @ V_R)
    identity_error = float(np.max(np.abs(theta - theta_from_K))) if theta.size else 0.0
    return RitzCompression(V_R=V_R, lambda_R=lambda_R, theta=theta, theta_from_K=theta_from_K, identity_error=identity_error)


def _stress_from_eigendecomposition(A: np.ndarray, Q: np.ndarray, lambda_value: float) -> float:
    vals, vecs = np.linalg.eigh(sym(A))
    tol = _material_negative_tolerance(vals)
    if vals.size and float(vals.min()) < -tol:
        raise ValueError(f"Stress operator has a material negative eigenvalue {float(vals.min()):.6g}.")
    vals = np.maximum(vals, 0.0)
    f_vals = vals / (float(lambda_value) + vals)
    q_diag = np.einsum("ij,ji->i", vecs.T @ sym(Q), vecs, optimize=True)
    value = float(np.sum(f_vals * q_diag))
    if value < -ASSERT_TOL or value > 1.0 + ASSERT_TOL:
        raise FloatingPointError(f"Stress statistic outside [0,1]: {value}.")
    return min(1.0, max(0.0, value))


def stress_trace(A: np.ndarray, Q: np.ndarray, lambda_value: float) -> float:
    """Compute ``tr[Q A(lambda I + A)^(-1)]`` stably by eigendecomposition."""
    if lambda_value <= 0:
        raise ValueError("lambda_value must be positive.")
    return _stress_from_eigendecomposition(A, Q, lambda_value)


def compressed_stress(Theta: np.ndarray, V_R: np.ndarray, Q: np.ndarray, lambda_value: float) -> float:
    """Compute the rank-R Ritz stress statistic."""
    Q_R = np.asarray(V_R, dtype=float).T @ sym(Q) @ np.asarray(V_R, dtype=float)
    return _stress_from_eigendecomposition(Theta, Q_R, lambda_value)


def ritz_residual(A: np.ndarray, V_R: np.ndarray, Theta: np.ndarray) -> RitzResidual:
    """Return ``A V_R - V_R Theta`` and its spectral norm."""
    residual = sym(A) @ np.asarray(V_R, dtype=float) - np.asarray(V_R, dtype=float) @ sym(Theta)
    residual_norm = float(np.linalg.svd(residual, compute_uv=False)[0]) if residual.size else 0.0
    return RitzResidual(matrix=residual, spectral_norm=residual_norm)


def ritz_certificate(
    A: np.ndarray,
    V_R: np.ndarray,
    Theta: np.ndarray,
    Q: np.ndarray,
    lambda_value: float,
) -> RitzCertificate:
    """Return the Ritz residual certificate for one state and rank."""
    if lambda_value <= 0:
        raise ValueError("lambda_value must be positive.")
    V = np.asarray(V_R, dtype=float)
    retained_mass = float(np.trace(V.T @ sym(Q) @ V))
    if -ASSERT_TOL <= retained_mass <= 0.0:
        retained_mass = 0.0
    if 1.0 <= retained_mass <= 1.0 + ASSERT_TOL:
        retained_mass = 1.0
    if retained_mass < -ASSERT_TOL or retained_mass > 1.0 + ASSERT_TOL:
        raise FloatingPointError(f"Retained probe mass outside [0,1]: {retained_mass}.")
    omitted_mass = 1.0 - retained_mass
    residual_norm = ritz_residual(A, V, Theta).spectral_norm
    residual_term = residual_norm / float(lambda_value)
    uncapped_bound = max(0.0, omitted_mass) + residual_term
    capped_bound = min(1.0, uncapped_bound)
    return RitzCertificate(
        omitted_mass=max(0.0, min(1.0, omitted_mass)),
        retained_mass=max(0.0, min(1.0, retained_mass)),
        residual_norm=residual_norm,
        residual_term=residual_term,
        uncapped_bound=float(uncapped_bound),
        capped_bound=float(capped_bound),
        bound_utilization=float("nan"),
    )


def summarize_statewise_approximation(
    A: np.ndarray,
    V_R: np.ndarray,
    Theta: np.ndarray,
    Q: np.ndarray,
    lambda_value: float,
) -> StateApproximationSummary:
    """Summarize exact, compressed, and certified stress approximation."""
    exact = stress_trace(A, Q, lambda_value)
    compressed = compressed_stress(Theta, V_R, Q, lambda_value)
    cert = ritz_certificate(A, V_R, Theta, Q, lambda_value)
    return StateApproximationSummary(
        exact_error=abs(exact - compressed),
        exact_stress=exact,
        compressed_stress=compressed,
        retained_mass=cert.retained_mass,
        omitted_mass=cert.omitted_mass,
        residual_norm=cert.residual_norm,
        residual_term=cert.residual_term,
        uncapped_bound=cert.uncapped_bound,
        capped_bound=cert.capped_bound,
        bound_utilization=float(abs(exact - compressed) / cert.uncapped_bound) if cert.uncapped_bound > 0 else 0.0,
    )


def severe_subspace_capture(V_R: np.ndarray, U_star: np.ndarray) -> tuple[float, float]:
    """Return worst and median distance of severe directions to the Ritz subspace."""
    U = np.asarray(U_star, dtype=float)
    if U.size == 0 or U.shape[1] == 0:
        return 0.0, 0.0
    V = np.asarray(V_R, dtype=float)
    projected_norm_sq = np.sum((V.T @ U) ** 2, axis=0)
    distances = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, projected_norm_sq)))
    return float(np.max(distances)), float(np.median(distances))


def certified_rank(
    state_operators: Iterable[np.ndarray],
    V: np.ndarray,
    Q: np.ndarray,
    lambda_value: float,
    epsilon: float,
) -> CertifiedRankResult:
    """Find the smallest rank whose maximum certificate is at most ``epsilon``."""
    ops = [sym(A) for A in state_operators]
    p = V.shape[0]
    best_rank = p
    best_bound = float("inf")
    for R in range(1, p + 1):
        V_R = V[:, :R]
        max_bound = 0.0
        for A in ops:
            theta = sym(V_R.T @ A @ V_R)
            cert = ritz_certificate(A, V_R, theta, Q, lambda_value)
            max_bound = max(max_bound, cert.bound)
        if max_bound < best_bound:
            best_rank = R
            best_bound = max_bound
        if max_bound <= epsilon:
            return CertifiedRankResult(float(epsilon), R, max_bound)
    return CertifiedRankResult(float(epsilon), best_rank, best_bound)


def interval_generalized_spectrum(
    K: np.ndarray,
    D: np.ndarray,
    interval: tuple[float, float],
    backend: str = "auto",
) -> IntervalSpectrumResult:
    """Extract generalized Hermitian eigenpairs in ``interval``.

    The default uses SciPy's generalized Hermitian interval extraction
    (``subset_by_value``), which is the small/medium-dimension analogue of the
    interval backend used in larger FEAST-style implementations.
    """
    lower, upper = float(interval[0]), float(interval[1])
    if lower > upper:
        raise ValueError("interval lower endpoint exceeds upper endpoint.")
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("interval endpoints must be finite.")
    if backend not in {"auto", "scipy_interval", "scipy_subset_by_value"}:
        raise ValueError(f"Unsupported interval backend: {backend}.")
    K_sym = sym(K)
    D_sym = sym(D)
    symmetric_eigendecomposition_psd(K_sym, name="K_hat(s)")
    d_eig = symmetric_eigendecomposition_psd(D_sym, name="D")
    if float(np.min(d_eig.values)) <= 0:
        raise ValueError("D must be positive definite.")

    vals, vecs = scipy_eigh(K_sym, D_sym, subset_by_value=(lower, upper), check_finite=False)
    vals = np.asarray(vals, dtype=float)
    vals_sel = vals
    vecs_sel = vecs
    if vecs_sel.size:
        order = np.argsort(vals_sel)[::-1]
        vals_sel = vals_sel[order]
        vecs_sel = vecs_sel[:, order]
        for j in range(vecs_sel.shape[1]):
            norm = float(np.sqrt(float(vecs_sel[:, j].T @ D_sym @ vecs_sel[:, j])))
            if norm <= 0:
                raise FloatingPointError("Selected generalized eigenvector has non-positive D norm.")
            vecs_sel[:, j] /= norm
    D_sqrt = sym((d_eig.vectors * np.sqrt(d_eig.values)[None, :]) @ d_eig.vectors.T)
    D_invsqrt = sym((d_eig.vectors * (1.0 / np.sqrt(d_eig.values))[None, :]) @ d_eig.vectors.T)
    A = sym(D_invsqrt @ K_sym @ D_invsqrt)
    U = D_sqrt @ vecs_sel if vecs_sel.size else np.empty((D_sym.shape[0], 0))
    residuals = []
    for j, val in enumerate(vals_sel):
        u = U[:, j]
        u_norm = float(np.linalg.norm(u))
        if abs(u_norm - 1.0) > 1.0e-6:
            raise FloatingPointError(f"Whitened generalized eigenvector is not unit length: {u_norm}.")
        eta_a = float(np.linalg.norm(A @ u - val * u))
        eta_w = float(np.linalg.norm(D_invsqrt @ (K_sym @ vecs_sel[:, j] - val * (D_sym @ vecs_sel[:, j]))))
        if abs(eta_a - eta_w) > 1.0e-6 + 1.0e-5 * max(eta_a, eta_w, 1.0):
            raise FloatingPointError("Whitened residual identity failed for generalized eigenpair.")
        residuals.append(max(eta_a, eta_w))
    return IntervalSpectrumResult(
        eigenvalues=vals_sel,
        eigenvectors_D_normalized=vecs_sel,
        whitened_eigenvectors=U,
        residuals=np.asarray(residuals, dtype=float),
        backend="scipy.linalg.eigh subset_by_value",
        interval=(lower, upper),
    )


def severe_direction_count(eigenvalues: np.ndarray, threshold: float) -> int:
    return int(np.sum(np.asarray(eigenvalues, dtype=float) >= float(threshold) - ASSERT_TOL))
