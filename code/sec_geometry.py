"""Graph-Laplacian and SEC feature utilities for the OVK robustness check."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphGeometry:
    """Container for a self-tuning kNN graph and its Laplacian geometry."""

    states: np.ndarray
    k: int
    alpha_density: float
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    affinity: np.ndarray
    normalized_affinity: np.ndarray
    degree: np.ndarray
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray


def default_knn_k(n_obs: int) -> int:
    """Return the default k-nearest-neighbor count from the SEC spec."""
    if n_obs < 3:
        raise ValueError("At least three observations are required for graph geometry.")
    return int(min(n_obs - 1, min(30, max(10, np.floor(np.sqrt(n_obs))))))


def candidate_knn_values(n_obs: int) -> list[int]:
    """Return a compact, deterministic set of k candidates around the default."""
    base = default_knn_k(n_obs)
    vals = {base, max(2, base - 5), min(n_obs - 1, base + 5)}
    vals.add(min(n_obs - 1, max(10, int(np.floor(np.sqrt(n_obs))))))
    return sorted(v for v in vals if 1 <= v < n_obs)


def _orient_eigenfunctions(phi: np.ndarray) -> np.ndarray:
    out = phi.copy()
    if out[0, 0] < 0:
        out[:, 0] *= -1.0
    for j in range(1, out.shape[1]):
        idx = int(np.argmax(np.abs(out[:, j])))
        if out[idx, j] < 0:
            out[:, j] *= -1.0
    return out


def build_graph_laplacian(
    states: np.ndarray,
    k: int | None = None,
    alpha_density: float = 0.5,
    l_max: int = 30,
) -> GraphGeometry:
    """Build a self-tuning kNN graph and symmetric normalized Laplacian.

    The stored eigenfunctions are diffusion eigenfunctions
    ``D^{-1/2} u_j`` derived from the symmetric normalized eigenvectors
    ``u_j``. With this convention the first eigenfunction is constant on
    connected components, matching the scalar SEC basis used downstream.
    """
    S = np.asarray(states, dtype=float)
    if S.ndim != 2 or not np.isfinite(S).all():
        raise ValueError("states must be a finite two-dimensional array")
    n_obs = S.shape[0]
    if k is None:
        k = default_knn_k(n_obs)
    k = int(min(max(1, k), n_obs - 1))
    if alpha_density not in {0.0, 0.5, 1.0}:
        raise ValueError("alpha_density must be one of 0.0, 0.5, or 1.0")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(S)
    distances, indices = nn.kneighbors(S)
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]
    sigma = np.maximum(neighbor_distances[:, -1], 1e-12)

    W = np.zeros((n_obs, n_obs), dtype=float)
    for i in range(n_obs):
        js = neighbor_indices[i]
        ds = neighbor_distances[i]
        denom = np.maximum(sigma[i] * sigma[js], 1e-12)
        W[i, js] = np.exp(-(ds**2) / denom)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    q = np.maximum(W.sum(axis=1), 1e-12)
    q_scale = q ** (-float(alpha_density))
    W_alpha = (q_scale[:, None] * W) * q_scale[None, :]
    W_alpha = 0.5 * (W_alpha + W_alpha.T)
    degree = np.maximum(W_alpha.sum(axis=1), 1e-12)
    inv_sqrt_d = 1.0 / np.sqrt(degree)
    L_sym = np.eye(n_obs) - (inv_sqrt_d[:, None] * W_alpha) * inv_sqrt_d[None, :]
    L_sym = 0.5 * (L_sym + L_sym.T)

    n_keep = int(min(n_obs, max(2, l_max + 1)))
    vals, vecs = eigh(L_sym, subset_by_index=[0, n_keep - 1])
    vals = np.maximum(vals, 0.0)
    phi = inv_sqrt_d[:, None] * vecs
    norms = np.sqrt(np.maximum(np.mean(phi**2, axis=0), 1e-24))
    phi = phi / norms[None, :]
    phi = _orient_eigenfunctions(phi)
    if np.nanstd(phi[:, 0]) > 1e-7:
        # Disconnected or nearly disconnected graphs can rotate the null space.
        # Pin the first basis function to the constant component for features.
        phi[:, 0] = 1.0
    else:
        phi[:, 0] = np.sign(np.nanmean(phi[:, 0]) or 1.0)

    return GraphGeometry(
        states=S,
        k=k,
        alpha_density=float(alpha_density),
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        affinity=W,
        normalized_affinity=W_alpha,
        degree=degree,
        laplacian=L_sym,
        eigenvalues=vals,
        eigenfunctions=phi,
    )


def scalar_features(eigenfunctions: np.ndarray, L: int) -> np.ndarray:
    """Return ``[1, phi_1, ..., phi_L]`` scalar SEC features."""
    phi = np.asarray(eigenfunctions, dtype=float)
    if L < 1:
        raise ValueError("L must be positive")
    if phi.shape[1] < L + 1:
        raise ValueError(f"Need at least {L + 1} eigenfunctions including the trivial one")
    return np.column_stack([np.ones(phi.shape[0]), phi[:, 1 : L + 1]])


def local_eigenfunction_gradients(
    states: np.ndarray,
    eigenfunctions: np.ndarray,
    neighbor_indices: np.ndarray,
    affinity: np.ndarray,
    L: int,
    ridge_scale: float = 1e-6,
) -> np.ndarray:
    """Approximate local gradients of nontrivial eigenfunctions.

    Returns an array with shape ``(T, L, d_state)`` for
    ``phi_1, ..., phi_L``.
    """
    S = np.asarray(states, dtype=float)
    phi = np.asarray(eigenfunctions, dtype=float)
    n_obs, d_state = S.shape
    if phi.shape[0] != n_obs:
        raise ValueError("states and eigenfunctions must have the same row count")
    L = int(min(L, phi.shape[1] - 1))
    grads = np.zeros((n_obs, L, d_state), dtype=float)
    eye = np.eye(d_state)
    for t in range(n_obs):
        nbr = np.asarray(neighbor_indices[t], dtype=int)
        X = S[nbr] - S[t]
        w = np.maximum(np.asarray(affinity[t, nbr], dtype=float), 1e-12)
        Xw = X * np.sqrt(w)[:, None]
        gram = Xw.T @ Xw
        ridge = ridge_scale * float(np.trace(gram) / max(d_state, 1))
        if not np.isfinite(ridge) or ridge <= 0:
            ridge = ridge_scale
        lhs = gram + ridge * eye
        for j in range(L):
            y = phi[nbr, j + 1] - phi[t, j + 1]
            rhs = Xw.T @ (y * np.sqrt(w))
            try:
                grads[t, j] = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                grads[t, j] = np.linalg.pinv(lhs) @ rhs
    if not np.isfinite(grads).all():
        raise FloatingPointError("Non-finite local SEC gradients")
    return grads


def directional_features(
    states: np.ndarray,
    eigenfunctions: np.ndarray,
    gradients: np.ndarray,
    eigenvalues: np.ndarray,
    L: int,
    max_pairs: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build SEC directional features along observed macro-state motion."""
    S = np.asarray(states, dtype=float)
    phi = np.asarray(eigenfunctions, dtype=float)
    grads = np.asarray(gradients, dtype=float)
    L = int(min(L, phi.shape[1] - 1, grads.shape[1]))
    max_pairs = int(max(1, max_pairs))
    phi_ext = np.column_stack([np.ones(S.shape[0]), phi[:, 1 : L + 1]])
    lam = np.asarray(eigenvalues[: L + 1], dtype=float)
    if len(lam) < L + 1:
        lam = np.pad(lam, (0, L + 1 - len(lam)), constant_values=np.nan)

    pairs: list[tuple[float, int, int]] = []
    for i in range(L + 1):
        for j in range(1, L + 1):
            pairs.append((float(lam[i] + lam[j]), i, j))
    pairs = sorted(pairs, key=lambda x: (x[0], x[1], x[2]))[:max_pairs]

    n_rows = S.shape[0] - 1
    Xi = np.zeros((n_rows, len(pairs)), dtype=float)
    delta = S[1:] - S[:-1]
    rows = []
    for c, (energy, i, j) in enumerate(pairs):
        direction = np.einsum("td,td->t", grads[:-1, j - 1, :], delta, optimize=True)
        Xi[:, c] = phi_ext[:-1, i] * direction
        rows.append(
            {
                "feature": f"xi_phi{i}_dphi{j}",
                "phi_index": i,
                "gradient_phi_index": j,
                "laplacian_energy": energy,
            }
        )
    if not np.isfinite(Xi).all():
        raise FloatingPointError("Non-finite directional SEC features")
    return Xi, pd.DataFrame(rows)


def eigenfunction_frame(
    dates: Iterable[pd.Timestamp],
    shock_definition: str,
    eigenfunctions: np.ndarray,
    L: int,
) -> pd.DataFrame:
    """Return a tidy eigenfunction frame suitable for CSV output."""
    date_values = pd.to_datetime(list(dates)).strftime("%Y-%m-%d")
    L = int(min(L, eigenfunctions.shape[1] - 1))
    data: dict[str, object] = {
        "date": date_values,
        "shock_definition": shock_definition,
        "phi_0": eigenfunctions[:, 0],
    }
    for j in range(1, L + 1):
        data[f"phi_{j}"] = eigenfunctions[:, j]
    return pd.DataFrame(data)
