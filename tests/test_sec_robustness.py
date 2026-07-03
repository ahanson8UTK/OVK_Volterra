"""Basic numerical tests for the SEC robustness modules.

Run from the repository root:

    python tests/test_sec_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from sec_geometry import build_graph_laplacian, local_eigenfunction_gradients, scalar_features
from sec_ovk import (
    A_from_log_predictions,
    build_macro_state_matrix,
    centered_kernel,
    construct_rank5_kernel_mean,
    gtilde_and_log_observations,
    matrix_exp_batch,
    matrix_log_batch,
    smat_batch,
    svec_batch,
    top_eigendecomposition,
    whitened_scores,
)
from sec_comparisons import principal_angles


def synthetic_panel(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(20260602)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "ip": 100 + np.cumsum(0.3 + 0.05 * rng.standard_normal(n)),
            "cpi": 100 + np.cumsum(0.2 + 0.03 * rng.standard_normal(n)),
            "unrate": 5 + 0.2 * rng.standard_normal(n),
            "gs2": 2 + 0.1 * rng.standard_normal(n),
            "baa10y": 1 + 0.1 * rng.standard_normal(n),
        }
    )


def test_sec_numerics() -> None:
    rng = np.random.default_rng(20260602)
    T, M, R = 72, 18, 5
    latent = rng.standard_normal((T, R))
    loadings, _ = np.linalg.qr(rng.standard_normal((M, R)))
    Q = latent @ np.diag([4.0, 3.0, 2.0, 1.2, 0.8]) @ loadings.T + 0.05 * rng.standard_normal((T, M))
    K, _, E = centered_kernel(Q)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(K)) > -1e-10

    eigvals, V, _ = top_eigendecomposition(K, R)
    assert np.allclose(V.T @ V, np.eye(R), atol=1e-10)
    Z = whitened_scores(E, V, eigvals)
    assert np.linalg.norm((Z.T @ Z) / T - np.eye(R), ord="fro") < 1e-6

    G, Gtilde, y = gtilde_and_log_observations(Z, 0.25)
    assert np.min(np.linalg.eigvalsh(G)) > 0
    logs = matrix_log_batch(Gtilde)
    exp_logs = matrix_exp_batch(logs)
    assert np.allclose(exp_logs, Gtilde, atol=1e-8)

    states = rng.standard_normal((T, 7))
    geom = build_graph_laplacian(states, k=10, alpha_density=0.5, l_max=10)
    assert np.allclose(geom.affinity, geom.affinity.T, atol=1e-12)
    assert np.min(geom.affinity) >= -1e-12
    assert np.min(geom.eigenvalues) >= -1e-10
    assert np.std(geom.eigenfunctions[:, 0]) < 1e-7
    grads = local_eigenfunction_gradients(states, geom.eigenfunctions, geom.neighbor_indices, geom.affinity, L=5)
    assert grads.shape == (T, 5, 7)
    assert np.isfinite(grads).all()

    Phi = scalar_features(geom.eigenfunctions, 5)
    B = np.linalg.lstsq(Phi, y, rcond=None)[0]
    A, tau = A_from_log_predictions(Phi @ B, R)
    assert np.min(np.linalg.eigvalsh(A)) > 0
    assert np.linalg.norm(A.mean(axis=0) - np.eye(R), ord="fro") < 1e-8
    K_rank = construct_rank5_kernel_mean(V, eigvals, A)
    K_retained = V @ np.diag(eigvals[:R]) @ V.T
    assert np.allclose(K_rank, K_retained, atol=1e-7)
    angles = principal_angles(V, V)
    assert np.all((angles >= -1e-10) & (angles <= 90 + 1e-10))
    assert np.all(np.isfinite(tau)) and np.min(tau) > 0

    panel = synthetic_panel()
    valid_idx = np.arange(20, 60)
    state_df, S = build_macro_state_matrix(panel, valid_idx, "synthetic")
    assert S.shape == (len(valid_idx), 7)
    assert (pd.to_datetime(state_df["state_source_date"]) < pd.to_datetime(state_df["date"])).all()


if __name__ == "__main__":
    test_sec_numerics()
    print("SEC robustness numerical tests passed.")
