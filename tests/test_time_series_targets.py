from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from time_series_targets import (  # noqa: E402
    check_psd_symmetric,
    make_bartlett_filtered_scores,
    make_diagonal_old_target,
    make_hac_filtered_target,
    make_time_weights,
    make_volterra_nonlinear_target,
    make_volterra_nonlinear_weights,
    make_volterra_signature_features,
    relative_geometry_from_target,
)


def _scores(seed: int = 123, n: int = 48, p: int = 8) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 3))
    loadings = rng.normal(size=(3, p))
    psi = factors @ loadings + 0.2 * rng.normal(size=(n, p))
    dates = pd.Series(pd.date_range("2000-01-01", periods=n, freq="MS"))
    return psi, dates


def test_hac_target_is_symmetric_psd() -> None:
    psi, dates = _scores()
    target = make_hac_filtered_target(psi, dates, L=4, time_bandwidth=6.0, kernel="gaussian")
    k_diag = check_psd_symmetric(target.K_by_state)
    c_diag = check_psd_symmetric(target.C_ref)
    assert k_diag["ok"]
    assert c_diag["ok"]
    assert np.allclose(target.weights.sum(axis=1), 1.0)


def test_grid_induced_reference_normalizes_tau_soft() -> None:
    psi, dates = _scores(n=56, p=10)
    target = make_volterra_nonlinear_target(
        psi,
        dates,
        L=3,
        r=3,
        level=2,
        half_lives=(3, 12),
        time_bandwidth=5.0,
        kernel="gaussian",
        reference="grid_or_empirical",
    )
    geom = relative_geometry_from_target(target.K_by_state, target.C_ref)
    q = np.full(len(geom["tau_soft"]), 1.0 / len(geom["tau_soft"]))
    assert abs(float(q @ geom["tau_soft"]) - 1.0) < 1e-10


def test_bartlett_filtered_scores_equal_direct_double_sum() -> None:
    rng = np.random.default_rng(44)
    psi = rng.normal(size=(22, 5))
    L = 4
    Z, valid, _ = make_bartlett_filtered_scores(psi, L=L)
    left = (Z.T @ Z) / len(Z)
    right = np.zeros((psi.shape[1], psi.shape[1]))
    for t in valid:
        accum = np.zeros_like(right)
        for ell in range(L + 1):
            for m in range(L + 1):
                accum += np.outer(psi[t - ell], psi[t - m]) / (L + 1)
        right += accum
    right /= len(valid)
    assert np.allclose(left, right, atol=1e-12)

    # For lag h, there are L+1-h ordered same-orientation pairs in the
    # expansion, giving the Bartlett coefficient 1 - h/(L+1) on
    # Gamma_h + Gamma_h' when grouped by lag.
    weights = np.array([(L + 1 - h) / (L + 1) for h in range(L + 1)])
    expected = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    assert np.allclose(weights, expected)


def test_volterra_features_and_weights_are_well_behaved() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(size=(40, 4))
    dates = pd.Series(pd.date_range("2010-01-01", periods=len(x), freq="MS"))
    phi1, meta1 = make_volterra_signature_features(x, half_lives=(3, 12), level=1)
    phi2, meta2 = make_volterra_signature_features(x, half_lives=(3, 12), level=2)
    assert np.isfinite(phi1).all()
    assert np.isfinite(phi2).all()
    assert meta2["raw_feature_dim"] > meta1["raw_feature_dim"]
    W1, _ = make_volterra_nonlinear_weights(dates, phi1, time_bandwidth=6.0, kernel="gaussian")
    W2, _ = make_volterra_nonlinear_weights(dates, phi2, time_bandwidth=6.0, kernel="gaussian")
    assert np.all(W1 >= 0.0)
    assert np.all(W2 >= 0.0)
    assert np.allclose(W1.sum(axis=1), 1.0)
    assert np.allclose(W2.sum(axis=1), 1.0)
    assert not np.allclose(W1, W2)

    psi, score_dates = _scores(n=40, p=12)
    target = make_volterra_nonlinear_target(
        psi,
        score_dates,
        L=3,
        r=4,
        level=2,
        half_lives=(3, 12),
        time_bandwidth=6.0,
        kernel="gaussian",
    )
    assert check_psd_symmetric(target.K_by_state)["ok"]


def test_old_diagonal_target_matches_outer_product_formula() -> None:
    psi, dates = _scores(n=20, p=6)
    W = make_time_weights(dates, bandwidth=4.0, kernel="gaussian")
    target = make_diagonal_old_target(psi, dates, time_bandwidth=4.0, kernel="gaussian", reference="empirical")
    expected = np.einsum("st,ti,tj->sij", W, psi, psi, optimize=True)
    expected = 0.5 * (expected + np.swapaxes(expected, 1, 2))
    assert np.allclose(target.K_by_state, expected)
