"""Simulation checks for the arithmetic outer-product covariance estimator."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OVK_PUBLICATION_ROOT", str(ROOT / "tmp" / "test_publication_grade_ovk"))
os.environ.setdefault("OVK_REPORTS_DIR", str(ROOT / "tmp" / "test_publication_grade_ovk" / "reports"))
os.environ.setdefault("OVK_DISABLE_CACHE", "1")
os.environ.setdefault("OVK_COVARIANCE_ESTIMATOR_MODE", "arithmetic_outer_product")
sys.path.insert(0, str(ROOT / "code"))

import run_publication_grade_ovk as pub  # noqa: E402


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2000-01-01", periods=n, freq="MS"))


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _simulated_scores(n: int = 48, seed: int = 44, nonzero_mean: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2), dtype=float)
    for t in range(n):
        phase = t / max(n - 1, 1)
        R = _rot(0.15 + 1.05 * phase)
        vals = np.asarray([0.25 + 1.25 * phase, 1.35 - 0.75 * phase])
        cov = R @ np.diag(vals) @ R.T
        mean = np.asarray([1.5 * np.sin(2 * np.pi * phase), 0.9 * np.cos(2 * np.pi * phase)]) if nonzero_mean else 0.0
        out[t] = mean + rng.multivariate_normal(np.zeros(2), cov)
    return out


def test_scalar_arithmetic_differs_from_geometric_mean() -> None:
    scalar_second_moments = np.asarray([1.0, 4.0])
    arithmetic = float(np.mean(scalar_second_moments))
    geometric = float(np.exp(np.mean(np.log(scalar_second_moments))))
    assert arithmetic == 2.5
    assert geometric == 2.0
    assert arithmetic > geometric


def test_noncommuting_state_varying_matrices_are_fit_without_log_geometry() -> None:
    A0 = _rot(0.2) @ np.diag([1.7, 0.35]) @ _rot(0.2).T
    A1 = _rot(1.0) @ np.diag([0.45, 1.55]) @ _rot(1.0).T
    assert np.linalg.norm(A0 @ A1 - A1 @ A0, ord="fro") > 0.1

    psi = _simulated_scores(nonzero_mean=False)
    res = pub.estimate_rank_model(psi, _dates(len(psi)), "sim", "simulation", 2, em_iters=2, estimator_mode="arithmetic_outer_product")
    assert res.estimator_mode == "arithmetic_outer_product"
    assert np.linalg.norm(res.A[:8].mean(axis=0) - res.A[-8:].mean(axis=0), ord="fro") > 1e-3


def test_nonzero_conditional_influence_means_return_mean_and_within_components() -> None:
    psi = _simulated_scores(nonzero_mean=True)
    res = pub.estimate_rank_model(psi, _dates(len(psi)), "mean", "mean simulation", 2, em_iters=2, estimator_mode="arithmetic_outer_product")
    total_tau = np.trace(res.total_second_moment_whitened, axis1=1, axis2=2) / 2
    mean_tau = np.trace(res.mean_component_whitened, axis1=1, axis2=2) / 2
    within_tau = np.trace(res.within_covariance_whitened, axis1=1, axis2=2) / 2
    assert np.nanmax(mean_tau) > 0.05
    assert np.all(total_tau >= -1e-10)
    assert np.all(within_tau >= -1e-10)


def test_nonuniform_reference_weights_define_reference_second_moment() -> None:
    psi = np.asarray([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [-1.0, 1.0]])
    weights = np.asarray([0.05, 0.15, 0.70, 0.10])
    basis = pub.covariance_basis(psi, 2, reference_weights=weights, estimator_mode="arithmetic_outer_product")
    expected = (psi.T * (weights / weights.sum())) @ psi
    assert np.allclose(basis["reference_weights"], weights / weights.sum())
    assert np.allclose(basis["reference_covariance"], expected)


def test_arithmetic_outputs_are_psd_in_both_coordinate_systems() -> None:
    psi = _simulated_scores(nonzero_mean=True)
    res = pub.estimate_rank_model(psi, _dates(len(psi)), "psd", "psd simulation", 2, em_iters=2, estimator_mode="arithmetic_outer_product")
    stacks = [
        res.total_second_moment_whitened,
        res.mean_component_whitened,
        res.within_covariance_whitened,
        res.total_second_moment_original,
        res.mean_component_original,
        res.within_covariance_original,
    ]
    for stack in stacks:
        eigs = np.linalg.eigvalsh(stack)
        assert np.min(eigs) >= -1e-8


def test_arithmetic_mode_does_not_call_matrix_log_or_exp() -> None:
    psi = _simulated_scores(n=32, nonzero_mean=True)
    old_log = pub.batched_spd_log
    old_exp = pub.batched_spd_exp

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("arithmetic_outer_product must not call matrix log/exp helpers")

    try:
        pub.batched_spd_log = forbidden  # type: ignore[assignment]
        pub.batched_spd_exp = forbidden  # type: ignore[assignment]
        res = pub.estimate_rank_model(
            psi,
            _dates(len(psi)),
            "nolog",
            "no log simulation",
            2,
            em_iters=1,
            estimator_mode="arithmetic_outer_product",
        )
        assert np.isfinite(res.tau).all()
    finally:
        pub.batched_spd_log = old_log  # type: ignore[assignment]
        pub.batched_spd_exp = old_exp  # type: ignore[assignment]
