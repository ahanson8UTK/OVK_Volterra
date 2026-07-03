from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

os.environ.setdefault("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "4")

import run_publication_grade_ovk as pub  # noqa: E402
import iv_ovk  # noqa: E402


def _synthetic_chi(seed: int = 20260623, n: int = 36, p: int = 10) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    load = rng.normal(size=(3, p))
    factors = rng.normal(size=(n, 3))
    chi = factors @ load + 0.15 * rng.normal(size=(n, p))
    dates = pd.Series(pd.date_range("2000-01-01", periods=n, freq="MS"))
    return chi, dates


def test_full_coordinate_covariances_are_symmetric_psd_and_centered() -> None:
    chi, dates = _synthetic_chi()
    result = pub.estimate_full_coordinate_kernel_model(
        chi,
        dates,
        variant="synthetic",
        label="Synthetic",
        outcome_labels=["A", "B"],
        kernel_eta=0.10,
    )
    assert np.max(np.abs(result.C_hat - result.C_hat.T)) < 1.0e-10
    assert np.max(np.abs(result.K_hat - np.swapaxes(result.K_hat, 1, 2))) < 1.0e-10
    assert float(np.linalg.eigvalsh(result.C_hat).min()) > -1.0e-8
    assert float(np.linalg.eigvalsh(result.K_hat).min()) > -1.0e-8
    assert float(np.linalg.eigvalsh(result.A_hat).min()) > -1.0e-8
    assert result.d_rho > 0.0
    assert abs(float(result.tau_soft.mean()) - 1.0) < 1.0e-8
    assert np.linalg.norm(result.K_hat.mean(axis=0) - result.C_hat, ord="fro") / np.linalg.norm(result.C_hat, ord="fro") < 1.0e-8


def test_full_coordinate_block_denominators_and_deterministic_bootstrap() -> None:
    chi, dates = _synthetic_chi(p=10)
    labels = ["Macro", "Financial"]
    result = pub.estimate_full_coordinate_kernel_model(
        chi,
        dates,
        variant="synthetic",
        label="Synthetic",
        outcome_labels=labels,
        kernel_eta=0.05,
    )
    block_df = pub.full_coordinate_block_shape_paths(result, H=4, pvars=2, labels=labels)
    assert (block_df["denominator"] > 0.0).all()
    draws1 = pub.full_coordinate_block_bootstrap_tau_draws(chi, dates, 5, seed=123, block_len=6, kernel_eta=0.05)
    draws2 = pub.full_coordinate_block_bootstrap_tau_draws(chi, dates, 5, seed=123, block_len=6, kernel_eta=0.05)
    assert np.allclose(draws1, draws2)


def test_section31_source_paths_do_not_use_rank_projection_tokens() -> None:
    source = "\n".join(
        [
            inspect.getsource(pub.build_full_coordinate_section31_outputs),
            inspect.getsource(pub.estimate_full_coordinate_kernel_model),
            inspect.getsource(pub.full_coordinate_block_shape_paths),
            inspect.getsource(iv_ovk.run_iv_tau_multiplicative_driver_diagnostic),
        ]
    )
    banned = ["V_R", "Lambda_R", "projected scores", "PCA cutoff", "headline.Z", "headline.V[:, :R]"]
    for token in banned:
        assert token not in source
    assert "OVK_WRITE_LEGACY_TOP5_COMPAT" in inspect.getsource(pub)


def test_proxy_iv_full_coordinate_energy_identity() -> None:
    rng = np.random.default_rng(44)
    n, p = 40, 8
    M = rng.normal(size=n)
    u = rng.normal(size=(n, p))
    pi_hat = 0.7
    chi_proxy = (M / pi_hat)[:, None] * u
    dates = pd.Series(pd.date_range("2001-01-01", periods=n, freq="MS"))
    result = pub.estimate_full_coordinate_kernel_model(
        chi_proxy,
        dates,
        variant="synthetic_iv",
        label="Synthetic IV",
        outcome_labels=["A", "B"],
        kernel_eta=0.07,
    )
    cD, lowerD = pub.cho_factor(result.D_rho, lower=True, check_finite=False)
    residual_solved = pub.cho_solve((cD, lowerD), u.T, check_finite=False).T
    score_solved = pub.cho_solve((cD, lowerD), chi_proxy.T, check_finite=False).T
    residual_energy = np.einsum("ti,ti->t", u, residual_solved) / result.d_rho
    score_energy = np.einsum("ti,ti->t", chi_proxy, score_solved) / result.d_rho
    exposure = (M / pi_hat) ** 2
    assert np.allclose(score_energy, exposure * residual_energy, atol=1.0e-10, rtol=1.0e-10)
