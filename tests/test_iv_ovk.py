"""Smoke tests for the proxy-IV LP/OVK appendix."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ["OVK_IV_ROOT"] = str(ROOT / "tmp" / "test_iv_ovk")
os.environ["OVK_PUBLICATION_ROOT"] = str(ROOT / "tmp" / "test_iv_publication")
os.environ["OVK_NESTED_OUTDIR"] = str(ROOT / "tmp" / "test_iv_nested")
os.environ["OVK_PUBLICATION_WORKERS"] = "1"
os.environ["OVK_PUBLICATION_BOOTSTRAP_DRAWS"] = "2"
os.environ["OVK_PUBLICATION_STATE_DRAWS"] = "2"
os.environ["OVK_IV_BOOTSTRAP_DRAWS"] = "2"
os.environ["OVK_IV_NESTED_BOOT_DRAWS"] = "5"
sys.path.insert(0, str(ROOT / "code"))

import download_iv_data  # noqa: E402
import iv_ovk  # noqa: E402


def synthetic_panel(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(20260604)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    z = rng.normal(0, 1, n)
    cbi = rng.normal(0, 0.5, n)
    x = np.zeros(n)
    x[0] = 4.0 + 0.5 * z[0]
    for t in range(1, n):
        x[t] = 0.86 * x[t - 1] + 0.35 * z[t] + 0.08 * cbi[t] + rng.normal(0, 0.15)
    growth = 0.1 + 0.02 * rng.normal(size=n) - 0.015 * x
    prices = 0.08 + 0.02 * rng.normal(size=n) + 0.004 * x
    return pd.DataFrame(
        {
            "date": dates,
            "ip": 100 + np.cumsum(growth),
            "cpi": 100 + np.cumsum(prices),
            "unrate": 5 + 0.08 * rng.normal(size=n) + 0.03 * x,
            "gs2": 2 + 0.25 * x + 0.05 * rng.normal(size=n),
            "baa10y": 1 + 0.04 * rng.normal(size=n) - 0.02 * x,
            "CBI_median_fallback": cbi,
            "dgs1_eom": x,
            "dgs1_eom_diff": np.r_[np.nan, np.diff(x)],
            "iv_z_raw": z,
            "iv_z_orth": z - 0.1 * cbi,
            "iv_z_preferred": z - 0.1 * cbi,
            "iv_z_source_column": "mps_orth",
            "iv_z_source_file": "synthetic",
        }
    )


def write_local_source_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    sf = tmp_path / "sf.csv"
    pd.DataFrame(
        {
            "meeting_date": pd.date_range("2000-01-15", periods=24, freq="MS"),
            "mps": np.linspace(-1.0, 1.0, 24),
            "mps_orth": np.linspace(-0.8, 0.8, 24),
            "ff4_path_surprise": np.linspace(-0.4, 0.4, 24),
        }
    ).to_csv(sf, index=False)
    dgs1 = tmp_path / "DGS1_source.csv"
    daily_dates = pd.date_range("1999-12-15", periods=750, freq="D")
    pd.DataFrame({"observation_date": daily_dates, "DGS1": 3.0 + np.sin(np.arange(len(daily_dates)) / 30.0)}).to_csv(dgs1, index=False)
    gs1 = tmp_path / "GS1_source.csv"
    pd.DataFrame({"observation_date": pd.date_range("1999-12-01", periods=30, freq="MS"), "GS1": 3.0}).to_csv(gs1, index=False)
    return sf, dgs1, gs1


def test_download_and_panel_processing_from_local_urls(tmp_path: Path) -> None:
    sf, dgs1, gs1 = write_local_source_files(tmp_path)
    raw_dir = tmp_path / "raw_iv"
    meta = download_iv_data.download_iv_data(
        raw_dir,
        sf_direct_url=sf.as_uri(),
        sf_landing_url=tmp_path.as_uri(),
        dgs1_url=dgs1.as_uri(),
        gs1_url=gs1.as_uri(),
    )
    assert Path(meta["sf_fed_path"]).exists()
    assert (raw_dir / "DGS1.csv").exists()
    assert (raw_dir / "dgs1_monthly_eom.csv").exists()
    assert (raw_dir / "iv_data_sources.json").exists()

    panel = synthetic_panel(30).drop(columns=["dgs1_eom", "dgs1_eom_diff", "iv_z_raw", "iv_z_orth", "iv_z_preferred", "iv_z_source_column", "iv_z_source_file"])
    panel_path = tmp_path / "baseline_panel.csv"
    out_path = tmp_path / "iv_proxy_policy_panel.csv"
    panel.to_csv(panel_path, index=False)
    processed, merge_meta = iv_ovk.build_iv_policy_panel(panel_path, raw_dir=raw_dir, out_path=out_path, allow_download=False)
    assert out_path.exists()
    assert processed["dgs1_eom"].notna().sum() >= 20
    assert np.isfinite(processed["iv_z_preferred"]).sum() >= 12
    assert merge_meta["instrument_preferred"] == "iv_z_orth"


def test_iv_scores_basis_and_rank5_smoke() -> None:
    panel = synthetic_panel(220)
    scores = iv_ovk.build_iv_lp_scores(panel, H=24, L=12, outcome_columns=tuple(iv_ovk.BASE_OUTCOME_COLUMNS))
    assert scores["Q_scores"].shape[1] == 25 * 5
    assert scores["Q_scores"].shape[0] == len(scores["dates"])
    assert np.isfinite(scores["first_stage"]["first_stage_f_stat"])
    sample_summary = iv_ovk.write_sample_summary(
        panel,
        scores,
        {"instrument_preferred": "iv_z_orth", "instrument_source_column": "mps_orth"},
    )
    sample_items = dict(zip(sample_summary["item"], sample_summary["value"].astype(str)))
    assert sample_items["State index attached to"] == "base month t (the plotted date)"
    assert "complete-coordinate sample" in sample_items["First-stage denominator sample"]
    assert sample_items["Common complete-coordinate score coverage"] == sample_items["Usable IV sample range"]
    basis = iv_ovk.covariance_basis(scores["Q_scores"], 5)
    assert basis["V"].shape == (25 * 5, 5)
    rank = iv_ovk.estimate_rank_model(
        scores["Q_scores"],
        scores["dates"],
        "test_iv",
        "Synthetic IV",
        5,
        em_iters=2,
        outcome_labels=scores["outcome_labels"],
    )
    assert np.isfinite(rank.tau).all()
    assert rank.A.shape[1:] == (5, 5)
    decomp = iv_ovk.write_score_energy_decomposition(scores, rank)
    summary = decomp["summary"].iloc[0]
    assert np.isfinite(summary["max_abs_diff_between_z_score_and_fitted_score_forms"])
    assert summary["max_abs_diff_between_z_score_and_fitted_score_forms"] < 1e-8
    assert set(decomp["top_tau"]["driver_label"]).issubset({"first-stage-driven", "residual-driven", "mixed"})
    assert (iv_ovk.TABLES / "iv_score_energy_decomposition_path.csv").exists()
    assert (iv_ovk.TABLES / "iv_top_tau_decomposition.csv").exists()
    assert (iv_ovk.TABLES / "iv_score_decomposition_summary.csv").exists()
    assert (iv_ovk.CHARTS / "iv_score_decomposition_top_tau.png").exists()
    exact = iv_ovk.run_iv_tau_exact_decomposition(scores, rank, em_iters=2)
    exact_df = exact["path"]
    assert {
        "tau_fs",
        "tau_rf",
        "tau_cross",
        "tau_total_dec",
        "tau_total_decomp",
        "tau_aug_implied",
        "raw_tau_existing_from_scores",
        "raw_tau_sum_from_components",
    }.issubset(exact_df.columns)
    assert np.max(np.abs(exact_df["tau_total_decomp"] - exact_df["tau_aug_implied"])) < 1e-7
    assert exact["diagnostics"]["max_abs_score_error"] < 1e-8
    assert exact["diagnostics"]["max_abs_if_error"] < 1e-8
    assert exact["diagnostics"]["max_abs_stored_if_error"] < 1e-8
    assert exact["diagnostics"]["max_abs_retained_direct_error"] < 1e-8
    assert abs(exact["diagnostics"]["raw_tau_existing_mean"] - 1.0) < 1e-8
    assert abs(exact["diagnostics"]["raw_aug_tau_bar"] - 1.0) < 1e-8
    assert abs(exact["diagnostics"]["mean_augmented_total_tau"] - 1.0) < 1e-8
    assert exact["diagnostics"]["used_correct_unnormalization"] is True
    assert (iv_ovk.TABLES / "iv_tau_exact_decomposition_timeseries.csv").exists()
    assert (iv_ovk.TABLES / "iv_tau_exact_decomposition_diagnostics.json").exists()
    assert (iv_ovk.FIGURES / "iv_tau_exact_decomposition_area.png").exists()
    assert (iv_ovk.FIGURES / "iv_tau_exact_decomposition_area.pdf").exists()
    assert (iv_ovk.FIGURES / "iv_tau_exact_decomposition_excess_area.png").exists()
    assert (iv_ovk.FIGURES / "iv_tau_exact_decomposition_excess_area.pdf").exists()
    assert (iv_ovk.FIGURES / "iv_tau_exact_decomposition_area_caption.tex").exists()
    drivers = iv_ovk.run_iv_tau_multiplicative_driver_diagnostic(scores, rank)
    driver_df = drivers["path"]
    assert {
        "tau_iv",
        "p_tau",
        "proxy_exposure",
        "p_proxy",
        "resid_energy",
        "p_resid",
        "driver_label",
        "tau_soft_existing",
        "tau_soft_reconstructed",
        "E_raw_smoothed",
        "R_exposure_weighted_raw",
        "exposure_factor",
        "residual_factor",
        "factor_product",
        "weight_row_sum",
        "effective_weight_count",
        "maximum_weight",
    }.issubset(driver_df.columns)
    assert drivers["diagnostics"]["coordinate_dimension"] == 25 * 5
    assert drivers["diagnostics"]["max_abs_multiplicative_energy_product_error"] < 1e-8
    assert drivers["diagnostics"]["rel_fro_multiplicative_energy_product_error"] < 1e-8
    assert drivers["diagnostics"]["covariance_estimator_linear"] is True
    assert drivers["diagnostics"]["max_rel_reconstruction_error"] < 1e-8
    assert drivers["diagnostics"]["max_rel_factor_product_error"] < 1e-8
    assert np.allclose(driver_df["tau_soft_existing"], driver_df["tau_soft_reconstructed"], atol=1e-8, rtol=1e-8)
    assert np.allclose(driver_df["tau_soft_existing"], driver_df["factor_product"], atol=1e-8, rtol=1e-8)
    assert (iv_ovk.TABLES / "iv_tau_driver_diagnostic_monthly.csv").exists()
    assert (iv_ovk.TABLES / "iv_tau_factor_decomposition_audit.csv").exists()
    assert (iv_ovk.TABLES / "iv_tau_driver_top_months.csv").exists()
    assert (iv_ovk.TABLES / "iv_tau_driver_high_tau_episodes.csv").exists()
    assert (iv_ovk.TABLES / "iv_tau_driver_diagnostic_summary.json").exists()
    assert (iv_ovk.FIGURES / "iv_tau_driver_diagnostic.png").exists()
    assert (iv_ovk.FIGURES / "iv_tau_driver_diagnostic.pdf").exists()
    assert (iv_ovk.FIGURES / "iv_tau_driver_heatmap.png").exists()
    assert (iv_ovk.FIGURES / "iv_tau_driver_heatmap.pdf").exists()
    assert (iv_ovk.FIGURES / "iv_tau_driver_diagnostic_caption.tex").exists()


def test_augmented_trace_decomposition_identity() -> None:
    rng = np.random.default_rng(123)
    R = 3
    raw = rng.normal(size=(7, 2 * R, 2 * R))
    Sigma = 0.5 * (raw + np.swapaxes(raw, 1, 2))
    out = iv_ovk.trace_decomposition_from_augmented_covariance(Sigma, R)
    assert np.allclose(out["tau_fs"] + out["tau_res"] + out["tau_cross"], out["tau_aug_implied"])
