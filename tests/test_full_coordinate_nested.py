from __future__ import annotations

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
os.environ.setdefault("OVK_NESTED_SCORE_BOOTSTRAP_DRAWS", "4")

import ovk_full_coordinate_nested as fcn  # noqa: E402
import ovk_nested_workflow as nested  # noqa: E402
from run_publication_grade_ovk import full_coordinate_K_from_weights, full_coordinate_temporal_weights, sym  # noqa: E402


def _scores(seed: int = 123, n: int = 80, p: int = 7) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    load = rng.normal(size=(3, p))
    factors = rng.normal(size=(n, 3))
    q = 0.4 + factors @ load + 0.2 * rng.normal(size=(n, p))
    dates = pd.Series(pd.date_range("2000-01-01", periods=n, freq="MS"))
    return q, dates


def _config(**kwargs) -> fcn.FullCoordinateNestedConfig:
    params = dict(
        eval_start_index=50,
        validation_months=15,
        min_initial_observations=20,
        mean_half_lives=(6.0, 12.0, np.inf),
        cov_half_lives=(6.0, 12.0, np.inf),
        bootstrap_draws=5,
        bootstrap_block_len=4,
        bootstrap_seed=99,
        expected_production_p=None,
    )
    params.update(kwargs)
    return fcn.FullCoordinateNestedConfig(**params)


def _synthetic_panel(n: int = 86) -> pd.DataFrame:
    rng = np.random.default_rng(20260623)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "ip": 100 + np.cumsum(0.2 + rng.normal(0, 0.01, n)),
            "cpi": 100 + np.cumsum(0.1 + rng.normal(0, 0.01, n)),
            "cpi10": 2 + rng.normal(0, 0.02, n),
            "unrate": 5 + rng.normal(0, 0.05, n),
            "gs2": 3 + rng.normal(0, 0.05, n),
            "baa10y": 1 + rng.normal(0, 0.03, n),
            "expinf5yr": 2 + rng.normal(0, 0.02, n),
            "mich": 3 + rng.normal(0, 0.03, n),
            "MP_used": rng.normal(0, 1, n),
            "CBI_used": rng.normal(0, 1, n),
        }
    )


def _dummy_geometry(p: int) -> fcn.FullCoordinateGeometry:
    eye = np.eye(p)
    return fcn.FullCoordinateGeometry(
        p=p,
        theta_hat=np.zeros(p),
        x=np.empty((0, p)),
        y=np.empty((0, p)),
        C_ref=eye,
        D_ref=eye,
        L_ref=eye,
        rho_rel=0.0,
        rho=0.0,
        trace_C_ref=float(p),
        d_rho=float(p),
        logdet_L_ref=0.0,
        ref_end=0,
        centering_convention="test_identity_geometry",
        cholesky_min_pivot=1.0,
        cholesky_max_pivot=1.0,
        cholesky_diag_condition=1.0,
    )


def test_complete_coordinate_operation_uses_input_dimension() -> None:
    q, dates = _scores(p=9)
    result = fcn.run_full_coordinate_nested(q, dates, _config())
    assert result.geometry.p == 9
    assert result.model_results["M3"].predicted_means.shape[1] == 9
    assert result.model_results["M3"].predicted_covariances.shape[1:] == (9, 9)
    assert result.metadata["centering_convention"] == "raw_lp_score_contributions_centered_by_pre_evaluation_theta_hat"


def test_boundary_nesting_equalities() -> None:
    q, dates = _scores(p=6)
    config = _config()
    geometry = fcn.build_reference_geometry(q, 50, config)
    y = geometry.y
    common = dict(k_const_fit_end=35, score_start=50, score_end=65)

    m0 = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M0", np.inf, np.inf), **common)
    m1_boundary = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M1", np.inf, np.inf), **common)
    m2_boundary = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M2", np.inf, np.inf), **common)
    assert np.allclose(m1_boundary.log_scores_x, m0.log_scores_x)
    assert np.allclose(m2_boundary.log_scores_x, m0.log_scores_x)
    assert np.allclose(m1_boundary.predicted_means, m0.predicted_means)
    assert np.allclose(m2_boundary.predicted_covariances, m0.predicted_covariances)

    m1 = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M1", np.inf, 6.0), **common)
    m3_mean_boundary = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M3", np.inf, 6.0), **common)
    assert np.allclose(m3_mean_boundary.log_scores_x, m1.log_scores_x)
    assert np.allclose(m3_mean_boundary.predicted_covariances, m1.predicted_covariances)

    m2 = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M2", 12.0, np.inf), **common)
    m3_cov_boundary = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M3", 12.0, np.inf), **common)
    assert np.allclose(m3_cov_boundary.log_scores_x, m2.log_scores_x)
    assert np.allclose(m3_cov_boundary.predicted_means, m2.predicted_means)

    m3_both_boundary = fcn.score_causal_model(y, dates, geometry, config, fcn._spec("M3", np.inf, np.inf), **common)
    assert np.allclose(m3_both_boundary.log_scores_x, m0.log_scores_x)


def test_no_lookahead_from_future_observations() -> None:
    q, dates = _scores(p=5)
    config = _config()
    q2 = q.copy()
    q2[56:] += 100.0
    g1 = fcn.build_reference_geometry(q, 50, config)
    g2 = fcn.build_reference_geometry(q2, 50, config)
    spec = fcn._spec("M3", 6.0, 6.0)
    common = dict(k_const_fit_end=35, score_start=50, score_end=56)
    r1 = fcn.score_causal_model(g1.y, dates, g1, config, spec, **common)
    r2 = fcn.score_causal_model(g2.y, dates, g2, config, spec, **common)
    assert np.allclose(r1.predicted_means, r2.predicted_means)
    assert np.allclose(r1.predicted_covariances, r2.predicted_covariances)
    assert np.allclose(r1.log_scores_x, r2.log_scores_x)


def test_no_lookahead_same_date_forecast_distribution_is_fixed() -> None:
    q, dates = _scores(p=5)
    q2 = q.copy()
    q2[50] += 100.0
    config = _config()
    g1 = fcn.build_reference_geometry(q, 50, config)
    g2 = fcn.build_reference_geometry(q2, 50, config)
    spec = fcn._spec("M3", 6.0, 6.0)
    common = dict(k_const_fit_end=35, score_start=50, score_end=52)
    r1 = fcn.score_causal_model(g1.y, dates, g1, config, spec, **common)
    r2 = fcn.score_causal_model(g2.y, dates, g2, config, spec, **common)
    assert np.allclose(r1.predicted_means[0], r2.predicted_means[0])
    assert np.allclose(r1.predicted_covariances[0], r2.predicted_covariances[0])
    assert not np.isclose(r1.log_scores_x[0], r2.log_scores_x[0])


def test_gaussian_score_matches_reference_and_decomposition() -> None:
    e = np.asarray([0.4, -1.2])
    V = np.asarray([[2.0, 0.3], [0.3, 1.5]])
    comp = fcn.cholesky_logpdf(e, V, max_jitter=1.0e-12)
    sign, logdet = np.linalg.slogdet(V)
    assert sign > 0
    quad = float(e @ np.linalg.solve(V, e))
    ref = -0.5 * (2 * np.log(2.0 * np.pi) + logdet + quad)
    assert np.isclose(comp.logpdf, ref)
    assert np.isclose(comp.constant_term + comp.logdet_term + comp.mahalanobis_term, comp.logpdf)
    assert np.allclose(comp.covariance_used, V)


def test_score_decomposition_reconstructs_total_and_covariance_is_consistent() -> None:
    q, dates = _scores(p=6)
    result = fcn.run_full_coordinate_nested(q, dates, _config())
    decomp = result.score_decomposition
    assert np.max(np.abs(decomp["constant_term"] + decomp["logdet_term"] + decomp["mahalanobis_term"] - decomp["log_score_y"])) < 1e-9
    pairs = result.pair_decomposition
    assert np.max(np.abs(pairs["reconstruction_error"])) < 1e-8
    m0 = result.model_results["M0"]
    e = result.geometry.y[result.eval_start] - m0.predicted_means[0]
    comp = fcn.cholesky_logpdf(e, m0.predicted_covariances[0], max_jitter=1.0e-12)
    assert np.isclose(comp.logdet, m0.covariance_logdets[0])
    assert np.isclose(comp.mahalanobis_squared, m0.mahalanobis_squared[0])


def test_spd_high_dimensional_stability_without_pseudodeterminants() -> None:
    q, dates = _scores(n=58, p=30)
    config = _config(eval_start_index=38, validation_months=12, min_initial_observations=20)
    result = fcn.run_full_coordinate_nested(q, dates, config)
    for model in fcn.MODEL_ORDER:
        scores = result.model_results[model]
        assert np.isfinite(scores.log_scores_x).all()
        assert np.isfinite(scores.covariance_logdets).all()
        assert (scores.min_cholesky_pivots > 0.0).all()
        assert np.max(np.abs(scores.predicted_covariances - np.swapaxes(scores.predicted_covariances, 1, 2))) < 1e-10
    diagnostics = result.covariance_diagnostics
    assert set(diagnostics["model"]) == set(fcn.MODEL_ORDER)
    assert (diagnostics["covariance_floor"] > 0.0).all()
    assert (diagnostics["min_predicted_cov_eigenvalue"] > 0.0).all()
    assert (diagnostics["jittered_periods"] >= 0).all()
    assert np.isfinite(diagnostics["max_cholesky_jitter"]).all()


def test_rank_deficient_p_greater_than_n_uses_shrinkage_not_floor_as_estimator() -> None:
    q, dates = _scores(n=64, p=60)
    config = _config(eval_start_index=44, validation_months=18, min_initial_observations=15, covariance_floor_rel=1.0e-10)
    result = fcn.run_full_coordinate_nested(q, dates, config, include_sensitivity=False)
    diag = result.covariance_diagnostics
    assert (diag["training_residual_count"] < diag["score_dimension"]).all()
    assert (diag["training_residual_rank"] < diag["score_dimension"]).all()
    assert (diag["min_eigenvalue_to_floor_ratio"] > 1000.0).all()


def test_floor_accounting_counts_below_floor_and_clipped_eigenvalues() -> None:
    raw = np.diag([1.0e-12, 1.0, 2.0])
    reg = fcn.CovarianceRegularizationState(
        estimator="ledoit_wolf",
        shrinkage=0.0,
        target_scale=1.0,
        covariance_floor_rel=1.0e-6,
        max_cholesky_jitter_rel=1.0e-8,
        training_residual_count=10,
        training_residual_rank=3,
    )
    out = fcn._regularize_forecast_covariance(raw, raw, reg)
    assert out["number_eigenvalues_below_floor"] == 1
    assert out["number_eigenvalues_clipped"] == 1
    assert out["regularized_min_eigenvalue"] >= out["floor_value"]


def test_structural_subspace_constraint_is_detected() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(60, 2))
    q = np.column_stack([x, x[:, 0] + x[:, 1]])
    diag = fcn.structural_subspace_diagnostics(q)
    row = diag.loc[diag["matrix"].eq("score_matrix")].iloc[0]
    assert row["numerical_rank"] == 2
    assert row["rank_deficiency"] == 1
    assert bool(row["structural_constraint_detected"])


def test_whitening_jacobian_cancels_in_model_differences() -> None:
    q, dates = _scores(p=6)
    result = fcn.run_full_coordinate_nested(q, dates, _config())
    m1 = result.model_results["M1"]
    m0 = result.model_results["M0"]
    assert np.allclose(m1.log_scores_x - m0.log_scores_x, m1.log_scores_y - m0.log_scores_y)


def test_mean_within_matrix_and_tau_identity() -> None:
    q, dates = _scores(p=5)
    config = _config()
    geometry = fcn.build_reference_geometry(q, 50, config)
    weights = full_coordinate_temporal_weights(dates, eta=config.kernel_eta)
    K_total = full_coordinate_K_from_weights(geometry.x, weights)
    local = weights @ geometry.x
    K_mean = np.einsum("ti,tj->tij", local, local, optimize=True)
    K_within = sym(K_total[10] - K_mean[10])
    assert np.allclose(K_total[10], K_mean[10] + K_within)

    diagnostic = fcn.compute_mean_within_diagnostic(dates, geometry, config)
    assert np.max(np.abs(diagnostic["tau_total"] - diagnostic["tau_mean"] - diagnostic["tau_within"])) < 1e-8


def test_paired_bootstrap_uses_joint_indices_and_zero_difference_interval() -> None:
    base = np.arange(20, dtype=float)
    scores = np.column_stack([base, base, base + 1.0, base + 1.0])
    draws, indices = fcn.paired_moving_block_bootstrap(scores, block_len=5, draws=6, seed=10)
    assert indices.shape == (6, 20)
    assert np.allclose(draws["M1_minus_M0"], 0.0)
    model_scores = pd.DataFrame({"log_score_M0": base, "log_score_M1": base, "log_score_M2": base, "log_score_M3": base})
    zero_draws, _ = fcn.paired_moving_block_bootstrap(np.column_stack([base, base, base, base]), block_len=5, draws=6, seed=10)
    summary = fcn.comparison_summary(model_scores, zero_draws, d_rho=4.0)
    row = summary.loc[summary["comparison"].eq("M1 - M0")].iloc[0]
    assert abs(row["p05"]) < 1e-12
    assert abs(row["p95"]) < 1e-12


def test_influence_diagnostics_reconcile_with_total_difference() -> None:
    pair = pd.DataFrame(
        {
            "evaluation_date": pd.date_range("2000-01-01", periods=4, freq="MS").strftime("%Y-%m-%d"),
            "comparison": ["M1 - M0"] * 4,
            "score_diff": [1.0, 2.0, 3.0, 4.0],
            "left_mahalanobis_per_dimension": [1.0] * 4,
            "right_mahalanobis_per_dimension": [1.0] * 4,
            "either_model_clipped": [False] * 4,
            "left_clipped_eigenvalues": [0] * 4,
            "right_clipped_eigenvalues": [0] * 4,
        }
    )
    summary = fcn.pair_score_summary_frame(pair).iloc[0]
    assert np.isclose(summary["top1_abs_date_fraction_of_total"], 0.4)
    top = fcn.influential_dates_frame(pair, top_n=2)
    assert np.isclose(top["fraction_of_total_difference"].sum(), 0.7)


def test_comparison_summary_reports_average_sum_and_per_coordinate_scale() -> None:
    base = np.arange(10, dtype=float)
    model_scores = pd.DataFrame(
        {
            "log_score_M0": base,
            "log_score_M1": base + 4.0,
            "log_score_M2": base,
            "log_score_M3": base + 4.0,
        }
    )
    draws, _ = fcn.paired_moving_block_bootstrap(
        model_scores[[f"log_score_{name}" for name in fcn.MODEL_ORDER]].to_numpy(float),
        block_len=4,
        draws=5,
        seed=10,
    )
    summary = fcn.comparison_summary(model_scores, draws, d_rho=5.0, score_dimension=20)
    row = summary.loc[summary["comparison"].eq("M1 - M0")].iloc[0]
    assert row["score_difference_unit"] == "average per evaluation date"
    assert row["n_evaluation_dates"] == 10
    assert row["score_dimension"] == 20
    assert row["avg_log_score_diff"] == 4.0
    assert row["sum_log_score_diff"] == 40.0
    assert np.isclose(row["avg_joint_log_score_diff_per_dimension"], 0.2)


def test_synthetic_gaussian_mahalanobis_per_dimension_calibration() -> None:
    rng = np.random.default_rng(20260625)
    n, p = 520, 6
    y = rng.normal(size=(n, p))
    dates = pd.Series(pd.date_range("1980-01-01", periods=n, freq="MS"))
    config = fcn.FullCoordinateNestedConfig(
        eval_start_index=360,
        validation_months=120,
        min_initial_observations=200,
        mean_half_lives=(np.inf,),
        cov_half_lives=(np.inf,),
        bootstrap_draws=3,
        bootstrap_block_len=12,
        expected_production_p=None,
    )
    result = fcn.score_causal_model(
        y,
        dates,
        _dummy_geometry(p),
        config,
        fcn._spec("M0", np.inf, np.inf),
        k_const_fit_end=240,
        score_start=360,
        score_end=n,
    )
    avg = float(np.mean(result.mahalanobis_per_dimension))
    assert 0.7 < avg < 1.3


def test_determinism_with_fixed_seed() -> None:
    q, dates = _scores(p=6)
    config = _config(bootstrap_seed=777)
    r1 = fcn.run_full_coordinate_nested(q, dates, config)
    r2 = fcn.run_full_coordinate_nested(q, dates, config)
    assert r1.tuning.selected_mean_half_life == r2.tuning.selected_mean_half_life
    assert r1.tuning.selected_cov_half_life == r2.tuning.selected_cov_half_life
    assert np.allclose(r1.model_scores.filter(like="log_score_M").to_numpy(float), r2.model_scores.filter(like="log_score_M").to_numpy(float))
    assert np.allclose(r1.comparisons["p05"].to_numpy(float), r2.comparisons["p05"].to_numpy(float))


def test_production_path_does_not_call_legacy_reduced_functions(monkeypatch, tmp_path) -> None:
    def _boom(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("legacy reduced-rank function was called")

    monkeypatch.setattr(nested, "covariance_eigenbasis", _boom)
    monkeypatch.setattr(nested, "build_mean_and_eval_basis", _boom)
    monkeypatch.setattr(nested, "upgraded_state_space_A_from_z", _boom)
    monkeypatch.setattr(nested, "estimate_alpha_and_state", _boom)
    monkeypatch.setattr(nested, "NESTED_MODE", "full-coordinate-arithmetic")
    monkeypatch.setattr(nested, "OUTDIR", tmp_path)
    monkeypatch.setattr(nested, "TABLE_DIR", tmp_path / "tables")
    monkeypatch.setattr(nested, "CHART_DIR", tmp_path / "charts")
    monkeypatch.setattr(nested, "CODE_DIR", tmp_path / "code")
    monkeypatch.setattr(nested, "NESTED_BOOTSTRAP_DRAWS", 2)
    monkeypatch.setenv("OVK_NESTED_EVAL_START_INDEX", "24")
    monkeypatch.setenv("OVK_NESTED_VALIDATION_MONTHS", "8")
    monkeypatch.setenv("OVK_NESTED_MIN_INITIAL_OBSERVATIONS", "8")
    monkeypatch.setenv("OVK_NESTED_MEAN_HALF_LIVES", "6,inf")
    monkeypatch.setenv("OVK_NESTED_COV_HALF_LIVES", "6,inf")
    monkeypatch.setenv("OVK_NESTED_EXPECTED_P", "none")
    spec = nested.NestedVariant("base5_headline", "Headline original five outcomes", tuple(nested.BASE_OUTCOME_COLUMNS))
    result = nested.run_nested_variant(_synthetic_panel(), spec)
    assert result["metadata"]["estimator_mode"] == "full_coordinate_arithmetic_causal_predictive"
    assert result["metadata"]["M"] == 125
    assert "covariance_diagnostics" in result
    assert (tmp_path / "tables" / "base5_headline" / "covariance_numerical_diagnostics.csv").exists()
    assert (tmp_path / "tables" / "base5_headline" / "nested_full_coordinate_covariance_diagnostics.csv").exists()
    assert (tmp_path / "tables" / "base5_headline" / "gaussian_score_decomposition.csv").exists()
    assert (tmp_path / "tables" / "base5_headline" / "model_pair_score_decomposition.csv").exists()
    assert (tmp_path / "tables" / "base5_headline" / "covariance_regularization_sensitivity.csv").exists()
    assert set(
        [
            "evaluation_date",
            "model",
            "log_score",
            "constant_term",
            "logdet_term",
            "mahalanobis_term",
            "number_eigenvalues_clipped",
            "training_residual_count",
        ]
    ).issubset(set(result["score_decomposition"].columns))
