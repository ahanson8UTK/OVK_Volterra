from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ritz_certification as rc  # noqa: E402
import run_ritz_certification as ritz_run  # noqa: E402


def _synthetic_psd(seed: int = 1234, p: int = 8, n: int = 80) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = 0.6 * X + rng.normal(scale=0.7, size=(n, p))
    C = rc.sym((X.T @ X) / n)
    K = rc.sym((Y.T @ Y) / n)
    return C, K, 0.25


def test_theta_identity_and_certificate() -> None:
    C, K, rho = _synthetic_psd()
    op = rc.build_ridge_relative_operator(C, K, rho)
    R = 4
    comp = rc.ritz_compression(op.eigen.vectors, op.eigen.values, K, rho, R)
    assert comp.identity_error < 1.0e-10

    Q = rc.build_probe_soft(C, rho)
    q_full = rc.stress_trace(op.A, Q, 1.0)
    q_r = rc.compressed_stress(comp.theta, comp.V_R, Q, 1.0)
    assert 0.0 <= q_full <= 1.0
    assert 0.0 <= q_r <= 1.0
    cert = rc.ritz_certificate(op.A, comp.V_R, comp.theta, Q, 1.0)
    assert abs(q_full - q_r) <= cert.bound + 1.0e-8
    assert abs(cert.retained_mass + cert.omitted_mass - 1.0) < 1.0e-10
    assert cert.uncapped_bound >= cert.capped_bound - 1.0e-12


def test_full_rank_certificate_is_zero() -> None:
    C, K, rho = _synthetic_psd()
    op = rc.build_ridge_relative_operator(C, K, rho)
    p = C.shape[0]
    Q = rc.build_probe_soft(C, rho)
    comp = rc.ritz_compression(op.eigen.vectors, op.eigen.values, K, rho, p)
    cert = rc.ritz_certificate(op.A, comp.V_R, comp.theta, Q, 1.0)
    q_full = rc.stress_trace(op.A, Q, 1.0)
    q_p = rc.compressed_stress(comp.theta, comp.V_R, Q, 1.0)
    assert cert.omitted_mass < 1.0e-10
    assert cert.residual_norm < 1.0e-10
    assert cert.bound < 1.0e-10
    assert abs(q_full - q_p) < 1.0e-10


def test_probes_are_psd_trace_one() -> None:
    C, _K, rho = _synthetic_psd()
    p = C.shape[0]
    probes = [
        rc.build_probe_soft(C, rho),
        rc.build_probe_direction(C, np.eye(p)[0]),
        rc.build_probe_block(C, [1, 3, 5]),
    ]
    for Q in probes:
        vals = np.linalg.eigvalsh(rc.sym(Q))
        assert vals.min() > -1.0e-9
        assert abs(float(np.trace(Q)) - 1.0) < 1.0e-9


def test_interval_generalized_spectrum_matches_A_eigenvalues() -> None:
    C, K, rho = _synthetic_psd()
    op = rc.build_ridge_relative_operator(C, K, rho)
    vals = np.linalg.eigvalsh(op.A)
    interval = (float(vals.min()) - 1.0e-8, float(vals.max()) + 1.0e-8)
    result = rc.interval_generalized_spectrum(K, op.D, interval)
    assert result.backend == "scipy.linalg.eigh subset_by_value"
    assert np.allclose(np.sort(result.eigenvalues), np.sort(vals), atol=1.0e-8)
    assert float(result.residuals.max()) < 1.0e-8
    assert rc.severe_direction_count(result.eigenvalues, 1.0) == int(np.sum(vals >= 1.0 - rc.ASSERT_TOL))


def test_macro_eta_selector_is_outcome_independent() -> None:
    source = inspect.getsource(ritz_run.select_macro_eta)
    banned = ["severe", "ritz", "capture", "stress", "exact_error", "d_2", "d_4"]
    for token in banned:
        assert token not in source.lower()
    assert "blocked_predictive" in source
    assert "validation_block" in source


def test_generated_empirical_outputs_exist_and_are_consistent() -> None:
    out = ROOT / "results" / "ritz_certification"
    expected = [
        out / "tables" / "ritz_certification_main.csv",
        out / "tables" / "ritz_rank_diagnostics.csv",
        out / "tables" / "macro_temporal_resolvent_eta_grid.csv",
        out / "tables" / "macro_temporal_resolvent_state_diagnostics.csv",
        out / "tables" / "macro_ritz_interval_spectrum_data.csv",
        out / "tables" / "hurricane_selector_audit.csv",
        out / "tables" / "hurricane_ritz_certification.csv",
        out / "tables" / "lalonde_ritz_validation_data.csv",
        out / "figures" / "macro_ritz_interval_spectrum.pdf",
        out / "figures" / "lalonde_ritz_validation.pdf",
        out / "latex" / "tab_ritz_certification_main.tex",
        out / "latex" / "tab_hurricane_ritz_certification.tex",
        out / "metadata" / "ritz_certification_metadata.json",
    ]
    missing = [p for p in expected if not p.exists()]
    assert not missing

    main = pd.read_csv(out / "tables" / "ritz_certification_main.csv")
    assert len(main) == 3
    assert list(main.columns) == [
        "application",
        "probe",
        "p",
        "display_R",
        "retained_probe_mass",
        "p95_exact_error",
        "max_exact_error",
        "max_ritz_bound",
    ]
    assert (main["max_exact_error"] <= main["max_ritz_bound"] + 1.0e-8).all()
    assert "R_0.10" not in main.columns
    assert set(main["application"]) == {"LaLonde ATE", "Hurricane full fiscal/post probe"}

    lalonde = pd.read_csv(out / "tables" / "lalonde_ritz_validation_data.csv")
    assert {"q_full", "q_5", "retained_probe_mass", "uncapped_ritz_bound", "capped_ritz_bound", "bound_utilization"}.issubset(lalonde.columns)
    assert (lalonde["exact_error"] <= lalonde["capped_ritz_bound"] + 1.0e-8).all()
    expected_utilization = lalonde["exact_error"] / lalonde["uncapped_ritz_bound"].replace(0.0, np.nan)
    assert np.allclose(lalonde["bound_utilization"], expected_utilization.fillna(0.0), atol=1.0e-10)
    assert not np.allclose(lalonde["bound_utilization"], 1.0)
    assert set(lalonde["probe"]) == {"overall ATE", "prior earnings block", "education block"}

    macro = pd.read_csv(out / "tables" / "macro_interval_spectrum.csv")
    assert {"d_2", "d_4", "kappa5_2", "kappa5_4", "max_interval_residual"}.issubset(macro.columns)
    assert float(macro["max_interval_residual"].max()) < 1.0e-6
    eta_grid = pd.read_csv(out / "tables" / "macro_temporal_resolvent_eta_grid.csv")
    assert eta_grid["candidate_valid"].any()
    assert "blocked_predictive_neg_loglik" in eta_grid.columns
    assert "loo_frobenius_criterion" not in eta_grid.columns
    assert "publication_screen_pass" not in eta_grid.columns
    state_diag = pd.read_csv(out / "tables" / "macro_temporal_resolvent_state_diagnostics.csv")
    assert float(state_diag["numerical_rank"].median()) > 1.0
    sensitivity = pd.read_csv(out / "tables" / "macro_eta_sensitivity.csv")
    assert {"selected", "more_smoothed", "less_smoothed"}.intersection(set(sensitivity["role"]))

    h_a = pd.read_csv(out / "tables" / "hurricane_ritz_certification_panel_a.csv")
    assert {"probe", "trace_share", "retained_probe_mass", "p95_exact_error", "share_error_gt_0_05", "p95_ritz_subspace_residual_term"}.issubset(h_a.columns)
    assert set(h_a["p"]) == {72, 28}
    assert (h_a["max_exact_error"] <= h_a["max_capped_ritz_bound"] + 1.0e-8).all()
    h_b = pd.read_csv(out / "tables" / "hurricane_ritz_certification_panel_b.csv")
    assert float(h_b["max_generalized_eigenpair_residual"].max()) < 1.0e-6
    audit = pd.read_csv(out / "tables" / "hurricane_selector_audit.csv")
    assert len(audit) == 72
    assert int(audit["included_in_fiscal_post"].sum()) == 28
    metadata = json.loads((out / "metadata" / "ritz_certification_metadata.json").read_text(encoding="utf-8"))
    assert "temporal-resolvent benchmark" in metadata["surfaces"]["macro"]["full_K_source"]
    assert metadata["surfaces"]["hurricane_fiscal_post"]["dimension"] == 28
    assert metadata["interval_backend"] == "scipy.linalg.eigh subset_by_value"
    macro_meta = metadata["surfaces"]["macro"]["extra"]
    assert "blocked predictive Gaussian" in macro_meta["validation_criterion"]
    assert "severe-direction counts" in macro_meta["eta_not_selected_using"]
    assert not metadata["validation_provenance"]["publication_readiness_flags"].get("macro_uses_stress_screen_for_eta", True)
    assert metadata["validation_provenance"]["publication_readiness_flags"]["macro_omitted_from_main_table"]
    assert metadata["validation_provenance"]["psd_checks_passed"]
    assert metadata["validation_provenance"]["certificate_checks_passed"]


def test_generated_manuscript_patch_contains_required_theory_material() -> None:
    patch = (ROOT / "results" / "ritz_certification" / "latex" / "ritz_manuscript_patch_fragment.tex").read_text(encoding="utf-8")
    assert r"\boxed{" in patch
    assert r"\label{eq:ritz-stress-bound}" in patch
    assert "Appendix~\\ref{app:ritz-bound-proof}" in patch
    assert r"\section{Spectral Compression for Interpretation}" in patch
    assert r"\subsection{Ritz residual certification}" in patch
    assert r"\subsection{Proof of the Ritz stress bound}" in patch
    assert "scipy.linalg.eigh" in patch
    assert "FEAST" in patch and "no FEAST computation is used" in patch
    assert "48-dimensional fiscal" not in patch
    assert "4 \\times 12 = 48" not in patch
