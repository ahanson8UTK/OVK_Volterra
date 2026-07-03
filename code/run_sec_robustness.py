#!/usr/bin/env python3
"""Run the SEC macro-state geometry robustness check for the OVK pipeline."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_SEED = 20260602
H = 24
L_LP = 12
RANK = 5
BOOT_BLOCK_LEN = 18


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "reportlab": "reportlab",
    "pypdf": "pypdf",
}


def check_dependencies() -> None:
    """Fail clearly if a required package is missing."""
    missing = [name for module, name in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            "Missing required package(s): "
            + ", ".join(missing)
            + ". Install the repository requirements before running SEC robustness."
        )


def import_publication_module(repo_root: Path, results_dir: Path):
    """Import publication-grade helpers without running the publication pipeline."""
    os.environ.setdefault("OVK_PUBLICATION_ROOT", str(results_dir / "publication_grade_ovk"))
    os.environ.setdefault("OVK_DATA_ZIP", str(repo_root / "data_raw" / "data.zip"))
    sys.path.insert(0, str(repo_root / "code"))
    import run_publication_grade_ovk as pub  # type: ignore

    return pub


def load_panel(repo_root: Path, results_dir: Path) -> pd.DataFrame:
    """Load the processed three-shock monthly panel."""
    candidates = [
        results_dir / "data_processed" / "processed_panel_three_shock_definitions.csv",
        results_dir / "top5_full_appended_results_pack" / "data_processed" / "processed_panel_three_shock_definitions.csv",
        repo_root / "data_processed" / "processed_panel_three_shock_definitions.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raise FileNotFoundError("Could not locate processed_panel_three_shock_definitions.csv")


def load_rank_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "publication_grade_ovk" / "outputs" / "tables" / "publication_grade_rank_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Publication-grade rank summary not found: {path}")


def load_baseline_path(results_dir: Path, variant: str) -> pd.DataFrame:
    """Load publication-grade baseline tau and A diagonal paths."""
    if variant == "median_fallback":
        candidates = [
            results_dir
            / "publication_grade_ovk"
            / "outputs"
            / "tables"
            / "publication_grade_headline_state_path.csv",
            results_dir
            / "top5_full_appended_results_pack"
            / "top5_baseline_state_space_results"
            / "tables"
            / "state_space_A_t_top5_drift_estimates_with_bands.csv",
        ]
    else:
        candidates = [
            results_dir
            / "top5_shock_robustness"
            / "outputs"
            / "tables"
            / f"{variant}_tau_and_A_diagonals.csv",
            results_dir
            / "top5_full_appended_results_pack"
            / "robustness_comparison_results"
            / "tables"
            / f"{variant}_tau_and_A_diagonals.csv",
            results_dir
            / "publication_grade_ovk"
            / "shock_robustness_outputs"
            / "tables"
            / f"{variant}_tau_and_A_diagonals.csv",
        ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            if "trace_A_over_R" in df.columns and "tau" not in df.columns:
                df = df.rename(columns={"trace_A_over_R": "tau"})
            for j in range(1, RANK + 1):
                compat = f"A{j}{j}_basis{j}"
                if compat in df.columns and f"A{j}{j}" not in df.columns:
                    df = df.rename(columns={compat: f"A{j}{j}"})
            need = {"date", "tau"} | {f"A{j}{j}" for j in range(1, RANK + 1)}
            missing = sorted(need - set(df.columns))
            if missing:
                raise ValueError(f"Baseline path {path} is missing columns: {missing}")
            return df[["date", "tau", *[f"A{j}{j}" for j in range(1, RANK + 1)]]].copy()
    raise FileNotFoundError(f"Could not locate publication-grade baseline path for {variant}")


def circular_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate circular moving-block bootstrap indices."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
    idx: list[int] = []
    for s in starts:
        idx.extend(((int(s) + np.arange(block_len)) % n).tolist())
    return np.asarray(idx[:n], dtype=int)


def alpha_l_candidates(n_obs: int) -> list[int]:
    cap = int(max(1, min(30, np.floor(n_obs / 5))))
    vals = sorted({min(cap, L) for L in [5, 10, 15, 20, 30] if min(cap, L) >= 1})
    return vals


def prepare_output_dirs(sec_root: Path, clean: bool) -> dict[str, Path]:
    """Create the SEC output directory tree."""
    if clean and sec_root.exists():
        shutil.rmtree(sec_root)
    paths = {
        "root": sec_root,
        "code": sec_root / "code",
        "data": sec_root / "data",
        "tables": sec_root / "tables",
        "charts": sec_root / "charts",
        "reports": sec_root / "reports",
        "math": sec_root / "math_appendix",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_run_order(repo_root: Path, bootstrap_draws: int) -> Path:
    """Write exact SEC reproduction instructions."""
    path = repo_root / "RUN_ORDER_SEC.md"
    text = f"""# SEC Robustness Run Order

This robustness check is now included in the default `ovk_pipeline.py run-all` sequence. It reads the publication-grade OVK outputs and does not replace or overwrite the baseline estimator. Use this standalone order only when rerunning the SEC pack by itself.

1. From the repository root, install the existing requirements if needed:

   `python -m pip install -r code/requirements.txt`

2. Run the SEC robustness pack:

   `python code/run_sec_robustness.py --bootstrap-draws {bootstrap_draws} --clean`

   If the pipeline outputs were written somewhere other than `results`, add `--results-dir path\\to\\outdir`.

3. Run the numerical checks:

   `python tests/test_sec_robustness.py`

Primary outputs:

- `monthly_ovk_top5_with_SEC_robustness_report.pdf`
- `monthly_ovk_top5_with_SEC_robustness_full_pack.zip`
- `monthly_ovk_top5_with_SEC_robustness_contact_sheet.jpg`
- `sec_robustness_results/reports/sec_robustness_appendix.pdf`
- `sec_robustness_results/tables/*.csv`

The SEC comparison baseline is the publication-grade `tau_t` path and publication-grade rank-five shock-robustness paths. Deprecated legacy top-five estimator outputs are not used as the benchmark.
"""
    path.write_text(text, encoding="utf-8")
    return path


def select_sec_level_candidate(
    variant: str,
    label: str,
    S: np.ndarray,
    Z: np.ndarray,
    lambda_grid: list[float],
    alpha_spd_grid: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select SEC graph and level-model hyperparameters by blocked CV."""
    from sec_geometry import build_graph_laplacian, candidate_knn_values, scalar_features
    from sec_ovk import (
        A_from_log_predictions,
        blocked_folds,
        covariance_proxy_loss,
        evaluate_ridge_grid,
        gtilde_and_log_observations,
        log_proxy_loss,
        penalty_vector,
        ridge_fit,
    )

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    T = len(S)
    L_candidates = alpha_l_candidates(T)
    k_candidates = candidate_knn_values(T)
    folds = blocked_folds(T)

    for k in k_candidates:
        for alpha_density in [0.0, 0.5, 1.0]:
            geom = build_graph_laplacian(S, k=k, alpha_density=alpha_density, l_max=max(L_candidates))
            for L_sec in L_candidates:
                Phi = scalar_features(geom.eigenfunctions, L_sec)
                eigpen = geom.eigenvalues[1 : L_sec + 1]
                penalties = penalty_vector(Phi.shape[1], eigpen)
                for alpha_spd in alpha_spd_grid:
                    _, Gtilde, y = gtilde_and_log_observations(Z, alpha_spd)
                    rec = evaluate_ridge_grid(Phi, y, lambda_grid, folds=folds, eigenvalues=eigpen, label="sec_level")
                    for _, r in rec.iterrows():
                        B = ridge_fit(Phi, y, float(r["ridge_lambda"]), penalties)
                        yhat = Phi @ B
                        A, tau = A_from_log_predictions(yhat, RANK)
                        eigs = np.linalg.eigvalsh(A)
                        row = {
                            "variant": variant,
                            "label": label,
                            "model": "sec_level",
                            "k": k,
                            "alpha_density": alpha_density,
                            "L_sec": L_sec,
                            "alpha_spd_proxy": alpha_spd,
                            "M_sec_pairs": np.nan,
                            **r.to_dict(),
                            "log_proxy_loss_in_sample": log_proxy_loss(y, yhat),
                            "covariance_proxy_loss_in_sample": covariance_proxy_loss(Gtilde, A),
                            "min_eig_A": float(np.min(eigs)),
                            "max_trace_A_over_5": float(np.max(tau)),
                            "mean_A_error": float(np.linalg.norm(A.mean(axis=0) - np.eye(RANK), ord="fro")),
                            "stable": bool((np.max(tau) < 20.0) and (np.min(eigs) > 1e-6) and np.isfinite(yhat).all()),
                        }
                        rows.append(row)
                        if row["stable"] and (best is None or row["cv_loss"] < best["row"]["cv_loss"]):
                            best = {
                                "row": row,
                                "geom": geom,
                                "Phi": Phi,
                                "eigpen": eigpen,
                                "alpha_spd": alpha_spd,
                                "Gtilde": Gtilde,
                                "y": y,
                                "B": B,
                                "yhat": yhat,
                                "A": A,
                                "tau": tau,
                            }
    if best is None:
        raise RuntimeError(f"No stable SEC level model found for {variant}")
    return best, pd.DataFrame(rows)


def fit_directional_diagnostic(
    variant: str,
    label: str,
    S: np.ndarray,
    geom: Any,
    L_sec: int,
    y: np.ndarray,
    lambda_grid: list[float],
    data_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Fit SEC directional drift diagnostics for several pair counts."""
    from sec_geometry import directional_features, local_eigenfunction_gradients
    from sec_ovk import fit_directional_drift_model

    gradients = local_eigenfunction_gradients(S, geom.eigenfunctions, geom.neighbor_indices, geom.affinity, L_sec)
    np.savez_compressed(data_dir / f"sec_laplacian_gradients_{variant}.npz", gradients=gradients)
    records = []
    best: dict[str, Any] | None = None
    best_Xi = None
    best_pairs = pd.DataFrame()
    for M_pairs in [25, 50, 100, 150]:
        Xi, pairs = directional_features(S, geom.eigenfunctions, gradients, geom.eigenvalues, L_sec, M_pairs)
        fit, dy = fit_directional_drift_model(y, Xi, lambda_grid)
        for _, row in fit.records.iterrows():
            rec = {
                "variant": variant,
                "label": label,
                "model": "sec_directional_drift",
                "k": geom.k,
                "alpha_density": geom.alpha_density,
                "L_sec": L_sec,
                "alpha_spd_proxy": np.nan,
                "M_sec_pairs": M_pairs,
                **row.to_dict(),
                "directional_target_loss_in_sample": float(np.mean((dy - fit.fitted) ** 2)),
                "stable": True,
            }
            records.append(rec)
        if best is None or fit.cv_loss < best["fit"].cv_loss:
            best = {"fit": fit, "M_sec_pairs": M_pairs, "dy": dy}
            best_Xi = Xi
            best_pairs = pairs.copy()
    assert best is not None and best_Xi is not None
    return pd.DataFrame(records), best_Xi, best_pairs, best


def run_bootstrap(
    variant: str,
    label: str,
    dates: pd.Series,
    Phi: np.ndarray,
    E: np.ndarray,
    y: np.ndarray,
    Kbar: np.ndarray,
    Vbase: np.ndarray,
    score_lambda: float,
    level_lambda: float,
    eigpen: np.ndarray,
    bootstrap_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run conditional moving-block bootstrap with fixed SEC geometry."""
    from sec_comparisons import principal_angles
    from sec_ovk import A_from_log_predictions, penalty_vector, ridge_fit, top_eigendecomposition

    rng = np.random.default_rng(seed)
    T = len(Phi)
    pen = penalty_vector(Phi.shape[1], eigpen)
    trace_total = max(float(np.trace(Kbar)), 1e-12)
    share_rows = []
    angle_rows = []
    tau_draws = np.empty((bootstrap_draws, T), dtype=float)
    for b in range(bootstrap_draws):
        ix = circular_block_indices(T, BOOT_BLOCK_LEN, rng)
        B_score = ridge_fit(Phi[ix], E[ix], score_lambda, pen)
        Ehat = Phi @ B_score
        K_score = (Ehat.T @ Ehat) / T
        vals, V_score, _ = top_eigendecomposition(K_score, RANK)
        angles = principal_angles(Vbase, V_score)
        share_rows.append(
            {
                "variant": variant,
                "label": label,
                "draw": b,
                "sec_score_top5_trace_share_of_total": float(vals[:RANK].sum() / trace_total),
                "sec_score_internal_top5_share": float(vals[:RANK].sum() / max(vals.sum(), 1e-12)),
            }
        )
        angle_rows.append(
            {
                "variant": variant,
                "label": label,
                "draw": b,
                "max_principal_angle_degrees": float(np.max(angles)),
                "mean_principal_angle_degrees": float(np.mean(angles)),
                **{f"angle_{j+1}_degrees": float(angles[j]) for j in range(len(angles))},
            }
        )
        B_level = ridge_fit(Phi[ix], y[ix], level_lambda, pen)
        A_b, tau_b = A_from_log_predictions(Phi @ B_level, RANK)
        tau_draws[b] = tau_b
    tau_q = np.nanquantile(tau_draws, [0.05, 0.50, 0.95], axis=0)
    tau_bands = pd.DataFrame(
        {
            "variant": variant,
            "label": label,
            "date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
            "tau_sec_p05": tau_q[0],
            "tau_sec_median": tau_q[1],
            "tau_sec_p95": tau_q[2],
        }
    )
    return pd.DataFrame(share_rows), pd.DataFrame(angle_rows), tau_bands


def build_summary_text(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Create concise report prose from SEC tables."""
    tau = tables.get("sec_tau_path_comparison", pd.DataFrame())
    diag = tables.get("sec_basis_diag_path_correlations", pd.DataFrame())
    trace = tables.get("sec_top5_trace_share_comparison", pd.DataFrame())
    lines = [
        "This SEC section is a final robustness check, not the main estimator. It asks whether low-rank monetary-transmission covariance geometry remains similar when the rank-five A_t path is regularized as a smooth function of predetermined macro-state graph geometry.",
        "The benchmark is the publication-grade algorithm and its tau_t path. Deprecated legacy top-five state-space outputs are not used as the comparison target.",
    ]
    if not trace.empty:
        vals = trace[["label", "baseline_top5_trace_share", "sec_score_top5_trace_share_of_total"]].copy()
        snippets = [f"{r.label}: {r.baseline_top5_trace_share:.3f} baseline, {r.sec_score_top5_trace_share_of_total:.3f} SEC-score" for r in vals.itertuples()]
        lines.append("Top-five trace-share comparison: " + "; ".join(snippets) + ".")
    if not tau.empty:
        snippets = [f"{r.label}: tau corr {r.tau_corr:.3f}, SEC max {r.sec_tau_max_month}" for r in tau.itertuples()]
        lines.append("Tau-path robustness: " + "; ".join(snippets) + ".")
    if not diag.empty:
        b45 = diag[diag["basis"].isin([4, 5])]
        snippets = [f"{r.label} basis {int(r.basis)} corr {r.A_diag_path_corr:.3f}" for r in b45.itertuples()]
        lines.append("Basis 4 and basis 5 remain the fragility checks: " + "; ".join(snippets) + ".")
    lines.append("SEC estimates state-conditioned response-score covariance geometry; it is not a standalone structural causal IRF.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEC robustness pack for publication-grade OVK outputs.")
    parser.add_argument("--repo-root", default=str(_repo_root_from_file()), help="Repository root")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Pipeline output directory containing publication-grade and top-five appended outputs.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=1000, help="Moving-block bootstrap draws")
    parser.add_argument("--clean", action="store_true", help="Remove existing sec_robustness_results before running")
    args = parser.parse_args()

    check_dependencies()
    repo_root = Path(args.repo_root).resolve()
    results_dir = _resolve_path(args.results_dir, repo_root)
    np.random.seed(RANDOM_SEED)
    sec_root = repo_root / "sec_robustness_results"
    paths = prepare_output_dirs(sec_root, clean=args.clean)
    run_order_path = write_run_order(repo_root, args.bootstrap_draws)

    pub = import_publication_module(repo_root, results_dir)
    from sec_comparisons import (
        basis_diag_correlations,
        principal_angles,
        state_manifold_extreme_states,
        tau_path_comparison,
        top_amplification_months,
    )
    from sec_geometry import eigenfunction_frame, scalar_features
    from sec_ovk import (
        build_macro_state_matrix,
        centered_kernel,
        construct_rank5_kernel_mean,
        covariance_proxy_loss,
        filtered_score_kernel,
        fit_sec_level_model,
        gtilde_and_log_observations,
        log_proxy_loss,
        top_eigendecomposition,
        var1_oos_loss,
        whitened_scores,
    )
    from sec_reporting import (
        build_sec_html,
        build_sec_pdf,
        copy_sec_code,
        create_contact_sheet,
        create_final_zip,
        make_all_charts,
        merge_pdfs,
        verify_pdf_readable,
        write_manifest,
        write_math_appendix,
    )

    panel = load_panel(repo_root, results_dir)
    rank_summary = load_rank_summary(results_dir)
    variants = [
        ("median_fallback", "MP_median with fallback", "MP_median_fallback", "CBI_median_fallback"),
        ("mp_pm_only", "MP_pm only", "MP_pm", "CBI_pm"),
        ("event_manual", "Event-level shocks aggregated manually", "MP_event_manual", "CBI_event_manual"),
    ]
    lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    alpha_spd_grid = [0.10, 0.25, 0.40]

    results: dict[str, dict[str, Any]] = {}
    state_frames = []
    model_selection_frames = []
    eigenvalue_rows = []
    eigenfunction_frames = []
    mean_rows = []
    trace_rows = []
    angle_rows = []
    tau_rows = []
    diag_frames = []
    top_frames = []
    top_diag_rows = []
    oos_rows = []
    extreme_frames = []
    boot_share_frames = []
    boot_angle_frames = []
    boot_tau_frames = []

    for var_i, (variant, label, shock_col, cbi_col) in enumerate(variants):
        print(f"SEC robustness: building publication-grade scores for {label}", flush=True)
        scores = pub.build_lp_scores(panel, shock_col, cbi_col, H=H, L=L_LP)
        Q_scores = np.asarray(scores["Q_scores"], dtype=float)
        dates = pd.to_datetime(scores["dates"])
        baseline_path = load_baseline_path(results_dir, variant)
        Kbar, beta, E = centered_kernel(Q_scores)
        eigvals, V, shares = top_eigendecomposition(Kbar, RANK)
        Z = whitened_scores(E, V, eigvals)
        whiten_error = float(np.linalg.norm((Z.T @ Z) / len(Z) - np.eye(RANK), ord="fro"))
        if whiten_error > 1e-6:
            print(f"Whitened score warning for {variant}: Frobenius error {whiten_error:.3e}", flush=True)

        state_df, S = build_macro_state_matrix(panel, np.asarray(scores["valid_idx"], dtype=int), variant)
        state_frames.append(state_df)

        best, model_records = select_sec_level_candidate(variant, label, S, Z, lambda_grid, alpha_spd_grid)
        model_selection_frames.append(model_records)
        geom = best["geom"]
        L_sec = int(best["row"]["L_sec"])
        Phi = best["Phi"]
        eigpen = best["eigpen"]
        _, Gtilde, y = gtilde_and_log_observations(Z, float(best["alpha_spd"]))
        level_fit = fit_sec_level_model(
            y,
            Phi,
            RANK,
            [float(best["row"]["ridge_lambda"])],
            eigpen,
        )
        A_sec = level_fit.A
        tau_sec = level_fit.tau

        K_score, Ehat, score_fit = filtered_score_kernel(E, Phi, lambda_grid, eigpen)
        score_vals, V_score, score_shares = top_eigendecomposition(K_score, RANK)
        K_rank5 = construct_rank5_kernel_mean(V, eigvals, A_sec)
        K_retained = V @ np.diag(eigvals[:RANK]) @ V.T
        rank5_mean_error = float(np.linalg.norm(K_rank5 - K_retained, ord="fro"))

        directional_records, Xi, pair_df, directional_best = fit_directional_diagnostic(
            variant, label, S, geom, L_sec, y, lambda_grid, paths["data"]
        )
        model_selection_frames.append(directional_records)

        eig_cols = {
            "variant": variant,
            "label": label,
            "k": geom.k,
            "alpha_density": geom.alpha_density,
            "selected_L": L_sec,
        }
        for j, val in enumerate(geom.eigenvalues[: L_sec + 1]):
            eigenvalue_rows.append({**eig_cols, "eigen_index": j, "eigenvalue": float(val)})
        eigenfunction_frames.append(eigenfunction_frame(dates, variant, geom.eigenfunctions, L_sec))

        scalar_df = pd.DataFrame(Phi, columns=["intercept", *[f"phi_{j}" for j in range(1, L_sec + 1)]])
        scalar_df.insert(0, "shock_definition", variant)
        scalar_df.insert(0, "date", dates.dt.strftime("%Y-%m-%d"))
        scalar_df.to_csv(paths["data"] / f"sec_scalar_features_{variant}.csv", index=False)
        dir_df = pd.DataFrame(Xi, columns=pair_df["feature"].tolist())
        dir_df.insert(0, "shock_definition", variant)
        dir_df.insert(0, "date", dates.iloc[:-1].dt.strftime("%Y-%m-%d"))
        dir_df.to_csv(paths["data"] / f"sec_directional_features_{variant}.csv", index=False)
        pair_df.to_csv(paths["data"] / f"sec_directional_feature_pairs_{variant}.csv", index=False)

        np.savez_compressed(
            paths["data"] / f"Kbar_SEC_score_{variant}.npz",
            Kbar_SEC_score=K_score,
            Ehat=Ehat,
            eigvals=score_vals,
            V=V_score,
            selected_L=L_sec,
            ridge_lambda=score_fit.lambda_value,
        )
        np.savez_compressed(
            paths["data"] / f"Kbar_SEC_rank5_{variant}.npz",
            Kbar_SEC_rank5=K_rank5,
            Kbar_retained_publication_grade=K_retained,
            A_SEC=A_sec,
            tau_SEC=tau_sec,
            V=V,
            eigvals=eigvals[:RANK],
            selected_L=L_sec,
            alpha_spd=float(best["alpha_spd"]),
            ridge_lambda=level_fit.ridge.lambda_value,
        )

        angles = principal_angles(V, V_score)
        angle_rows.append(
            {
                "variant": variant,
                "label": label,
                "max_angle_degrees": float(np.max(angles)),
                "mean_angle_degrees": float(np.mean(angles)),
                **{f"angle_{j+1}_degrees": float(angles[j]) for j in range(len(angles))},
            }
        )
        rank_row = rank_summary[(rank_summary["variant"].eq(variant)) & (rank_summary["rank"].eq(RANK))]
        baseline_share = float(rank_row["retained_trace_share"].iloc[0]) if len(rank_row) else float(shares[:RANK].sum())
        trace_rows.append(
            {
                "variant": variant,
                "label": label,
                "baseline_top5_trace_share": baseline_share,
                "baseline_recomputed_top5_trace_share": float(shares[:RANK].sum()),
                "sec_score_top5_trace_share_of_total": float(score_vals[:RANK].sum() / max(np.trace(Kbar), 1e-12)),
                "sec_score_internal_top5_share": float(score_vals[:RANK].sum() / max(score_vals.sum(), 1e-12)),
                "sec_rank5_kernel_trace_share": float(eigvals[:RANK].sum() / max(eigvals.sum(), 1e-12)),
            }
        )
        mean_rows.append(
            {
                "variant": variant,
                "label": label,
                "baseline_trace_Kbar": float(np.trace(Kbar)),
                "sec_score_trace": float(np.trace(K_score)),
                "sec_score_trace_share_of_baseline_total": float(np.trace(K_score) / max(np.trace(Kbar), 1e-12)),
                "sec_rank5_mean_recovery_fro_error": rank5_mean_error,
                "mean_A_SEC_minus_I_fro_error": level_fit.mean_A_error,
                "whitened_score_cov_minus_I_fro_error": whiten_error,
                "selected_k": geom.k,
                "selected_alpha_density": geom.alpha_density,
                "selected_L": L_sec,
                "selected_alpha_spd_proxy": float(best["alpha_spd"]),
                "level_ridge_lambda": level_fit.ridge.lambda_value,
                "score_ridge_lambda": score_fit.lambda_value,
            }
        )

        tau_rows.append(tau_path_comparison(variant, label, dates, tau_sec, baseline_path))
        top_df, top_diag = top_amplification_months(variant, label, dates, tau_sec, A_sec, baseline_path)
        top_frames.append(top_df)
        top_diag_rows.append(top_diag)
        diag_frames.append(basis_diag_correlations(variant, label, dates, A_sec, baseline_path))
        embedding = geom.eigenfunctions[:, 1:3] if geom.eigenfunctions.shape[1] >= 3 else np.column_stack([geom.eigenfunctions[:, 1], np.zeros(len(S))])
        extreme_frames.append(state_manifold_extreme_states(variant, label, dates, embedding, S, tau_sec))

        oos_rows.extend(
            [
                {"variant": variant, "label": label, "model": "publication_grade_log_VAR_proxy", "target": "y_t", "loss": var1_oos_loss(y)},
                {"variant": variant, "label": label, "model": "SEC_scalar_level", "target": "y_t", "loss": level_fit.ridge.cv_loss},
                {"variant": variant, "label": label, "model": "naive_A_equals_I", "target": "y_t", "loss": float(np.mean(y * y))},
                {
                    "variant": variant,
                    "label": label,
                    "model": "SEC_directional_drift",
                    "target": "Delta y_t",
                    "loss": float(directional_best["fit"].cv_loss),
                    "M_sec_pairs": int(directional_best["M_sec_pairs"]),
                },
                {
                    "variant": variant,
                    "label": label,
                    "model": "SEC_covariance_proxy_in_sample",
                    "target": "Gtilde_t",
                    "loss": covariance_proxy_loss(Gtilde, A_sec),
                },
            ]
        )

        share_b, angle_b, tau_b = run_bootstrap(
            variant,
            label,
            dates,
            Phi,
            E,
            y,
            Kbar,
            V,
            score_fit.lambda_value,
            level_fit.ridge.lambda_value,
            eigpen,
            args.bootstrap_draws,
            RANDOM_SEED + 1000 * (var_i + 1),
        )
        boot_share_frames.append(share_b)
        boot_angle_frames.append(angle_b)
        boot_tau_frames.append(tau_b)

        results[variant] = {
            "variant": variant,
            "label": label,
            "dates": dates,
            "scores": scores,
            "Kbar": Kbar,
            "beta": beta,
            "E": E,
            "eigvals": eigvals,
            "shares": shares,
            "V": V,
            "Z": Z,
            "Gtilde": Gtilde,
            "y": y,
            "geom": geom,
            "selected_L": L_sec,
            "Phi": Phi,
            "A_sec": A_sec,
            "tau_sec": tau_sec,
            "baseline_path": baseline_path,
            "embedding": embedding,
        }

    top_diag_df = pd.DataFrame(top_diag_rows)[
        [
            "variant",
            "label",
            "top10_overlap_count",
            "top10_overlap_months",
            "march_2020_sec_rank",
            "march_2020_baseline_rank",
        ]
    ]
    tables: dict[str, pd.DataFrame] = {
        "sec_model_selection": pd.concat(model_selection_frames, ignore_index=True),
        "sec_mean_kernel_comparison": pd.DataFrame(mean_rows),
        "sec_top5_trace_share_comparison": pd.DataFrame(trace_rows),
        "sec_principal_angles": pd.DataFrame(angle_rows),
        "sec_tau_path_comparison": pd.DataFrame(tau_rows).merge(top_diag_df, on=["variant", "label"], how="left"),
        "sec_basis_diag_path_correlations": pd.concat(diag_frames, ignore_index=True),
        "sec_top_amplification_months": pd.concat(top_frames, ignore_index=True),
        "sec_oos_loss_comparison": pd.DataFrame(oos_rows),
        "sec_state_manifold_extreme_states": pd.concat(extreme_frames, ignore_index=True),
        "sec_laplacian_eigenvalues": pd.DataFrame(eigenvalue_rows),
        "sec_laplacian_eigenfunctions": pd.concat(eigenfunction_frames, ignore_index=True),
        "sec_bootstrap_top5_trace_share": pd.concat(boot_share_frames, ignore_index=True),
        "sec_bootstrap_principal_angles": pd.concat(boot_angle_frames, ignore_index=True),
        "sec_bootstrap_tau_bands": pd.concat(boot_tau_frames, ignore_index=True),
    }
    pd.concat(state_frames, ignore_index=True).to_csv(paths["data"] / "sec_macro_state_matrix.csv", index=False)
    for name, df in tables.items():
        out = paths["tables"] / f"{name}.csv"
        df.to_csv(out, index=False)
    # Also keep graph eigenfunction tables under data because they are feature inputs.
    tables["sec_laplacian_eigenvalues"].to_csv(paths["data"] / "sec_laplacian_eigenvalues.csv", index=False)
    tables["sec_laplacian_eigenfunctions"].to_csv(paths["data"] / "sec_laplacian_eigenfunctions.csv", index=False)

    from sec_reporting import make_all_charts

    chart_paths = make_all_charts(results, tables, paths["charts"])
    summary_text = build_summary_text(tables)
    appendix_pdf = paths["reports"] / "sec_robustness_appendix.pdf"
    appendix_html = paths["reports"] / "sec_robustness_appendix.html"
    build_sec_pdf(appendix_pdf, tables, chart_paths, summary_text)
    build_sec_html(appendix_html, tables, chart_paths, summary_text)
    write_math_appendix(paths["math"] / "sec_math_appendix.tex", paths["math"] / "sec_math_appendix.pdf")
    copy_sec_code(repo_root, paths["code"])

    base_pdf = results_dir / "top5_full_appended_results_pack" / "reports" / "monthly_ovk_top5_full_appended_report.pdf"
    if not base_pdf.exists():
        base_pdf = results_dir / "reports" / "top5_full_appended_report.pdf"
    if not base_pdf.exists():
        raise FileNotFoundError("Could not locate base appended top-five PDF to merge")
    final_pdf = repo_root / "monthly_ovk_top5_with_SEC_robustness_report.pdf"
    merge_pdfs(base_pdf, appendix_pdf, final_pdf)
    verify_pdf_readable(appendix_pdf)
    verify_pdf_readable(final_pdf)
    contact_sheet = repo_root / "monthly_ovk_top5_with_SEC_robustness_contact_sheet.jpg"
    create_contact_sheet(chart_paths, contact_sheet)
    final_zip = repo_root / "monthly_ovk_top5_with_SEC_robustness_full_pack.zip"

    metadata = {
        "random_seed": RANDOM_SEED,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_block_length": BOOT_BLOCK_LEN,
        "results_dir": str(results_dir),
        "baseline": "publication_grade_tau_t_paths",
        "smoothing": {
            "operator": "graph_resolvent",
            "graph": "self_tuning_knn_symmetric_normalized_laplacian",
            "basis": "laplacian_eigenfunctions",
            "selection": "blocked_cv_ridge",
        },
        "deprecated_top5_used_as_benchmark": False,
        "final_pdf": str(final_pdf),
        "final_zip": str(final_zip),
        "manifest_rows": None,
    }
    (paths["root"] / "sec_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest = write_manifest(paths["root"], paths["tables"] / "sec_file_manifest.csv")
    metadata["manifest_rows"] = int(len(manifest))
    (paths["root"] / "sec_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = write_manifest(paths["root"], paths["tables"] / "sec_file_manifest.csv")
    create_final_zip(
        repo_root,
        paths["root"],
        final_zip,
        extra_files=[run_order_path, final_pdf, contact_sheet, repo_root / "code" / "requirements.txt"],
    )

    trace = tables["sec_top5_trace_share_comparison"]
    angles = tables["sec_principal_angles"]
    tau = tables["sec_tau_path_comparison"]
    diag = tables["sec_basis_diag_path_correlations"]
    print("SEC robustness completed.")
    print("Top-five trace share, baseline vs SEC:")
    print(trace[["label", "baseline_top5_trace_share", "sec_score_top5_trace_share_of_total", "sec_rank5_kernel_trace_share"]].to_string(index=False))
    print("Principal angles:")
    print(angles[["label", "max_angle_degrees", "mean_angle_degrees"]].to_string(index=False))
    print("Tau path correlations:")
    print(tau[["label", "tau_corr", "tau_rmse", "march_2020_sec_rank", "march_2020_baseline_rank"]].to_string(index=False))
    print("March 2020 ranks:")
    print(tau[["label", "march_2020_sec_rank", "march_2020_baseline_rank"]].to_string(index=False))
    print("Basis 4 and basis 5 diagonal-path correlations:")
    print(diag[diag["basis"].isin([4, 5])][["label", "basis", "A_diag_path_corr"]].to_string(index=False))
    print("Output ZIP path:", final_zip)
    print("Output PDF path:", final_pdf)


if __name__ == "__main__":
    main()
