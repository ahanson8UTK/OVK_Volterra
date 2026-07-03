#!/usr/bin/env python3
"""Generate corrected empirical Ritz-certification outputs."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ritz_certification as rc  # noqa: E402


CONFIG: dict[str, Any] = {
    "lambda_main": 1.0,
    "certification_epsilons": [0.10, 0.05],
    "severe_thresholds": [2.0, 4.0],
    "display_rank": 5,
    "hurricane_main_ranks": [5, 15],
    "hurricane_rank_grid": [5, 10, 15],
    "macro_eta_grid_size": 31,
    "macro_validation_block_months": 12,
    "macro_eta_lower_limit": 1.0e-12,
    "macro_eta_upper_limit": 1.0e6,
    "macro_eta_extension_factor": 100.0,
    "macro_eta_max_extensions": 4,
    "macro_lambda_grid": np.logspace(-2.0, 2.0, 200).tolist(),
    "macro_selected_dates": ["2007-12", "2014-10", "2020-03"],
    "interval_residual_tolerance": 1.0e-6,
}

ETA_NOT_SELECTED_USING = [
    "severe-direction counts",
    "Ritz approximation errors",
    "rank-five capture",
    "plotted stress variation",
]

OUT = ROOT / "results" / "ritz_certification"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LATEX = OUT / "latex"
META = OUT / "metadata"
for _path in [TABLES, FIGURES, LATEX, META]:
    _path.mkdir(parents=True, exist_ok=True)


@dataclass
class SurfaceData:
    key: str
    application: str
    surface: str
    principal_probe: str
    C: np.ndarray
    K_states: np.ndarray
    state_frame: pd.DataFrame
    rho: float
    Q: np.ndarray
    source_files: list[str]
    state_mask_note: str
    extra: dict[str, Any]


@dataclass
class SurfaceEvaluation:
    eig: rc.SymmetricEigendecomposition
    K_v_stack: np.ndarray
    A_stack: np.ndarray
    Q_v: np.ndarray
    rank_summary: pd.DataFrame
    frames: dict[int, pd.DataFrame]
    validation: dict[str, Any]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def archive_previous_invalid_outputs() -> str:
    """Move old rank-one-proxy Ritz outputs aside when their metadata is present."""
    metadata_path = META / "ritz_certification_metadata.json"
    if not metadata_path.exists():
        return ""
    text = metadata_path.read_text(encoding="utf-8", errors="ignore").lower()
    if "monthly score proxy" not in text and "rank-one" not in text:
        return ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = OUT / "superseded_invalid_proxy" / stamp
    candidates = [
        TABLES / "ritz_certification_main.csv",
        TABLES / "macro_ritz_interval_spectrum_data.csv",
        TABLES / "macro_interval_spectrum.csv",
        TABLES / "hurricane_ritz_certification.csv",
        TABLES / "hurricane_ritz_certification_panel_a.csv",
        TABLES / "hurricane_ritz_certification_panel_b.csv",
        TABLES / "lalonde_ritz_validation_data.csv",
        FIGURES / "macro_ritz_interval_spectrum.pdf",
        FIGURES / "macro_ritz_interval_spectrum.png",
        FIGURES / "lalonde_ritz_validation.pdf",
        FIGURES / "lalonde_ritz_validation.png",
        LATEX / "tab_ritz_certification_main.tex",
        LATEX / "tab_hurricane_ritz_certification.tex",
        LATEX / "ritz_paper_insertions.tex",
        LATEX / "ritz_standalone_check.tex",
        metadata_path,
        META / "paper_integration_status.json",
    ]
    moved = []
    for src in candidates:
        if src.exists():
            target = dest / src.relative_to(OUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
            moved.append(rel(target))
    if moved:
        write_json(dest / "archive_note.json", {"reason": "Previous outputs used invalid rank-one macro proxy.", "moved": moved})
    return rel(dest) if moved else ""


def fmt_num(value: float) -> str:
    if not np.isfinite(value):
        return ""
    av = abs(float(value))
    if 0.0 < av < 0.001:
        return r"$<0.001$"
    return f"{float(value):.3f}"


def fmt_sci(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.1e}"


def fmt_bound(value: float) -> str:
    if np.isfinite(value) and float(value) >= 1.0 - 5.0e-8:
        return "1.000 (vacuous)"
    return fmt_num(value)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def centered_covariance(scores: np.ndarray, demean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(scores, dtype=float)
    E = q - q.mean(axis=0, keepdims=True) if demean else q
    C = rc.sym((E.T @ E) / max(len(E), 1))
    rc.symmetric_eigendecomposition_psd(C, name="C_hat")
    return C, E


def positive_eigen_part(C: np.ndarray) -> tuple[rc.SymmetricEigendecomposition, np.ndarray, np.ndarray, np.ndarray, float]:
    eig = rc.symmetric_eigendecomposition_psd(C, name="C_hat")
    tol = max(1.0e-12, float(eig.values[0]) * 1.0e-12 if eig.values.size else 1.0e-12)
    pos = eig.values > tol
    Vp = eig.vectors[:, pos]
    lamp = eig.values[pos]
    return eig, Vp, lamp, np.sqrt(lamp), tol


def _invsqrt_spd(A: np.ndarray) -> np.ndarray:
    eig = rc.symmetric_eigendecomposition_psd(A, name="normalization covariance")
    vals = np.maximum(eig.values, 1.0e-14)
    return rc.sym((eig.vectors * (1.0 / np.sqrt(vals))[None, :]) @ eig.vectors.T)


def load_macro_full_scores() -> tuple[np.ndarray, pd.Series, dict[str, Any], list[str]]:
    import run_publication_grade_ovk as pub  # noqa: WPS433

    data_zip = ROOT / "data_raw" / "data.zip"
    if not data_zip.exists():
        raise FileNotFoundError(f"Macro source zip not found: {data_zip}")
    panels = pub.load_panels_from_zip(data_zip)
    panel = pub.add_placebo_shocks(panels["panel"], seed=pub.PLACEBO_SEED, shift_months=pub.PLACEBO_SHIFT_MONTHS)
    panel, sf_meta = pub.add_sf_fed_shocks(panel, pub.SF_FED_SURPRISES)
    scores = pub.build_lp_scores(
        panel,
        "MP_median_fallback",
        "CBI_median_fallback",
        H=pub.H,
        L=pub.L,
        outcome_columns=tuple(pub.BASE_OUTCOME_COLUMNS),
    )
    chi = np.asarray(scores["Q_scores"], dtype=float)
    dates = pd.Series(pd.to_datetime(scores["dates"]))
    if chi.shape[1] != 125:
        raise RuntimeError(f"Expected macro full pre-PCA influence surface dimension 125, found {chi.shape[1]}.")
    meta = {
        "shock_col": scores["shock_col"],
        "cbi_col": scores["cbi_col"],
        "H": int(pub.H),
        "L": int(pub.L),
        "outcome_labels": list(scores["outcome_labels"]),
        "outcome_columns": list(scores["outcome_columns"]),
        "sf_fed_metadata": sf_meta,
    }
    sources = [data_zip, ROOT / "code" / "run_publication_grade_ovk.py"]
    sf_path = Path(str(sf_meta.get("sf_fed_path", ""))) if sf_meta.get("sf_fed_path") else None
    if sf_path is not None and sf_path.exists():
        sources.append(sf_path)
    return chi, dates, meta, [rel(p) for p in sources]


def load_macro_rho() -> tuple[float, str]:
    rank_summary_path = ROOT / "results" / "publication_grade_ovk" / "outputs" / "tables" / "publication_grade_rank_summary.csv"
    if rank_summary_path.exists():
        rank_summary = pd.read_csv(rank_summary_path)
        row = rank_summary[(rank_summary["variant"].eq("base5_headline")) & (rank_summary["rank"].astype(int).eq(5))]
        if len(row) and "alpha_hat" in row:
            return float(row["alpha_hat"].iloc[0]), rel(rank_summary_path)
    return 0.03, "fallback alpha=0.03; publication rank summary not found"


def path_graph_laplacian(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    dt = pd.to_datetime(dates)
    month_index = dt.dt.year.to_numpy(int) * 12 + dt.dt.month.to_numpy(int)
    gaps = np.diff(month_index)
    if np.any(gaps <= 0):
        raise RuntimeError("Macro dates must be strictly increasing monthly states.")
    T = len(dt)
    L = np.zeros((T, T), dtype=float)
    edge_weights = 1.0 / np.maximum(gaps.astype(float), 1.0) ** 2
    for i, w in enumerate(edge_weights):
        L[i, i] += w
        L[i + 1, i + 1] += w
        L[i, i + 1] -= w
        L[i + 1, i] -= w
    return rc.sym(L), edge_weights


def temporal_resolvent_weights(L: np.ndarray, eta: float) -> np.ndarray:
    T = L.shape[0]
    W = float(eta) * np.linalg.solve(float(eta) * np.eye(T) + rc.sym(L), np.eye(T))
    return rc.sym(W)


def weight_diagnostics(W: np.ndarray, C: np.ndarray, E: np.ndarray) -> dict[str, float]:
    T = W.shape[0]
    mean_K = rc.sym((E.T * (W.sum(axis=0) / max(T, 1))[None, :]) @ E)
    eff = 1.0 / np.maximum(np.sum(np.maximum(W, 0.0) ** 2, axis=1), 1.0e-300)
    return {
        "min_weight": float(W.min()),
        "max_weight": float(W.max()),
        "max_row_sum_error": float(np.max(np.abs(W.sum(axis=1) - 1.0))),
        "max_column_sum_error": float(np.max(np.abs(W.sum(axis=0) - 1.0))),
        "symmetry_error": float(np.max(np.abs(W - W.T))),
        "mean_C_relative_error": float(np.linalg.norm(mean_K - C, ord="fro") / max(np.linalg.norm(C, ord="fro"), 1.0e-12)),
        "effective_n_min": float(eff.min()),
        "effective_n_median": float(np.median(eff)),
        "effective_n_max": float(eff.max()),
        "diagonal_weight_median": float(np.median(np.diag(W))),
    }


def validation_blocks(n: int, block_len: int) -> list[np.ndarray]:
    block_len = int(max(1, block_len))
    return [np.arange(start, min(start + block_len, n), dtype=int) for start in range(0, n, block_len)]


def eta_grid_from_laplacian(L: np.ndarray, *, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    vals = np.linalg.eigvalsh(rc.sym(L))
    pos = vals[vals > max(1.0e-12, float(vals.max()) * 1.0e-12 if vals.size else 1.0e-12)]
    if pos.size:
        lo = float(pos.min()) * 1.0e-4 if lower is None else float(lower)
        hi = float(pos.max()) * 1.0e2 if upper is None else float(upper)
    else:
        lo = 1.0e-6 if lower is None else float(lower)
        hi = 1.0e2 if upper is None else float(upper)
    lo = max(lo, float(CONFIG["macro_eta_lower_limit"]))
    hi = min(hi, float(CONFIG["macro_eta_upper_limit"]))
    if lo >= hi:
        hi = min(max(lo * 10.0, lo + 1.0e-12), float(CONFIG["macro_eta_upper_limit"]))
    return np.unique(np.logspace(np.log10(lo), np.log10(hi), int(CONFIG["macro_eta_grid_size"])))


def blocked_predictive_gaussian_score(E: np.ndarray, W: np.ndarray, rho: float, blocks: list[np.ndarray]) -> dict[str, float]:
    p = E.shape[1]
    const = p * math.log(2.0 * math.pi)
    losses = []
    eff = []
    min_train_weight = []
    for block in blocks:
        block_mask = np.zeros(E.shape[0], dtype=bool)
        block_mask[block] = True
        for t in block:
            w = np.maximum(W[t].copy(), 0.0)
            w[block_mask] = 0.0
            sw = float(w.sum())
            if sw <= 1.0e-14:
                return {
                    "blocked_predictive_neg_loglik": math.inf,
                    "blocked_predictive_effective_n_median": 0.0,
                    "blocked_predictive_min_training_weight_sum": sw,
                }
            w /= sw
            K = rc.sym((E.T * w[None, :]) @ E)
            S = rc.sym(K + float(rho) * np.eye(p))
            try:
                c, lower = cho_factor(S, lower=True, check_finite=False)
                alpha = cho_solve((c, lower), E[t], check_finite=False)
                logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
            except Exception:
                sign, logdet_np = np.linalg.slogdet(S)
                if sign <= 0:
                    return {
                        "blocked_predictive_neg_loglik": math.inf,
                        "blocked_predictive_effective_n_median": 0.0,
                        "blocked_predictive_min_training_weight_sum": sw,
                    }
                alpha = np.linalg.solve(S, E[t])
                logdet = float(logdet_np)
            losses.append(0.5 * (const + logdet + float(E[t] @ alpha)))
            eff.append(float(1.0 / np.sum(w * w)))
            min_train_weight.append(sw)
    return {
        "blocked_predictive_neg_loglik": float(np.mean(losses)),
        "blocked_predictive_effective_n_median": float(np.median(eff)),
        "blocked_predictive_min_training_weight_sum": float(np.min(min_train_weight)),
    }


def evaluate_macro_eta_grid(C: np.ndarray, E: np.ndarray, L: np.ndarray, rho: float, etas: np.ndarray) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    blocks = validation_blocks(E.shape[0], int(CONFIG["macro_validation_block_months"]))
    rows = []
    candidates: dict[float, np.ndarray] = {}
    for eta in np.asarray(etas, dtype=float):
        W = temporal_resolvent_weights(L, float(eta))
        score = blocked_predictive_gaussian_score(E, W, float(rho), blocks)
        diag = weight_diagnostics(W, C, E)
        valid = (
            diag["min_weight"] >= -1.0e-10
            and diag["max_row_sum_error"] <= 1.0e-8
            and diag["max_column_sum_error"] <= 1.0e-8
            and diag["symmetry_error"] <= 1.0e-8
            and diag["mean_C_relative_error"] <= 1.0e-8
            and np.isfinite(score["blocked_predictive_neg_loglik"])
        )
        rows.append({"eta": float(eta), "candidate_valid": bool(valid), **score, **diag})
        if valid:
            candidates[float(eta)] = W
    return pd.DataFrame(rows), candidates


def select_macro_eta(C: np.ndarray, E: np.ndarray, dates: pd.Series, rho: float) -> tuple[float, np.ndarray, pd.DataFrame, dict[str, Any]]:
    L, edge_weights = path_graph_laplacian(dates)
    grid = eta_grid_from_laplacian(L)
    all_rows = []
    all_candidates: dict[float, np.ndarray] = {}
    extension_history = []
    selected_row: pd.Series | None = None
    for extension in range(int(CONFIG["macro_eta_max_extensions"]) + 1):
        frame, candidates = evaluate_macro_eta_grid(C, E, L, float(rho), grid)
        frame["grid_extension_round"] = extension
        all_rows.append(frame)
        all_candidates.update(candidates)
        combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["eta"], keep="last").sort_values("eta").reset_index(drop=True)
        valid_grid = combined[combined["candidate_valid"]].copy()
        if valid_grid.empty:
            raise RuntimeError("No valid temporal-resolvent eta satisfied stochastic/symmetry/average-preserving checks.")
        best = valid_grid.sort_values(["blocked_predictive_neg_loglik", "eta"]).iloc[0]
        valid_etas = valid_grid["eta"].to_numpy(float)
        at_lower = bool(np.isclose(float(best["eta"]), float(valid_etas.min())))
        at_upper = bool(np.isclose(float(best["eta"]), float(valid_etas.max())))
        extension_history.append(
            {
                "round": extension,
                "eta_min": float(valid_etas.min()),
                "eta_max": float(valid_etas.max()),
                "best_eta": float(best["eta"]),
                "best_score": float(best["blocked_predictive_neg_loglik"]),
                "boundary": "lower" if at_lower else ("upper" if at_upper else "interior"),
            }
        )
        if not (at_lower or at_upper):
            selected_row = best
            break
        if extension >= int(CONFIG["macro_eta_max_extensions"]):
            selected_row = best
            break
        factor = float(CONFIG["macro_eta_extension_factor"])
        if at_lower:
            new_lower = max(float(valid_etas.min()) / factor, float(CONFIG["macro_eta_lower_limit"]))
            new_upper = float(valid_etas.max())
            if new_lower >= float(valid_etas.min()) or new_lower <= float(CONFIG["macro_eta_lower_limit"]) and float(valid_etas.min()) <= float(CONFIG["macro_eta_lower_limit"]):
                selected_row = best
                break
        else:
            new_lower = float(valid_etas.min())
            new_upper = min(float(valid_etas.max()) * factor, float(CONFIG["macro_eta_upper_limit"]))
            if new_upper <= float(valid_etas.max()) or new_upper >= float(CONFIG["macro_eta_upper_limit"]) and float(valid_etas.max()) >= float(CONFIG["macro_eta_upper_limit"]):
                selected_row = best
                break
        grid = eta_grid_from_laplacian(L, lower=new_lower, upper=new_upper)
        existing = set(float(x) for x in combined["eta"].to_numpy(float))
        grid = np.asarray([float(x) for x in grid if float(x) not in existing], dtype=float)
        if grid.size == 0:
            selected_row = best
            break
    if selected_row is None:
        raise RuntimeError("Macro eta selection failed unexpectedly.")
    final_grid = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["eta"], keep="last").sort_values("eta").reset_index(drop=True)
    final_grid.to_csv(TABLES / "macro_temporal_resolvent_eta_grid.csv", index=False)
    eta = float(selected_row["eta"])
    W = all_candidates[eta]
    valid_grid = final_grid[final_grid["candidate_valid"]].copy()
    eta_min = float(valid_grid["eta"].min())
    eta_max = float(valid_grid["eta"].max())
    boundary = "lower" if np.isclose(eta, eta_min) else ("upper" if np.isclose(eta, eta_max) else "interior")
    top_candidates = valid_grid.sort_values(["blocked_predictive_neg_loglik", "eta"]).head(8)
    diagnostics = {
        "selected_eta": eta,
        "selected_criterion": float(selected_row["blocked_predictive_neg_loglik"]),
        "validation_criterion": "blocked predictive Gaussian quasi-negative-log-likelihood on held-out full 125-dimensional influence vectors",
        "selection_rule": "Minimum blocked predictive Gaussian quasi-negative-log-likelihood over admissible temporal-resolvent weights.",
        "validation_block_months": int(CONFIG["macro_validation_block_months"]),
        "validation_ridge": float(rho),
        "eta_boundary_solution": bool(boundary != "interior"),
        "eta_boundary": boundary,
        "eta_boundary_stop_reason": (
            "interior optimum"
            if boundary == "interior"
            else f"{boundary}-boundary solution; grid extension found no more {boundary} eta satisfying the stochastic, nonnegative-weight, and average-preserving validation checks before stopping."
        ),
        "eta_grid_min": eta_min,
        "eta_grid_max": eta_max,
        "eta_evaluated_min": float(final_grid["eta"].min()),
        "eta_evaluated_max": float(final_grid["eta"].max()),
        "eta_grid_extension_history": extension_history,
        "eta_top_predictive_candidates": top_candidates[
            ["eta", "blocked_predictive_neg_loglik", "blocked_predictive_effective_n_median", "effective_n_median", "mean_C_relative_error"]
        ].to_dict(orient="records"),
        "eta_not_selected_using": ETA_NOT_SELECTED_USING,
        "path_graph_edge_weight_min": float(edge_weights.min()) if edge_weights.size else 0.0,
        "path_graph_edge_weight_max": float(edge_weights.max()) if edge_weights.size else 0.0,
        "selected_weight_checks": {
            k: float(selected_row[k])
            for k in [
                "blocked_predictive_neg_loglik",
                "blocked_predictive_effective_n_median",
                "blocked_predictive_min_training_weight_sum",
                "min_weight",
                "max_weight",
                "max_row_sum_error",
                "max_column_sum_error",
                "symmetry_error",
                "mean_C_relative_error",
                "effective_n_min",
                "effective_n_median",
                "effective_n_max",
                "diagonal_weight_median",
            ]
        },
        "eta_grid_csv": rel(TABLES / "macro_temporal_resolvent_eta_grid.csv"),
    }
    return eta, W, final_grid, diagnostics


def smoothed_covariance_states(E: np.ndarray, W: np.ndarray) -> np.ndarray:
    K_states = np.empty((W.shape[0], E.shape[1], E.shape[1]), dtype=float)
    for i in range(W.shape[0]):
        w = np.maximum(W[i], 0.0)
        w /= max(float(w.sum()), 1.0e-300)
        X = E * np.sqrt(w)[:, None]
        K_states[i] = rc.sym(X.T @ X)
    return K_states


def numerical_ranks(K_states: np.ndarray) -> np.ndarray:
    ranks = []
    for K in K_states:
        vals = np.linalg.eigvalsh(rc.sym(K))
        tol = max(1.0e-10, float(vals.max()) * 1.0e-10 if vals.size else 1.0e-10)
        ranks.append(int(np.sum(vals > tol)))
    return np.asarray(ranks, dtype=int)


def macro_eta_sensitivity_from_weights(C: np.ndarray, E: np.ndarray, rho: float, selected_eta: float, weights_by_eta: dict[float, np.ndarray]) -> pd.DataFrame:
    etas = np.asarray(sorted(weights_by_eta), dtype=float)
    roles: list[tuple[str, float]] = [("selected", float(selected_eta))]
    lower = etas[etas < selected_eta / 1.5]
    upper = etas[etas > selected_eta * 1.5]
    if lower.size:
        roles.append(("more_smoothed", float(lower[-1])))
    if upper.size:
        roles.append(("less_smoothed", float(upper[0])))
    eig = rc.symmetric_eigendecomposition_psd(C, name="macro C_hat")
    scale = 1.0 / np.sqrt(eig.values + float(rho))
    Q = rc.build_probe_soft(C, float(rho))
    Q_v = rc.sym(eig.vectors.T @ Q @ eig.vectors)
    rows = []
    for role, eta in roles:
        W = weights_by_eta[eta]
        K_states = smoothed_covariance_states(E, W)
        exact_errors = []
        d_counts: dict[int, list[int]] = {int(thr): [] for thr in CONFIG["severe_thresholds"]}
        captures: dict[int, list[float]] = {int(thr): [] for thr in CONFIG["severe_thresholds"]}
        for K in K_states:
            K_v = rc.sym(eig.vectors.T @ K @ eig.vectors)
            A = rc.sym((scale[:, None] * K_v) * scale[None, :])
            vals, vecs = np.linalg.eigh(A)
            q_full = stress_from_eig(vals, vecs, Q_v, float(CONFIG["lambda_main"]))
            theta5 = rc.sym(A[:5, :5])
            q5 = rc.stress_trace(theta5, Q_v[:5, :5], float(CONFIG["lambda_main"]))
            exact_errors.append(abs(q_full - q5))
            for thr in CONFIG["severe_thresholds"]:
                key = int(thr)
                mask = vals >= float(thr) - rc.ASSERT_TOL
                d = int(np.sum(mask))
                d_counts[key].append(d)
                if d:
                    U = vecs[:, mask]
                    captures[key].append(float(np.linalg.norm(U[:5, :], ord="fro") ** 2 / d))
        row: dict[str, Any] = {
            "role": role,
            "eta": eta,
            "effective_n_median": float(np.median(1.0 / np.maximum(np.sum(np.maximum(W, 0.0) ** 2, axis=1), 1.0e-300))),
            "max_rank5_exact_error_lambda1": float(np.max(exact_errors)),
        }
        for thr in CONFIG["severe_thresholds"]:
            key = int(thr)
            arr = np.asarray(d_counts[key], dtype=int)
            cap = np.asarray(captures[key], dtype=float)
            row[f"median_severe_direction_count_{key}"] = float(np.median(arr))
            row[f"max_severe_direction_count_{key}"] = int(np.max(arr))
            row[f"median_severe_subspace_capture_{key}"] = float(np.median(cap)) if cap.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def macro_surface() -> SurfaceData:
    scores, dates, score_meta, source_files = load_macro_full_scores()
    rho, rho_source = load_macro_rho()
    C, E = centered_covariance(scores, demean=True)
    eta, W, grid, eta_meta = select_macro_eta(C, E, dates, float(rho))
    K_states = smoothed_covariance_states(E, W)
    ranks = numerical_ranks(K_states)
    diag = weight_diagnostics(W, C, E)
    state_diag = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "date_month": dates.dt.strftime("%Y-%m"),
            "effective_local_sample_size": 1.0 / np.maximum(np.sum(np.maximum(W, 0.0) ** 2, axis=1), 1.0e-300),
            "self_weight": np.diag(W),
            "numerical_rank": ranks,
            "trace_K": np.trace(K_states, axis1=1, axis2=2),
        }
    )
    state_diag.to_csv(TABLES / "macro_temporal_resolvent_state_diagnostics.csv", index=False)
    if float(np.median(ranks)) <= 1.0:
        raise RuntimeError("Macro temporal-resolvent benchmark failed: median numerical rank is not greater than one.")
    if diag["mean_C_relative_error"] > 1.0e-8:
        raise RuntimeError("Macro temporal-resolvent benchmark failed average-preserving covariance check.")
    L, _edge_weights = path_graph_laplacian(dates)
    sensitivity_etas = sorted(set(float(x) for x in grid.loc[grid["candidate_valid"], "eta"].to_numpy(float)))
    sensitivity_weights = {float(x): temporal_resolvent_weights(L, float(x)) for x in sensitivity_etas}
    sensitivity = macro_eta_sensitivity_from_weights(C, E, float(rho), float(eta), sensitivity_weights)
    sensitivity_path = TABLES / "macro_eta_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    state_frame = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "date_month": dates.dt.strftime("%Y-%m")})
    return SurfaceData(
        key="macro",
        application="Macro / monetary policy",
        surface="125-coordinate full-space temporal-resolvent benchmark",
        principal_probe="whole-surface soft probe",
        C=C,
        K_states=K_states,
        state_frame=state_frame,
        rho=float(rho),
        Q=rc.build_probe_soft(C, float(rho)),
        source_files=source_files + [rho_source],
        state_mask_note="All usable base-five headline monthly LP influence rows rebuilt from original pre-PCA contributions.",
        extra={
            **score_meta,
            **eta_meta,
            "rho_source": rho_source,
            "full_K_source": "Full-space temporal-resolvent benchmark K_hat_eta(t)=sum_j W_eta[t,j] chi_j chi_j' rebuilt from the original 125-dimensional pre-PCA LP influence contributions chi_t. This is not a full-dimensional latent state-space estimate.",
            "disallowed_sources_not_used": [
                "cached rank-five Q_scores",
                "zero-padded or lifted rank-five object",
                "single monthly outer product without smoothing",
            ],
            "weight_diagnostics": diag,
            "median_numerical_rank": float(np.median(ranks)),
            "min_numerical_rank": int(ranks.min()),
            "state_diagnostics_csv": rel(TABLES / "macro_temporal_resolvent_state_diagnostics.csv"),
            "eta_sensitivity_csv": rel(sensitivity_path),
        },
    )


def full_state_covariances_from_kernel_weights(
    C: np.ndarray,
    E: np.ndarray,
    state_values: np.ndarray,
    grid_values: np.ndarray,
    bandwidth_values: np.ndarray,
    alpha: float,
    support_weights_for_normalization: bool = True,
) -> np.ndarray:
    _eig, Vp, lamp, lam_half, _tol = positive_eigen_part(C)
    Z = (E @ Vp) / np.sqrt(lamp)[None, :]
    raw = []
    masses = []
    I = np.eye(Z.shape[1])
    x = np.asarray(state_values, dtype=float)
    for s, bw in zip(grid_values, bandwidth_values):
        weights = np.exp(-0.5 * ((x - float(s)) / max(float(bw), 1.0e-12)) ** 2)
        sw = float(weights.sum())
        if sw <= 0:
            M = I
        else:
            M = (Z.T * weights[None, :]) @ Z / sw
        raw.append(rc.sym(float(alpha) * I + (1.0 - float(alpha)) * M))
        masses.append(sw)
    raw_arr = np.stack(raw, axis=0)
    if support_weights_for_normalization:
        w = np.asarray(masses, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        w = w / max(float(w.sum()), 1.0e-12)
        mean_A = rc.sym(np.einsum("s,sij->ij", w, raw_arr, optimize=True))
    else:
        mean_A = rc.sym(raw_arr.mean(axis=0))
    mean_inv = _invsqrt_spd(mean_A)
    A_norm = rc.sym(np.einsum("ij,sjk,kl->sil", mean_inv, raw_arr, mean_inv, optimize=True))
    K_states = np.zeros((len(A_norm), C.shape[0], C.shape[0]), dtype=float)
    for i, A in enumerate(A_norm):
        K_pos = (lam_half[:, None] * A) * lam_half[None, :]
        K_states[i] = rc.sym(Vp @ K_pos @ Vp.T)
    return K_states


def lalonde_surfaces() -> tuple[SurfaceData, dict[str, np.ndarray]]:
    root = ROOT / "results" / "lalonde_cross_ovk"
    tables = root / "tables"
    core_path = tables / "ovk_core_arrays.npz"
    path_1d = tables / "ovk_pscore_1d_path.csv"
    state_path = tables / "state_scores.csv"
    theta_path = tables / "theta_estimates.csv"
    metadata_path = root / "lalonde_run_metadata.json"
    with np.load(core_path, allow_pickle=False) as z:
        C = rc.sym(np.asarray(z["K_bar"], dtype=float))
        E = np.asarray(z["E"], dtype=float)
    one_d = pd.read_csv(path_1d)
    state = pd.read_csv(state_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    alpha = float(metadata["smoothing"]["alpha"])
    K_states = full_state_covariances_from_kernel_weights(
        C,
        E,
        state["logit_pscore_std"].to_numpy(float),
        one_d["logit_pscore_std"].to_numpy(float),
        one_d["bandwidth"].to_numpy(float),
        alpha,
        support_weights_for_normalization=True,
    )
    theta = pd.read_csv(theta_path)
    components = theta["component"].astype(str).tolist()
    comp_index = {name: i for i, name in enumerate(components)}
    required = ["ATE_all", "ATE_re74_zero", "ATE_re75_zero", "ATE_nodegree", "ATE_degree", "ATE_low_educ_le_10", "ATE_high_educ_gt_10"]
    missing = [x for x in required if x not in comp_index]
    if missing:
        raise RuntimeError(f"LaLonde probe labels missing from theta_estimates.csv: {missing}")
    g = np.zeros(len(components))
    g[comp_index["ATE_all"]] = 1.0
    probes = {
        "overall ATE": rc.build_probe_direction(C, g),
        "prior earnings block": rc.build_probe_block(C, [comp_index["ATE_re74_zero"], comp_index["ATE_re75_zero"]]),
        "education block": rc.build_probe_block(
            C,
            [comp_index["ATE_nodegree"], comp_index["ATE_degree"], comp_index["ATE_low_educ_le_10"], comp_index["ATE_high_educ_gt_10"]],
        ),
    }
    surface = SurfaceData(
        key="lalonde",
        application="LaLonde ATE",
        surface="11-coordinate subgroup ATE surface",
        principal_probe="overall ATE direction",
        C=C,
        K_states=K_states,
        state_frame=one_d.copy(),
        rho=alpha,
        Q=probes["overall ATE"],
        source_files=[rel(p) for p in [core_path, path_1d, state_path, theta_path, metadata_path]],
        state_mask_note="All points on the published one-dimensional logit-propensity support grid.",
        extra={
            "components": components,
            "full_K_source": "Full 11-coordinate local covariance rebuilt from demeaned influence matrix E and the published 1D logit-propensity kernel weights.",
        },
    )
    return surface, probes


def load_hurricane_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    app = ROOT / "applications" / "deryugina_hurricanes"
    scripts = app / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from deryugina_common import load_config  # type: ignore  # noqa: WPS433
    from deryugina_pipeline import add_spatial_fields  # type: ignore  # noqa: WPS433

    inter = app / "outputs" / "intermediate"
    psi_path = inter / "Psi_corrected.parquet"
    obs_path = inter / "influence_observations_corrected.csv"
    surface_path = inter / "surface_metadata_corrected.csv"
    psi = pd.read_parquet(psi_path).to_numpy(float)
    obs = pd.read_csv(obs_path)
    if "region" not in obs.columns:
        obs = add_spatial_fields(obs)
    surface = pd.read_csv(surface_path)
    return psi, obs, surface, load_config(), [rel(p) for p in [psi_path, obs_path, surface_path]]


def region_year_full_covariances(psi: np.ndarray, obs: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    C = rc.sym((psi.T @ psi) / max(len(psi), 1))
    _eig, Vp, lamp, lam_half, _tol = positive_eigen_part(C)
    Z = (psi @ Vp) / np.sqrt(lamp)[None, :]
    regions = sorted(obs["region"].fillna("Unknown").astype(str).unique())
    years = np.arange(int(obs["year"].min()), int(obs["year"].max()) + 1)
    year_arr = pd.to_numeric(obs["year"], errors="coerce").to_numpy(int)
    region_arr = obs["region"].fillna("Unknown").astype(str).to_numpy()

    stats: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
    for region in regions:
        region_mask = region_arr == region
        for year in years:
            mask = region_mask & (year_arr == int(year))
            count = int(mask.sum())
            stats[(region, int(year))] = (count, Z[mask].T @ Z[mask] if count else np.zeros((Z.shape[1], Z.shape[1])))

    bw = float(cfg["smoother"]["time_bandwidth_years"])
    base_rho = float(cfg["smoother"]["ridge_rho"])
    min_eff = float(cfg["smoother"]["min_effective_obs"])
    I = np.eye(Z.shape[1])
    raw = []
    rows = []
    for region in regions:
        counts = np.asarray([stats[(region, int(year))][0] for year in years], dtype=float)
        cross = np.stack([stats[(region, int(year))][1] for year in years], axis=0)
        for y0 in years:
            wy = np.exp(-0.5 * ((years.astype(float) - float(y0)) / max(bw, 1.0e-12)) ** 2)
            sw = float(np.sum(wy * counts))
            denom_eff = float(np.sum(counts * wy * wy))
            eff = float(sw * sw / max(denom_eff, 1.0e-12)) if sw > 0 else 0.0
            if sw <= 0:
                A_raw = I.copy()
                rho_eff = 1.0
                status = "identity_fallback"
            else:
                M = np.einsum("y,yij->ij", wy, cross, optimize=True) / sw
                extra = max(0.0, min_eff - eff) / max(min_eff, 1.0e-12)
                rho_eff = min(0.70, base_rho + 0.35 * extra)
                A_raw = rc.sym((1.0 - rho_eff) * M + rho_eff * I)
                status = "sparse_blended" if eff < min_eff else "smoothed"
            raw.append(A_raw)
            rows.append(
                {
                    "region": region,
                    "year": int(y0),
                    "weight_sum": sw,
                    "effective_obs": eff,
                    "ridge_rho_effective": rho_eff,
                    "smoother_status": status,
                    "leave_out": "none",
                }
            )
    raw_arr = np.stack(raw, axis=0)
    mean_inv = _invsqrt_spd(raw_arr.mean(axis=0))
    A_norm = rc.sym(np.einsum("ij,sjk,kl->sil", mean_inv, raw_arr, mean_inv, optimize=True))
    K_states = np.zeros((len(A_norm), C.shape[0], C.shape[0]), dtype=float)
    for i, A in enumerate(A_norm):
        K_pos = (lam_half[:, None] * A) * lam_half[None, :]
        K_states[i] = rc.sym(Vp @ K_pos @ Vp.T)
    return C, pd.DataFrame(rows), K_states


def hurricane_surfaces() -> tuple[SurfaceData, SurfaceData]:
    psi, obs, surface, cfg, source_files = load_hurricane_inputs()
    base_rho = float(cfg["smoother"]["ridge_rho"])
    fiscal_mask = surface["block"].astype(str).eq("fiscal")
    fiscal_post_mask = fiscal_mask & surface["bin_block"].astype(str).isin(["early_post", "late_post"])
    if int(surface.shape[0]) != 72:
        raise RuntimeError(f"Expected 72-coordinate full hurricane surface, found {int(surface.shape[0])}.")
    if not fiscal_post_mask.any():
        raise RuntimeError("Hurricane fiscal/post selector is empty.")
    fiscal_post_count = int(fiscal_post_mask.sum())
    audit = surface.copy()
    audit["included_in_fiscal"] = fiscal_mask.to_numpy(bool)
    audit["included_in_fiscal_post"] = fiscal_post_mask.to_numpy(bool)
    audit["included_in_full_surface"] = True
    audit_path = TABLES / "hurricane_selector_audit.csv"
    audit.to_csv(audit_path, index=False)

    C_full, cells_full, K_full = region_year_full_covariances(psi, obs, cfg)
    full = SurfaceData(
        key="hurricane_full",
        application="Hurricane full fiscal/post probe",
        surface="72-coordinate full event-study surface",
        principal_probe="fiscal/post block probe",
        C=C_full,
        K_states=K_full,
        state_frame=cells_full,
        rho=base_rho,
        Q=rc.build_probe_block(C_full, fiscal_post_mask.to_numpy(bool)),
        source_files=source_files + [rel(ROOT / "applications" / "deryugina_hurricanes" / "config" / "deryugina_config.yaml")],
        state_mask_note="All published region-year cells from the corrected hurricane space-time smoother.",
        extra={
            "full_K_source": "Full 72-coordinate region-year covariance rebuilt from Psi_corrected with the published region-year kernel weights and ridge_rho_effective rule.",
            "selector_audit_csv": rel(audit_path),
            "selected_coordinates": fiscal_post_count,
            "full_surface_dimension": int(surface.shape[0]),
        },
    )

    idx = surface.loc[fiscal_post_mask, "surface_index"].astype(int).to_numpy()
    C_restricted, cells_restricted, K_restricted = region_year_full_covariances(psi[:, idx], obs, cfg)
    restricted = SurfaceData(
        key="hurricane_fiscal_post",
        application="Hurricane fiscal/post restricted surface",
        surface=f"{fiscal_post_count}-coordinate fiscal/post event-study surface",
        principal_probe="whole-surface soft probe",
        C=C_restricted,
        K_states=K_restricted,
        state_frame=cells_restricted,
        rho=base_rho,
        Q=rc.build_probe_soft(C_restricted, base_rho),
        source_files=source_files + [rel(ROOT / "applications" / "deryugina_hurricanes" / "config" / "deryugina_config.yaml")],
        state_mask_note=f"All published region-year cells, restricted to the true fiscal/post intersection ({fiscal_post_count} coordinates).",
        extra={
            "full_K_source": f"Full {fiscal_post_count}-coordinate fiscal/post covariance rebuilt from the matching columns of Psi_corrected with the same region-year weights used for the 72-coordinate surface.",
            "selector_audit_csv": rel(audit_path),
            "selected_coordinates": fiscal_post_count,
            "full_surface_dimension": int(surface.shape[0]),
        },
    )
    return full, restricted


def operator_stack_in_C_basis(surface: SurfaceData) -> tuple[rc.SymmetricEigendecomposition, np.ndarray, np.ndarray, dict[str, float]]:
    eig = rc.symmetric_eigendecomposition_psd(surface.C, name=f"{surface.key} C_hat")
    scale = 1.0 / np.sqrt(eig.values + surface.rho)
    A_stack = np.empty_like(surface.K_states)
    K_v_stack = np.empty_like(surface.K_states)
    min_k = math.inf
    min_a = math.inf
    for i, K in enumerate(surface.K_states):
        k_eig = rc.symmetric_eigendecomposition_psd(K, name=f"{surface.key} K_hat state {i}")
        min_k = min(min_k, float(k_eig.min_raw_eigenvalue))
        K_v = rc.sym(eig.vectors.T @ K @ eig.vectors)
        K_v_stack[i] = K_v
        A_stack[i] = rc.sym((scale[:, None] * K_v) * scale[None, :])
        a_eig = rc.symmetric_eigendecomposition_psd(A_stack[i], name=f"{surface.key} A_rho state {i}")
        min_a = min(min_a, float(a_eig.min_raw_eigenvalue))
    diagnostics = {
        "min_raw_eigenvalue_C": float(eig.min_raw_eigenvalue),
        "min_raw_eigenvalue_K_states": float(min_k),
        "min_raw_eigenvalue_A_states": float(min_a),
    }
    return eig, K_v_stack, A_stack, diagnostics


def stress_from_eig(vals: np.ndarray, vecs: np.ndarray, Q: np.ndarray, lambda_value: float) -> float:
    vals = np.maximum(np.asarray(vals, dtype=float), 0.0)
    f_vals = vals / (float(lambda_value) + vals)
    q_diag = np.einsum("ij,ji->i", vecs.T @ rc.sym(Q), vecs, optimize=True)
    value = float(np.sum(f_vals * q_diag))
    if value < -rc.ASSERT_TOL or value > 1.0 + rc.ASSERT_TOL:
        raise FloatingPointError(f"Stress statistic outside [0,1]: {value}")
    return min(1.0, max(0.0, value))


def full_stress_values(A_stack: np.ndarray, Q_v: np.ndarray, lambda_value: float) -> np.ndarray:
    values = []
    for A in A_stack:
        vals, vecs = np.linalg.eigh(rc.sym(A))
        values.append(stress_from_eig(vals, vecs, Q_v, lambda_value))
    return np.asarray(values, dtype=float)


def rank_frame_from_full_stress(
    A_stack: np.ndarray,
    Q_v: np.ndarray,
    q_full: np.ndarray,
    R: int,
    lambda_value: float,
) -> pd.DataFrame:
    p = A_stack.shape[1]
    R = int(min(max(R, 1), p))
    V_R = np.eye(p)[:, :R]
    rows = []
    for i, A in enumerate(A_stack):
        theta = rc.sym(A[:R, :R])
        q_rank = rc.stress_trace(theta, Q_v[:R, :R], lambda_value)
        cert = rc.ritz_certificate(A, V_R, theta, Q_v, lambda_value)
        error = abs(float(q_full[i]) - q_rank)
        if error > cert.capped_bound + 2.0e-7:
            raise AssertionError(f"Ritz certificate failed at state {i}, R={R}: error {error}, bound {cert.capped_bound}.")
        utilization = error / cert.uncapped_bound if cert.uncapped_bound > 1.0e-14 else 0.0
        rows.append(
            {
                "state_index": i,
                "rank": R,
                "q_full": float(q_full[i]),
                "q_rank": q_rank,
                "exact_error": error,
                "retained_probe_mass": cert.retained_mass,
                "omitted_probe_mass": cert.omitted_mass,
                "ritz_subspace_residual_norm": cert.residual_norm,
                "generalized_eigenpair_residual_norm": np.nan,
                "residual_term": cert.residual_term,
                "uncapped_ritz_bound": cert.uncapped_bound,
                "capped_ritz_bound": cert.capped_bound,
                "bound_utilization": float(utilization),
            }
        )
    return pd.DataFrame(rows)


def rank_metrics_for_A_stack(A_stack: np.ndarray, Q_v: np.ndarray, ranks: list[int], lambda_value: float) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    p = A_stack.shape[1]
    q_full = full_stress_values(A_stack, Q_v, lambda_value)
    frames: dict[int, pd.DataFrame] = {}
    rows = []
    for R0 in sorted(set([int(min(max(r, 1), p)) for r in ranks] + [p])):
        frame = rank_frame_from_full_stress(A_stack, Q_v, q_full, R0, lambda_value)
        frames[R0] = frame
        rows.append(
            {
                "rank": R0,
                "retained_probe_mass": float(frame["retained_probe_mass"].iloc[0]),
                "omitted_probe_mass": float(frame["omitted_probe_mass"].iloc[0]),
                "p95_exact_error": float(frame["exact_error"].quantile(0.95)),
                "max_exact_error": float(frame["exact_error"].max()),
                "p95_residual_term": float(frame["residual_term"].quantile(0.95)),
                "max_uncapped_ritz_bound": float(frame["uncapped_ritz_bound"].max()),
                "max_capped_ritz_bound": float(frame["capped_ritz_bound"].max()),
                "min_bound_utilization": float(frame["bound_utilization"].min()),
                "max_q_full": float(frame["q_full"].max()),
                "max_q_rank": float(frame["q_rank"].max()),
            }
        )
    full = frames[p]
    for col in ["exact_error", "omitted_probe_mass", "ritz_subspace_residual_norm", "residual_term", "capped_ritz_bound"]:
        if float(full[col].abs().max()) > 1.0e-6:
            raise AssertionError(f"Full-rank {col} is not numerically zero.")
    return pd.DataFrame(rows), frames


def max_ritz_identity_error(K_v_stack: np.ndarray, A_stack: np.ndarray, rho: float, eigenvalues: np.ndarray, ranks: list[int]) -> float:
    scale = 1.0 / np.sqrt(np.asarray(eigenvalues, dtype=float) + float(rho))
    max_err = 0.0
    p = A_stack.shape[1]
    for R0 in sorted(set(int(min(max(r, 1), p)) for r in ranks)):
        theta_from_K = (scale[:R0, None] * K_v_stack[:, :R0, :R0]) * scale[None, None, :R0]
        theta = A_stack[:, :R0, :R0]
        max_err = max(max_err, float(np.max(np.abs(theta - theta_from_K))) if theta.size else 0.0)
    return max_err


def evaluate_surface(surface: SurfaceData, ranks: list[int]) -> SurfaceEvaluation:
    eig, K_v_stack, A_stack, psd_diag = operator_stack_in_C_basis(surface)
    Q_v = rc.sym(eig.vectors.T @ surface.Q @ eig.vectors)
    summary, frames = rank_metrics_for_A_stack(A_stack, Q_v, ranks, float(CONFIG["lambda_main"]))
    max_violation = max(float((frame["exact_error"] - frame["capped_ritz_bound"]).max()) for frame in frames.values())
    identity_error = max_ritz_identity_error(K_v_stack, A_stack, surface.rho, eig.values, list(frames.keys()))
    validation = {
        **psd_diag,
        "max_ritz_compression_identity_error": identity_error,
        "max_certificate_violation": max(0.0, max_violation),
        "certificate_valid": bool(max_violation <= 2.0e-7),
        "ritz_identity_valid": bool(identity_error <= rc.RITZ_IDENTITY_TOL),
        "psd_valid": bool(
            psd_diag["min_raw_eigenvalue_C"] >= -rc.PSD_ATOL
            and psd_diag["min_raw_eigenvalue_K_states"] >= -rc.PSD_ATOL
            and psd_diag["min_raw_eigenvalue_A_states"] >= -rc.PSD_ATOL
        ),
    }
    return SurfaceEvaluation(eig=eig, K_v_stack=K_v_stack, A_stack=A_stack, Q_v=Q_v, rank_summary=summary, frames=frames, validation=validation)


def main_row(surface: SurfaceData, evaluation: SurfaceEvaluation, R: int) -> dict[str, Any]:
    R = int(min(R, surface.C.shape[0]))
    frame = evaluation.frames[R]
    return {
        "application": surface.application,
        "probe": surface.principal_probe,
        "p": int(surface.C.shape[0]),
        "display_R": R,
        "retained_probe_mass": float(frame["retained_probe_mass"].iloc[0]),
        "p95_exact_error": float(frame["exact_error"].quantile(0.95)),
        "max_exact_error": float(frame["exact_error"].max()),
        "max_ritz_bound": float(frame["capped_ritz_bound"].max()),
    }


def surface_metadata(surface: SurfaceData, evaluation: SurfaceEvaluation) -> dict[str, Any]:
    trace = max(float(np.trace(surface.C)), 1.0e-12)
    return {
        "key": surface.key,
        "application": surface.application,
        "surface": surface.surface,
        "principal_probe": surface.principal_probe,
        "source_files": surface.source_files,
        "full_K_source": surface.extra.get("full_K_source", ""),
        "C_hat_source": "Empirical covariance of full influence contributions for this application.",
        "K_hat_construction": surface.extra.get("full_K_source", ""),
        "rho": float(surface.rho),
        "lambda": float(CONFIG["lambda_main"]),
        "rank_grid": sorted(int(x) for x in evaluation.frames.keys()),
        "probe_definition": surface.principal_probe,
        "interval_spectrum_backend": "scipy.linalg.eigh subset_by_value",
        "dimension": int(surface.C.shape[0]),
        "state_count": int(len(surface.state_frame)),
        "state_mask_note": surface.state_mask_note,
        "top5_trace_share": float(evaluation.eig.values[: min(5, len(evaluation.eig.values))].sum() / trace),
        "validation": evaluation.validation,
        "publication_ready": bool(
            evaluation.validation["psd_valid"]
            and evaluation.validation["ritz_identity_valid"]
            and evaluation.validation["certificate_valid"]
        ),
        "extra": surface.extra,
    }


def rank_diagnostics(surface: SurfaceData, evaluation: SurfaceEvaluation) -> list[dict[str, Any]]:
    p = surface.C.shape[0]
    epsilons = [float(x) for x in CONFIG["certification_epsilons"]]
    q_full = full_stress_values(evaluation.A_stack, evaluation.Q_v, float(CONFIG["lambda_main"]))
    found: dict[float, dict[str, Any]] = {eps: {} for eps in epsilons}
    rank_curve_rows = []
    for R in range(1, p + 1):
        frame = rank_frame_from_full_stress(evaluation.A_stack, evaluation.Q_v, q_full, R, float(CONFIG["lambda_main"]))
        max_exact = float(frame["exact_error"].max())
        max_bound = float(frame["capped_ritz_bound"].max())
        rank_curve_rows.append(
            {
                "rank": R,
                "max_exact_error": max_exact,
                "max_capped_ritz_bound": max_bound,
                "max_uncapped_ritz_bound": float(frame["uncapped_ritz_bound"].max()),
                "retained_probe_mass": float(frame["retained_probe_mass"].iloc[0]),
            }
        )
        for eps in epsilons:
            if "exact_rank" not in found[eps] and max_exact <= eps:
                found[eps]["exact_rank"] = R
                found[eps]["exact_rank_max_error"] = max_exact
            if "certified_rank" not in found[eps] and max_bound <= eps:
                found[eps]["certified_rank"] = R
                found[eps]["certified_rank_max_bound"] = max_bound
        if all("exact_rank" in found[eps] and "certified_rank" in found[eps] for eps in epsilons):
            break
    pd.DataFrame(rank_curve_rows).to_csv(TABLES / f"{surface.key}_rank_curve.csv", index=False)
    rows = []
    for eps in epsilons:
        rows.append(
            {
                "application": surface.application,
                "probe": surface.principal_probe,
                "surface": surface.surface,
                "p": p,
                "epsilon": eps,
                "exact_rank": int(found[eps].get("exact_rank", p)),
                "exact_rank_max_error": float(found[eps].get("exact_rank_max_error", np.nan)),
                "certified_rank": int(found[eps].get("certified_rank", p)),
                "certified_rank_max_bound": float(found[eps].get("certified_rank_max_bound", np.nan)),
                "rank_curve_csv": rel(TABLES / f"{surface.key}_rank_curve.csv"),
            }
        )
    return rows


def interval_summary(surface: SurfaceData, evaluation: SurfaceEvaluation, capture_rank: int = 5) -> pd.DataFrame:
    thresholds = [float(x) for x in CONFIG["severe_thresholds"]]
    D = surface.C + surface.rho * np.eye(surface.C.shape[0])
    V_capture = evaluation.eig.vectors[:, : min(capture_rank, surface.C.shape[0])]
    rows = []
    for i, K in enumerate(surface.K_states):
        vals_full = np.linalg.eigvalsh(rc.sym(evaluation.A_stack[i]))
        upper = float(max(float(vals_full.max()), max(thresholds)) * 1.01 + 1.0e-8)
        result = rc.interval_generalized_spectrum(K, D, (min(thresholds) - 1.0e-8, upper), backend="scipy_interval")
        dense_counts = {thr: int(np.sum(vals_full >= thr - rc.ASSERT_TOL)) for thr in thresholds}
        max_resid = float(result.residuals.max()) if len(result.residuals) else 0.0
        if max_resid > float(CONFIG["interval_residual_tolerance"]):
            raise AssertionError(f"Interval generalized eigenpair residual too large for {surface.key}, state {i}: {max_resid}")
        row: dict[str, Any] = {
            "state_index": i,
            "solver_backend": result.backend,
            "max_generalized_residual": max_resid,
            "max_interval_residual": max_resid,
        }
        for thr in thresholds:
            count = rc.severe_direction_count(result.eigenvalues, thr)
            if count != dense_counts[thr]:
                raise AssertionError(f"Interval count disagrees with dense eigendecomposition for {surface.key}, state {i}, threshold {thr}.")
            row[f"d_{int(thr)}"] = count
            if count > 0:
                U = result.whitened_eigenvectors[:, result.eigenvalues >= thr - rc.ASSERT_TOL]
                capture = float(np.linalg.norm(V_capture.T @ U, ord="fro") ** 2 / count)
                max_dist, med_dist = rc.severe_subspace_capture(V_capture, U)
                row[f"kappa{capture_rank}_{int(thr)}"] = min(1.0, max(0.0, capture))
                row[f"max_distance{capture_rank}_{int(thr)}"] = max_dist
                row[f"median_distance{capture_rank}_{int(thr)}"] = med_dist
            else:
                row[f"kappa{capture_rank}_{int(thr)}"] = np.nan
                row[f"max_distance{capture_rank}_{int(thr)}"] = np.nan
                row[f"median_distance{capture_rank}_{int(thr)}"] = np.nan
        rows.append({**surface.state_frame.iloc[i].to_dict(), **row})
    return pd.DataFrame(rows)


def make_macro_figure(surface: SurfaceData, evaluation: SurfaceEvaluation, spectrum: pd.DataFrame) -> None:
    lambda_grid = np.asarray(CONFIG["macro_lambda_grid"], dtype=float)
    selected = [str(x) for x in CONFIG["macro_selected_dates"]]
    dates = surface.state_frame["date_month"].astype(str)
    selected_indices: list[int] = []
    for month in selected:
        matches = np.flatnonzero(dates.eq(month).to_numpy())
        if len(matches):
            selected_indices.append(int(matches[0]))
    if not selected_indices:
        selected_indices = [int(np.argmax(evaluation.frames[5]["q_full"].to_numpy(float)))]

    rows = []
    for idx in selected_indices:
        A = evaluation.A_stack[idx]
        vals, vecs = np.linalg.eigh(rc.sym(A))
        theta5 = rc.sym(A[:5, :5])
        for lam in lambda_grid:
            q_full = stress_from_eig(vals, vecs, evaluation.Q_v, float(lam))
            q_5 = rc.stress_trace(theta5, evaluation.Q_v[:5, :5], float(lam))
            rows.append(
                {
                    "date": surface.state_frame.loc[idx, "date"],
                    "date_month": surface.state_frame.loc[idx, "date_month"],
                    "lambda": float(lam),
                    "q_full": q_full,
                    "q_5": q_5,
                    "exact_error": abs(q_full - q_5),
                }
            )
    curves = pd.DataFrame(rows)
    companion = curves.merge(
        spectrum[["date", "d_2", "d_4", "kappa5_2", "kappa5_4", "max_generalized_residual", "solver_backend"]],
        on="date",
        how="left",
    )
    companion.to_csv(TABLES / "macro_ritz_interval_spectrum_data.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected_indices), 1)))
    for color, idx in zip(colors, selected_indices):
        label = str(surface.state_frame.loc[idx, "date_month"])
        sub = curves[curves["date_month"].eq(label)]
        axes[0].plot(sub["lambda"], sub["q_full"], color=color, label=f"{label} full")
        axes[0].plot(sub["lambda"], sub["q_5"], color=color, linestyle="--", label=f"{label} rank-5 Ritz")
    axes[0].plot(lambda_grid, 1.0 / (1.0 + lambda_grid), color="0.65", linewidth=1.0, label="neutral")
    axes[0].axvline(float(CONFIG["lambda_main"]), color="0.2", linewidth=0.8, alpha=0.6)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("stress")
    axes[0].set_title("A. Full Q-rho stress and unnormalized rank-5 Ritz approximation")
    axes[0].legend(fontsize=7, ncol=2)

    xdates = pd.to_datetime(spectrum["date"])
    axes[1].plot(xdates, spectrum["d_2"], label="a >= 2", color="tab:red")
    axes[1].plot(xdates, spectrum["d_4"], label="a >= 4", color="tab:purple")
    axes[1].set_ylabel("count")
    axes[1].set_title("B. Severe generalized-eigenvalue counts")
    axes[1].legend(fontsize=8)

    axes[2].plot(xdates, spectrum["kappa5_2"], label="capture, a >= 2", color="tab:red")
    axes[2].plot(xdates, spectrum["kappa5_4"], label="capture, a >= 4", color="tab:purple")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("fraction")
    axes[2].set_title("C. Rank-five capture of severe eigenspaces")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.savefig(FIGURES / "macro_ritz_interval_spectrum.pdf")
    fig.savefig(FIGURES / "macro_ritz_interval_spectrum.png", dpi=180)
    plt.close(fig)


def make_lalonde_figure(surface: SurfaceData, probes: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for probe_name, Q in probes.items():
        proxy = SurfaceData(
            key=surface.key,
            application=surface.application,
            surface=surface.surface,
            principal_probe=probe_name,
            C=surface.C,
            K_states=surface.K_states,
            state_frame=surface.state_frame,
            rho=surface.rho,
            Q=Q,
            source_files=surface.source_files,
            state_mask_note=surface.state_mask_note,
            extra=surface.extra,
        )
        evaluation = evaluate_surface(proxy, [5])
        frame = evaluation.frames[5].copy()
        frame["probe"] = probe_name
        frame["support_value"] = surface.state_frame["logit_pscore_original"].to_numpy(float)
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data = data.rename(columns={"q_rank": "q_5"})
    data = data[
        [
            "support_value",
            "probe",
            "q_full",
            "q_5",
            "exact_error",
            "retained_probe_mass",
            "omitted_probe_mass",
            "ritz_subspace_residual_norm",
            "residual_term",
            "uncapped_ritz_bound",
            "capped_ritz_bound",
            "bound_utilization",
        ]
    ]
    if (data["exact_error"] > data["capped_ritz_bound"] + 2.0e-7).any():
        raise AssertionError("LaLonde validation certificate falls below exact error.")
    data.to_csv(TABLES / "lalonde_ritz_validation_data.csv", index=False)

    probes_order = list(probes.keys())
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), constrained_layout=True)
    for j, probe_name in enumerate(probes_order):
        sub = data[data["probe"].eq(probe_name)].sort_values("support_value")
        axes[0, j].plot(sub["support_value"], sub["q_full"], label="full", color="tab:blue")
        axes[0, j].plot(sub["support_value"], sub["q_5"], label="rank 5", color="tab:orange", linestyle="--")
        axes[0, j].set_title(probe_name)
        axes[0, j].grid(True, alpha=0.2)
        axes[1, j].plot(sub["support_value"], np.maximum(sub["exact_error"], 1.0e-6), label="exact error", color="tab:blue")
        axes[1, j].plot(sub["support_value"], np.maximum(sub["capped_ritz_bound"], 1.0e-6), label="capped bound", color="tab:orange", linestyle="--")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("logit propensity")
        axes[1, j].grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("stress")
    axes[1, 0].set_ylabel("lambda = 1")
    axes[0, -1].legend(fontsize=8)
    axes[1, -1].legend(fontsize=8)
    fig.savefig(FIGURES / "lalonde_ritz_validation.pdf")
    fig.savefig(FIGURES / "lalonde_ritz_validation.png", dpi=180)
    plt.close(fig)
    return data


def hurricane_tables(full: SurfaceData, restricted: SurfaceData, full_eval: SurfaceEvaluation, restricted_eval: SurfaceEvaluation) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_a_rows = []
    panel_b_rows = []
    for surface, evaluation in [(full, full_eval), (restricted, restricted_eval)]:
        ranks = [r for r in CONFIG["hurricane_rank_grid"] if r <= surface.C.shape[0]]
        for R in ranks:
            if R not in evaluation.frames:
                continue
            frame = evaluation.frames[R]
            trace_total = max(float(evaluation.eig.values.sum()), 1.0e-12)
            panel_a_rows.append(
                {
                    "surface": surface.surface,
                    "probe": surface.principal_probe,
                    "p": int(surface.C.shape[0]),
                    "rank": int(R),
                    "trace_share": float(evaluation.eig.values[:R].sum() / trace_total),
                    "retained_probe_mass": float(frame["retained_probe_mass"].iloc[0]),
                    "p95_exact_error": float(frame["exact_error"].quantile(0.95)),
                    "max_exact_error": float(frame["exact_error"].max()),
                    "share_error_gt_0_05": float((frame["exact_error"] > 0.05).mean()),
                    "p95_ritz_subspace_residual_term": float(frame["residual_term"].quantile(0.95)),
                    "max_capped_ritz_bound": float(frame["capped_ritz_bound"].max()),
                    "max_uncapped_ritz_bound": float(frame["uncapped_ritz_bound"].max()),
                }
            )
        spectrum = interval_summary(surface, evaluation)
        spectrum.to_csv(TABLES / f"{surface.key}_interval_spectrum.csv", index=False)
        for thr in CONFIG["severe_thresholds"]:
            d_col = f"d_{int(thr)}"
            k_col = f"kappa5_{int(thr)}"
            cond = spectrum.loc[spectrum[d_col] > 0, k_col]
            panel_b_rows.append(
                {
                    "surface": surface.surface,
                    "probe": "not probe-specific",
                    "p": int(surface.C.shape[0]),
                    "threshold": float(thr),
                    "median_d": float(spectrum[d_col].median()),
                    "max_d": int(spectrum[d_col].max()),
                    "share_d_gt_0": float((spectrum[d_col] > 0).mean()),
                    "median_kappa_conditional": float(cond.median()) if len(cond) else np.nan,
                    "p90_kappa_conditional": float(cond.quantile(0.90)) if len(cond) else np.nan,
                    "max_generalized_eigenpair_residual": float(spectrum["max_generalized_residual"].max()),
                    "solver_backend": str(spectrum["solver_backend"].iloc[0]) if len(spectrum) else "",
                }
            )
    panel_a = pd.DataFrame(panel_a_rows)
    panel_b = pd.DataFrame(panel_b_rows)
    panel_a.to_csv(TABLES / "hurricane_ritz_certification_panel_a.csv", index=False)
    panel_b.to_csv(TABLES / "hurricane_ritz_certification_panel_b.csv", index=False)
    pd.concat([panel_a.assign(panel="A"), panel_b.assign(panel="B")], ignore_index=True, sort=False).to_csv(
        TABLES / "hurricane_ritz_certification.csv",
        index=False,
    )
    return panel_a, panel_b


def write_main_latex(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Finite-rank approximation errors and Ritz bounds}",
        r"\label{tab:ritz-certification-main}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Application & Probe & $p$ & $R$ & Retained mass & p95 $e_R$ & $\max e_R$ & $\max B_R$ \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.application)} & {latex_escape(row.probe)} & {int(row.p)} & {int(row.display_R)} & "
            f"{fmt_num(row.retained_probe_mass)} & {fmt_num(row.p95_exact_error)} & {fmt_num(row.max_exact_error)} & {fmt_bound(row.max_ritz_bound)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\footnotesize Notes: $e_R$ is the exact full-coordinate stress approximation error at $\lambda_0=1$ and $B_R$ is the capped deterministic Ritz bound. The macro row uses the full-space temporal-resolvent benchmark built from 125-dimensional pre-PCA influence contributions; it is not a full-dimensional latent state-space estimate. Bounds are conditional on the estimated regularized covariance operators.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (LATEX / "tab_ritz_certification_main.tex").write_text("\n".join(lines), encoding="utf-8")


def write_hurricane_latex(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Hurricane Ritz certification and severe-direction geometry}",
        r"\label{tab:hurricane-ritz-certification}",
        r"\scriptsize",
        r"\textbf{Panel A. Accuracy}\\[0.3em]",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Surface and rank & Probe & Trace share & Retained mass & p95 $e_R$ & $\max e_R$ & Share $e_R>.05$ & p95 Ritz subspace residual term \\",
        r"\midrule",
    ]
    for row in panel_a.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.surface)}, R={int(row.rank)} & {latex_escape(row.probe)} & {fmt_num(row.trace_share)} & {fmt_num(row.retained_probe_mass)} & "
            f"{fmt_num(row.p95_exact_error)} & {fmt_num(row.max_exact_error)} & {fmt_num(row.share_error_gt_0_05)} & {fmt_num(row.p95_ritz_subspace_residual_term)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\\[0.8em]",
            r"\textbf{Panel B. Severe Geometry}\\[0.3em]",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Surface and threshold & Median $d$ & Max $d$ & Share $d>0$ & Median $\kappa$ & p90 $\kappa$ & Max generalized-eigenpair residual \\",
            r"\midrule",
        ]
    )
    for row in panel_b.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.surface)}, $a_\\star={row.threshold:.0f}$ & {fmt_num(row.median_d)} & {int(row.max_d)} & "
            f"{fmt_num(row.share_d_gt_0)} & {fmt_num(row.median_kappa_conditional)} & {fmt_num(row.p90_kappa_conditional)} & {fmt_sci(row.max_generalized_eigenpair_residual)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\footnotesize Notes: The restricted surface is the true fiscal/post intersection derived from checked-in surface metadata. Panel A's residual term is $r_R(s)/\lambda$ with $\lambda=1$, where $r_R(s)$ is the Ritz subspace residual. Panel B defines $\kappa_5(s;a_\star)=\|V_5'U_\star(s)\|_F^2/d_\star(s;a_\star)$ when $d_\star>0$. Interval eigenpairs are extracted with \texttt{scipy.linalg.eigh(..., subset\_by\_value=...)} and verified against full dense eigenvalue counts; the reported residual is the numerical generalized-eigenpair residual.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (LATEX / "tab_hurricane_ritz_certification.tex").write_text("\n".join(lines), encoding="utf-8")


def write_text_snippets(metadata: dict[str, Any]) -> None:
    text = r"""
% Main empirical-section paragraph.
Table~\ref{tab:ritz-certification-main} reports finite-rank approximation errors and Ritz bounds for the principal stress probe in each application at $\lambda_0=1$. The exact error column compares the rank-$R$ Ritz approximation with the full-coordinate stress statistic. The Ritz bound is a deterministic certification bound conditional on the estimated covariance operators. It is intentionally conservative: it combines omitted probe mass with a residual term for the Ritz subspace and is capped at one, so entries marked as vacuous should not be read as evidence that the approximation error is large.

The rows are probe-specific. A low-rank approximation can be accurate for one stress probe while remaining uncertified or inaccurate for another. This is especially important in the hurricane appendix, where the 72-dimensional full-surface rows use a fiscal/post block probe, whereas the restricted 28-dimensional fiscal/post rows use the restricted surface's whole-surface soft probe.

The macro benchmark is a full-space temporal-resolvent benchmark, $K_\eta(t)=\sum_j W_{\eta,tj}\chi_j\chi_j'$, rebuilt from the original 125-dimensional pre-PCA influence contributions. It is not a full-dimensional latent state-space fit. The stress curves in Figure~\ref{fig:macro-ritz-interval-spectrum} use the ridge-adjusted reference scale. Dashed curves are unnormalized rank-five Ritz approximations to the full $Q_\rho$ stress, not legacy trace-normalized rank-five stress statistics.

The temporal-resolvent smoothing parameter is selected by blocked predictive Gaussian quasi-likelihood on the saved eta grid. The metadata reports whether the selected value is an interior or boundary solution and lists the best neighboring candidates. Severe-direction counts, Ritz errors, subspace capture, and plotted stress variation are reported only after selection and do not enter the eta criterion.

Severe-direction diagnostics ask a different question from stress approximation. They count generalized eigen-directions whose ridge-adjusted amplification exceeds a threshold and then report how much of those eigenspaces is captured by the leading empirical covariance subspace. Low capture means that high-amplification directions are geometrically different from the leading covariance directions, even when exact low-rank stress errors appear moderate.

% Appendix inclusions.
\input{results/ritz_certification/latex/tab_ritz_certification_main.tex}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{results/ritz_certification/figures/macro_ritz_interval_spectrum.pdf}
\caption{Macro Ritz stress and interval-spectrum diagnostics. Notes: the full calculation is a 125-dimensional temporal-resolvent benchmark, not a full-dimensional version of the nonlinear rank-five latent state-space model. Eta is selected by blocked predictive Gaussian quasi-likelihood on held-out full influence vectors; the metadata reports whether the selected value is a boundary solution. The rank-five curve compresses the same ridge-relative operator. Amplification thresholds are relative to $\widehat C+\rho I$ and are therefore ridge-adjusted. The interval-spectrum backend is \texttt{scipy.linalg.eigh(..., subset\_by\_value=...)}, with the maximum generalized-eigenpair residual reported in metadata.}
\label{fig:macro-ritz-interval-spectrum}
\end{figure}

\input{results/ritz_certification/latex/tab_hurricane_ritz_certification.tex}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{results/ritz_certification/figures/lalonde_ritz_validation.pdf}
\caption{LaLonde Ritz-bound validation. The lower row uses a log scale so the exact rank-five errors remain visible against the deterministic Ritz certificates.}
\label{fig:lalonde-ritz-validation}
\end{figure}
"""
    note = {
        "paper_source_status": "No primary LaTeX manuscript was found in the workspace; generated fragments and inclusion snippets were written instead.",
        "source_metadata_keys": sorted(metadata.keys()),
    }
    (LATEX / "ritz_paper_insertions.tex").write_text(text.strip() + "\n", encoding="utf-8")
    write_json(META / "paper_integration_status.json", note)


def latex_macro_value(value: Any) -> str:
    if isinstance(value, str):
        return latex_escape(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        av = abs(float(value))
        if 0.0 < av < 0.001 or av >= 1000:
            return f"{float(value):.3g}"
        return f"{float(value):.3f}"
    return latex_escape(value)


def write_result_macros(metadata: dict[str, Any], main_df: pd.DataFrame) -> Path:
    rank_diag = pd.read_csv(TABLES / "ritz_rank_diagnostics.csv")
    h_top = pd.read_csv(ROOT / "applications" / "deryugina_hurricanes" / "outputs" / "tables" / "top_tau_space_time_cells_corrected.csv")
    h_diag = pd.read_csv(ROOT / "applications" / "deryugina_hurricanes" / "outputs" / "tables" / "whitening_diagnostics_corrected.csv").iloc[0]
    macro_extra = metadata["surfaces"]["macro"]["extra"]
    max_resid = metadata["max_interval_residual"]
    validation = metadata["validation_provenance"]
    macro_rank_rows = rank_diag[
        (rank_diag["application"].eq("Macro / monetary policy")) & (rank_diag["epsilon"].astype(float).eq(0.10))
    ]
    values = {
        "RitzMacroEta": macro_extra["selected_eta"],
        "RitzMacroEtaBoundary": macro_extra["eta_boundary"],
        "RitzMacroValidationCriterion": macro_extra["validation_criterion"],
        "RitzMacroEffectiveNMedian": macro_extra["weight_diagnostics"]["effective_n_median"],
        "RitzMacroP": metadata["surfaces"]["macro"]["dimension"],
        "RitzHurricaneFiscalPostDimension": int(h_diag["p_surface"]),
        "RitzHurricaneFiscalPostTraceShare": float(h_diag["top5_trace_share"]),
        "RitzHurricaneTopRegion": str(h_top.iloc[0]["region"]),
        "RitzHurricaneTopYear": int(h_top.iloc[0]["year"]),
        "RitzHurricaneTopTau": float(h_top.iloc[0]["tau"]),
        "RitzMaxIdentityError": validation["max_ritz_compression_identity_error"],
        "RitzMaxCertificateViolation": validation["max_certificate_violation"],
        "RitzMaxGeneralizedEigenpairResidual": max(max_resid.values()),
    }
    if len(macro_rank_rows):
        values["RitzMacroRankTenCertified"] = int(macro_rank_rows.iloc[0]["certified_rank"])
    for row in main_df.itertuples(index=False):
        prefix = {
            "Macro / monetary policy": "RitzMainMacro",
            "LaLonde ATE": "RitzMainLalonde",
            "Hurricane full fiscal/post probe": f"RitzMainHurricaneR{int(row.display_R)}",
        }[row.application]
        values[f"{prefix}RetainedMass"] = float(row.retained_probe_mass)
        values[f"{prefix}P95ExactError"] = float(row.p95_exact_error)
        values[f"{prefix}MaxExactError"] = float(row.max_exact_error)
        values[f"{prefix}MaxBound"] = "1.000 (vacuous)" if float(row.max_ritz_bound) >= 1.0 - 5.0e-8 else float(row.max_ritz_bound)
    lines = ["% Auto-generated from Ritz certification result files. Do not edit numerical values by hand."]
    for name, value in values.items():
        lines.append(rf"\newcommand{{\{name}}}{{{latex_macro_value(value)}}}")
    path = LATEX / "ritz_result_macros.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manuscript_patch_fragment(metadata: dict[str, Any]) -> Path:
    text = r"""
% Self-contained manuscript patch fragment for the Ritz-certification revision.
% This repository snapshot does not contain a primary manuscript root; integrate
% this fragment into the source root used by the paper build.
\input{results/ritz_certification/latex/ritz_result_macros.tex}

\subsection{Bounded Stress Diagnostics}
\label{sec:bounded-stress-diagnostics}

Let $\widehat D=\widehat C+\rho I$ and
$\widehat A_\rho(s)=\widehat D^{-1/2}\widehat K(s)\widehat D^{-1/2}$.
For a positive trace-one probe $Q$ and $\lambda>0$, define
$f_\lambda(a)=a/(\lambda+a)$ and
\[
  \widehat q_{\lambda,Q}(s)
  =
  \operatorname{tr}\{Q f_\lambda(\widehat A_\rho(s))\}.
\]
This bounded stress diagnostic is the population object for display. A
finite-rank display is a numerical and interpretive approximation to the same
ridge-relative operator, not a distinct population construction.

Let $P$ be an orthogonal projection and
$\widehat q^P_{\lambda,Q}(s)=
\operatorname{tr}\{Q P f_\lambda(P\widehat A_\rho(s)P)P\}$. Then
\begin{equation}
\boxed{
\left|
\widehat q_{\lambda,Q}(s)-\widehat q^P_{\lambda,Q}(s)
\right|
\le
\operatorname{tr}\{Q(I-P)\}
+
\frac{\|(I-P)\widehat A_\rho(s)P\|}{\lambda}.
}
\label{eq:ritz-stress-bound}
\end{equation}
The first term is omitted probe mass: it is small only when the displayed
subspace contains the directions to which $Q$ assigns mass. The second term is
the Ritz subspace residual: it vanishes when the selected subspace is invariant
for the local ridge-relative operator. Both terms depend on $Q$ and $\lambda$,
so explained unconditional covariance is not by itself a stress certificate.
The proof appears in Appendix~\ref{app:ritz-bound-proof}.

At the dimensions considered here, we use generalized Hermitian interval
extraction through \texttt{scipy.linalg.eigh(..., subset\_by\_value=...)} and
verify selected counts against the exact dense generalized spectrum. FEAST,
rational Krylov, and shift-and-invert methods target the same spectral
projectors in larger problems; no FEAST computation is used in the empirical
results reported here.

\paragraph{Empirical Ritz results.}
Table~\ref{tab:ritz-certification-main} reports exact finite-rank approximation
errors and generic Ritz upper bounds. The generic bound is conservative because
it controls worst-case cross-subspace coupling; a capped value of one is valid
but empirically vacuous. In the generated results, the LaLonde ATE rank-five
error is \RitzMainLalondeMaxExactError, while the corresponding generic bound is
\RitzMainLalondeMaxBound. The hurricane rank-five error is materially larger
(\RitzMainHurricaneR5MaxExactError), supporting the use of rank-five output as
an interpretive display rather than a stand-alone certification. Rank-five
adequacy is probe-specific: a small ATE-probe error does not establish adequacy
for every LaLonde probe.

\input{results/ritz_certification/latex/tab_ritz_certification_main.tex}

The hurricane principal figure uses the true
\RitzHurricaneFiscalPostDimension-coordinate fiscal/post surface. The highest
amplification cell is \RitzHurricaneTopRegion{} in \RitzHurricaneTopYear{}
with $\tau=\RitzHurricaneTopTau$. Severe-direction counts distinguish
concentrated fragility from multidimensional fragility. Severe-subspace capture
measures whether covariance-PCA contains those directions; low capture means
the largest ridge-relative amplification directions lie outside the dominant
unconditional covariance modes.

\input{results/ritz_certification/latex/tab_hurricane_ritz_certification.tex}

The macro appendix is a full-space temporal-resolvent benchmark with
$p=\RitzMacroP$, not a full-dimensional latent state-space fit. Eta is selected
by \RitzMacroValidationCriterion; the selected value is \RitzMacroEta{} and is
classified as \RitzMacroEtaBoundary{} on the generated grid.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{results/ritz_certification/figures/macro_ritz_interval_spectrum.pdf}
\caption{Macro Ritz stress and interval-spectrum diagnostics. The full calculation is a 125-dimensional temporal-resolvent benchmark. The rank-five curve is the rank-five Ritz approximation to the same full soft-probe stress. Amplification thresholds are ridge-adjusted relative to $\widehat C+\rho I$. The interval-spectrum backend is generalized Hermitian interval extraction; the maximum generalized-eigenpair residual is \RitzMaxGeneralizedEigenpairResidual.}
\label{fig:macro-ritz-interval-spectrum}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.96\linewidth]{results/ritz_certification/figures/lalonde_ritz_validation.pdf}
\caption{LaLonde Ritz-bound validation. The low-dimensional application permits exact calculation of full-versus-rank-five error for the overall ATE, prior-earnings, and education probes. The generic bound is verified and conservative because it controls worst-case cross-subspace coupling. All quantities are deterministic conditional on the estimated covariance operators.}
\label{fig:lalonde-ritz-validation}
\end{figure}

\section{Spectral Compression for Interpretation}
\label{app:spectral-compression-interpretation}

Let $\widehat C=V\Lambda V'$ with eigenvalues in descending order and
$V_R=V[:,1\!:\!R]$. The finite-rank Ritz matrix is
\[
  \Theta_R(s)=V_R'\widehat A_\rho(s)V_R
  =
  (\Lambda_R+\rho I)^{-1/2}
  V_R'\widehat K(s)V_R
  (\Lambda_R+\rho I)^{-1/2}.
\]
The empirical code asserts this identity numerically. The approximation
$\operatorname{tr}\{(V_R' Q V_R) f_\lambda(\Theta_R(s))\}$ is therefore a
Galerkin/Ritz display of the same ridge-relative stress operator.

\subsection{Ritz residual certification}
\label{app:ritz-residual-certification}

For $P_R=V_RV_R'$, the Ritz subspace residual is
$\operatorname{Residual}_R(s)=\widehat A_\rho(s)V_R-V_R\Theta_R(s)$ and
$r_R(s)=\|\operatorname{Residual}_R(s)\|_2$. The generic bound used in the
tables is
\[
  B_R(s;\lambda,Q)
  =
  \min\left\{1,\,
  1-\operatorname{tr}(V_R'QV_R)+r_R(s)/\lambda
  \right\}.
\]
The empirical validation metadata reports maximum Ritz identity error
\RitzMaxIdentityError{} and maximum certificate violation
\RitzMaxCertificateViolation.

\subsection{Proof of the Ritz stress bound}
\label{app:ritz-bound-proof}

Let $A=\widehat A_\rho(s)$ and decompose the space as
$P\mathcal H\oplus(I-P)\mathcal H$. Let
$A_0=PAP+(I-P)A(I-P)$ be the block-diagonal part of $A$. Since
$f_\lambda(x)=x(\lambda+x)^{-1}=I-\lambda(\lambda I+x)^{-1}$ is a positive
contraction on positive semidefinite operators,
$0\le f_\lambda((I-P)A(I-P))\le I-P$. Thus, without assuming that $Q$ commutes
with $P$,
\[
\left|
\operatorname{tr}\{Q f_\lambda(A_0)\}
-
\operatorname{tr}\{Q P f_\lambda(PAP)P\}
\right|
\le
\operatorname{tr}\{Q(I-P)\}.
\]
For the remaining term, the resolvent identity gives
\[
f_\lambda(A)-f_\lambda(A_0)
=
\lambda(\lambda I+A)^{-1}(A-A_0)(\lambda I+A_0)^{-1}.
\]
Both resolvents have norm at most $1/\lambda$, and
$A-A_0=(I-P)AP+PA(I-P)$ has norm $\|(I-P)AP\|$. Hence
\[
\|f_\lambda(A)-f_\lambda(A_0)\|
\le
\|(I-P)AP\|/\lambda.
\]
Because $Q$ is positive with trace one,
$|\operatorname{tr}\{Q[f_\lambda(A)-f_\lambda(A_0)]\}|
\le \|f_\lambda(A)-f_\lambda(A_0)\|$. Combining the two inequalities proves
Equation~\eqref{eq:ritz-stress-bound}.
"""
    path = LATEX / "ritz_manuscript_patch_fragment.tex"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def write_standalone_latex_check() -> Path:
    tex = r"""
\documentclass{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
\input{tab_ritz_certification_main.tex}
\begin{figure}[!htbp]\centering
\includegraphics[width=0.95\linewidth]{../figures/macro_ritz_interval_spectrum.pdf}
\caption{Macro Ritz interval-spectrum diagnostic.}
\end{figure}
\input{tab_hurricane_ritz_certification.tex}
\begin{figure}[!htbp]\centering
\includegraphics[width=0.95\linewidth]{../figures/lalonde_ritz_validation.pdf}
\caption{LaLonde Ritz validation.}
\end{figure}
\end{document}
"""
    path = LATEX / "ritz_standalone_check.tex"
    path.write_text(tex.strip() + "\n", encoding="utf-8")
    return path


def git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_status_label() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, check=False, capture_output=True, text=True)
    except Exception as exc:
        return f"git_unavailable: {exc}"
    if proc.returncode != 0:
        return "not_a_git_repository"
    return "inside_git_repository"


def main() -> int:
    archived = archive_previous_invalid_outputs()
    metadata: dict[str, Any] = {
        "config": CONFIG,
        "git_commit": git_commit(),
        "git_commit_status": git_status_label(),
        "interval_backend": "scipy.linalg.eigh subset_by_value",
        "previous_invalid_proxy_archive": archived,
        "surfaces": {},
    }
    main_rows: list[dict[str, Any]] = []
    rank_diag_rows: list[dict[str, Any]] = []

    macro = macro_surface()
    macro_eval = evaluate_surface(macro, [int(CONFIG["display_rank"])])
    macro_spectrum = interval_summary(macro, macro_eval)
    macro_spectrum.to_csv(TABLES / "macro_interval_spectrum.csv", index=False)
    d2_unique = sorted(int(x) for x in macro_spectrum["d_2"].dropna().unique())
    d4_unique = sorted(int(x) for x in macro_spectrum["d_4"].dropna().unique())
    macro_severe_valid = bool(max(d2_unique or [0]) > 1 and len(d2_unique) > 2 and max(d4_unique or [0]) > 1 and len(d4_unique) > 2)
    macro_severe = {
        "d_2_unique_values": d2_unique,
        "d_4_unique_values": d4_unique,
        "d_2_share_positive": float((macro_spectrum["d_2"] > 0).mean()),
        "d_4_share_positive": float((macro_spectrum["d_4"] > 0).mean()),
        "max_generalized_residual": float(macro_spectrum["max_generalized_residual"].max()),
        "publication_validation_pass": macro_severe_valid,
        "publication_validation_note": "Macro omitted from the main Ritz table when selected eta yields mechanically zero/one severe-direction diagnostics.",
    }
    macro.extra["severe_count_diagnostics"] = macro_severe
    make_macro_figure(macro, macro_eval, macro_spectrum)
    metadata["surfaces"][macro.key] = surface_metadata(macro, macro_eval)
    metadata["surfaces"][macro.key]["publication_ready"] = bool(metadata["surfaces"][macro.key]["publication_ready"] and macro_severe_valid)
    metadata["surfaces"][macro.key]["omitted_from_main_table"] = not macro_severe_valid
    if macro_severe_valid:
        main_rows.append(main_row(macro, macro_eval, int(CONFIG["display_rank"])))
        rank_diag_rows.extend(rank_diagnostics(macro, macro_eval))

    lalonde, lalonde_probes = lalonde_surfaces()
    lalonde_eval = evaluate_surface(lalonde, [int(CONFIG["display_rank"])])
    main_rows.append(main_row(lalonde, lalonde_eval, int(CONFIG["display_rank"])))
    rank_diag_rows.extend(rank_diagnostics(lalonde, lalonde_eval))
    make_lalonde_figure(lalonde, lalonde_probes)
    metadata["surfaces"][lalonde.key] = surface_metadata(lalonde, lalonde_eval)

    hurricane_full, hurricane_restricted = hurricane_surfaces()
    hurricane_full_eval = evaluate_surface(hurricane_full, list(CONFIG["hurricane_main_ranks"]) + list(CONFIG["hurricane_rank_grid"]))
    for R in CONFIG["hurricane_main_ranks"]:
        main_rows.append(main_row(hurricane_full, hurricane_full_eval, int(R)))
    rank_diag_rows.extend(rank_diagnostics(hurricane_full, hurricane_full_eval))
    restricted_eval = evaluate_surface(hurricane_restricted, list(CONFIG["hurricane_rank_grid"]))
    rank_diag_rows.extend(rank_diagnostics(hurricane_restricted, restricted_eval))
    panel_a, panel_b = hurricane_tables(hurricane_full, hurricane_restricted, hurricane_full_eval, restricted_eval)
    metadata["surfaces"][hurricane_full.key] = surface_metadata(hurricane_full, hurricane_full_eval)
    metadata["surfaces"][hurricane_restricted.key] = surface_metadata(hurricane_restricted, restricted_eval)

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(TABLES / "ritz_certification_main.csv", index=False)
    pd.DataFrame(rank_diag_rows).to_csv(TABLES / "ritz_rank_diagnostics.csv", index=False)
    write_main_latex(main_df)
    write_hurricane_latex(panel_a, panel_b)

    metadata["outputs"] = {
        "main_table_csv": rel(TABLES / "ritz_certification_main.csv"),
        "rank_diagnostics_csv": rel(TABLES / "ritz_rank_diagnostics.csv"),
        "macro_eta_grid_csv": rel(TABLES / "macro_temporal_resolvent_eta_grid.csv"),
        "macro_figure_pdf": rel(FIGURES / "macro_ritz_interval_spectrum.pdf"),
        "macro_companion_csv": rel(TABLES / "macro_ritz_interval_spectrum_data.csv"),
        "hurricane_selector_audit_csv": rel(TABLES / "hurricane_selector_audit.csv"),
        "hurricane_table_csv": rel(TABLES / "hurricane_ritz_certification.csv"),
        "hurricane_table_tex": rel(LATEX / "tab_hurricane_ritz_certification.tex"),
        "lalonde_figure_pdf": rel(FIGURES / "lalonde_ritz_validation.pdf"),
        "lalonde_companion_csv": rel(TABLES / "lalonde_ritz_validation_data.csv"),
    }
    metadata["max_interval_residual"] = {
        "macro": float(macro_spectrum["max_generalized_residual"].max()),
        "hurricane_full": float(pd.read_csv(TABLES / "hurricane_full_interval_spectrum.csv")["max_generalized_residual"].max()),
        "hurricane_fiscal_post": float(pd.read_csv(TABLES / "hurricane_fiscal_post_interval_spectrum.csv")["max_generalized_residual"].max()),
    }
    metadata["surfaces"]["macro"]["validation"]["max_generalized_eigenpair_residual"] = metadata["max_interval_residual"]["macro"]
    metadata["surfaces"]["hurricane_full"]["validation"]["max_generalized_eigenpair_residual"] = metadata["max_interval_residual"]["hurricane_full"]
    metadata["surfaces"]["hurricane_fiscal_post"]["validation"]["max_generalized_eigenpair_residual"] = metadata["max_interval_residual"]["hurricane_fiscal_post"]
    metadata["surfaces"]["lalonde"]["validation"]["max_generalized_eigenpair_residual"] = None
    surface_validations = [surface_meta["validation"] for surface_meta in metadata["surfaces"].values()]
    metadata["validation_provenance"] = {
        "max_ritz_compression_identity_error": float(max(v["max_ritz_compression_identity_error"] for v in surface_validations)),
        "max_certificate_violation": float(max(v["max_certificate_violation"] for v in surface_validations)),
        "min_raw_eigenvalue_C": float(min(v["min_raw_eigenvalue_C"] for v in surface_validations)),
        "min_raw_eigenvalue_K_states": float(min(v["min_raw_eigenvalue_K_states"] for v in surface_validations)),
        "min_raw_eigenvalue_A_states": float(min(v["min_raw_eigenvalue_A_states"] for v in surface_validations)),
        "psd_checks_passed": bool(all(v["psd_valid"] for v in surface_validations)),
        "ritz_identity_checks_passed": bool(all(v["ritz_identity_valid"] for v in surface_validations)),
        "certificate_checks_passed": bool(all(v["certificate_valid"] for v in surface_validations)),
        "interval_checks_passed": bool(max(metadata["max_interval_residual"].values()) <= float(CONFIG["interval_residual_tolerance"])),
        "test_results": {
            "status": "pending_external_test_run",
            "commands": [
                "python -m pytest tests\\test_ritz_certification.py -q",
                "python -m pytest applications\\lalonde_ovk\\test_lalonde_ovk.py applications\\deryugina_hurricanes\\tests\\test_ovk_outputs_corrected.py tests\\test_publication_variants.py -q",
                "pdflatex -interaction=nonstopmode -halt-on-error ritz_standalone_check.tex",
            ],
        },
        "publication_readiness_flags": {
            "macro_uses_full_pre_pca_temporal_resolvent": True,
            "macro_uses_stress_screen_for_eta": False,
            "macro_eta_boundary_solution": bool(macro.extra.get("eta_boundary_solution", False)),
            "macro_publication_ready": bool(metadata["surfaces"]["macro"]["publication_ready"]),
            "macro_omitted_from_main_table": bool(metadata["surfaces"]["macro"].get("omitted_from_main_table", False)),
            "hurricane_principal_restricted_dimension": 28,
            "hurricane_probe_column_reported": True,
            "bounds_equal_one_marked_vacuous_in_latex": True,
            "paper_fragments_generated_only": True,
        },
    }
    metadata["paper_source_status"] = "No primary LaTeX manuscript was found in the workspace; generated LaTeX fragments were not inserted into a manuscript."
    write_json(META / "ritz_certification_metadata.json", metadata)
    result_macros = write_result_macros(metadata, main_df)
    patch_fragment = write_manuscript_patch_fragment(metadata)
    write_text_snippets(metadata)
    write_standalone_latex_check()
    metadata["outputs"]["result_macros_tex"] = rel(result_macros)
    metadata["outputs"]["manuscript_patch_fragment_tex"] = rel(patch_fragment)
    write_json(META / "ritz_certification_metadata.json", metadata)
    write_json(META / "publication_readiness_metadata.json", metadata)
    print(f"Wrote Ritz certification outputs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
