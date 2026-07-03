#!/usr/bin/env python3
"""Proxy-IV LP/OVK appendix for the monthly monetary-policy pipeline."""
from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import cho_factor, cho_solve

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception as exc:  # pragma: no cover - environment dependency
    raise RuntimeError("ReportLab is required to build the IV OVK report.") from exc

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from download_iv_data import download_iv_data, sha256_file, write_sources_csv  # noqa: E402
from ovk_data import BASE_OUTCOME_COLUMNS, OUTCOME_SPECS, build_outcome_frame, outcome_group_indices  # noqa: E402
from run_publication_grade_ovk import (  # noqa: E402
    ALPHA_GRID,
    BOOT_BLOCK_LEN,
    BOOT_EM_ITERS,
    BOOT_SEED,
    EM_ITERS,
    H as DEFAULT_H,
    HEADLINE_R,
    L as DEFAULT_L,
    PUBLICATION_WORKERS,
    RANKS,
    ROBUST_NU,
    StateFit,
    arithmetic_matrix_series_from_state,
    arithmetic_outer_product_observations,
    batched_spd_exp,
    circular_block_indices,
    covariance_basis,
    estimate_alpha_and_state,
    estimate_rank_model,
    estimate_full_coordinate_kernel_model,
    fit_em_state_space,
    ffbs_state_draws,
    matrix_series_from_state,
    positive_simultaneous_band,
    principal_angles,
    rank_summary_row,
    run_parallel_tasks,
    scale_shape_from_A,
    smat,
    state_draw_scale_shape,
    surface_shape_from_A,
)
from ovk_nested_workflow import (  # noqa: E402
    ModelParams,
    block_bootstrap_ci,
    build_mean_and_eval_basis,
    covariance_eigenbasis,
    filtered_gamma_path,
    run_predictive_scores,
    tune_model,
    upgraded_state_space_A_from_z,
)


BASE_DIR = Path(os.environ.get("OVK_BASE_DIR", str(REPO_ROOT / "results")))
IV_ROOT = Path(os.environ.get("OVK_IV_ROOT", str(BASE_DIR / "iv_ovk")))
TABLES = IV_ROOT / "tables"
CHARTS = IV_ROOT / "charts"
FIGURES = IV_ROOT / "figures"
CODE_OUT = IV_ROOT / "code"
REPORTS = Path(os.environ.get("OVK_REPORTS_DIR", str(BASE_DIR / "reports")))
FINAL_PDF = Path(os.environ.get("OVK_IV_FINAL_PDF", str(REPORTS / "iv_ovk_report.pdf")))
FINAL_HTML = Path(os.environ.get("OVK_IV_FINAL_HTML", str(REPORTS / "iv_ovk_report.html")))
FINAL_ZIP = Path(os.environ.get("OVK_IV_FINAL_ZIP", str(REPORTS / "iv_ovk_bundle.zip")))
IV_RAW_DIR = Path(os.environ.get("OVK_IV_RAW_DIR", str(REPO_ROOT / "data_raw" / "external" / "iv")))
PROCESSED_PANEL = Path(
    os.environ.get("OVK_IV_PANEL_PATH", str(BASE_DIR / "data_processed" / "iv_proxy_policy_panel.csv"))
)
BASELINE_PANEL_CANDIDATES = [
    Path(os.environ["OVK_IV_BASELINE_PANEL"]) if os.environ.get("OVK_IV_BASELINE_PANEL") else None,
    BASE_DIR / "data_processed" / "processed_panel_three_shock_definitions.csv",
    BASE_DIR / "data_processed" / "ovk_monetary_panel_monthly_fixed_full.csv",
    REPO_ROOT / "data_processed" / "processed_panel_three_shock_definitions.csv",
    REPO_ROOT / "data_processed" / "ovk_monetary_panel_monthly_fixed_full.csv",
]

H = int(os.environ.get("OVK_IV_H", str(DEFAULT_H)))
L = int(os.environ.get("OVK_IV_L", str(DEFAULT_L)))
IV_BOOT_DRAWS = int(os.environ.get("OVK_IV_BOOTSTRAP_DRAWS", os.environ.get("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "1000")))
IV_NESTED_BOOT_DRAWS = int(os.environ.get("OVK_IV_NESTED_BOOT_DRAWS", os.environ.get("OVK_NESTED_SCORE_BOOTSTRAP_DRAWS", "2000")))
IV_NESTED_BOOT_BLOCK = int(os.environ.get("OVK_IV_NESTED_BOOT_BLOCK_LEN", "12"))
IV_MIN_INSTRUMENT_OBS = int(os.environ.get("OVK_IV_MIN_INSTRUMENT_OBS", "12"))
HIGH_TAU_QUANTILE = float(os.environ.get("OVK_IV_HIGH_TAU_QUANTILE", os.environ.get("OVK_IV_HIGH_TAU_THRESHOLD", "0.90")))
DRIVER_FACTOR_QUANTILE = float(os.environ.get("OVK_IV_DRIVER_FACTOR_QUANTILE", os.environ.get("OVK_IV_DRIVER_THRESHOLD", "0.75")))
IV_FACTOR_ATOL = float(os.environ.get("OVK_IV_FACTOR_ATOL", "1e-10"))
IV_FACTOR_RTOL = float(os.environ.get("OVK_IV_FACTOR_RTOL", "1e-8"))
IV_FACTOR_ZERO_TOL = float(os.environ.get("OVK_IV_FACTOR_ZERO_TOL", "1e-14"))

for d in [IV_ROOT, TABLES, CHARTS, FIGURES, CODE_OUT, REPORTS, PROCESSED_PANEL.parent]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class IVVariantSpec:
    key: str = "iv_proxy_policy_dgs1_sf_fed"
    label: str = "Proxy-IV policy indicator: DGS1 instrumented by SF Fed monetary-policy surprise"
    group: str = "iv_proxy"
    score_type: str = "iv"
    outcome_columns: tuple[str, ...] = tuple(BASE_OUTCOME_COLUMNS)
    x_col: str = "dgs1_eom"
    z_col: str = "iv_z_preferred"
    cbi_col: str | None = "CBI_median_fallback"
    run_ranks: tuple[int, ...] = tuple(RANKS) if tuple(RANKS) else (HEADLINE_R,)


SPEC = IVVariantSpec()
_IV_BOOTSTRAP_CONTEXT: dict[str, Any] = {}


def _now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _complete_monthly_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep row offsets equal to calendar-month offsets, preserving missing months as NaNs."""
    if panel.empty or "date" not in panel.columns:
        return panel.copy()
    out = panel.copy()
    out["date"] = _month_start(out["date"])
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        return out
    calendar = pd.DataFrame({"date": pd.date_range(out["date"].min(), out["date"].max(), freq="MS")})
    return calendar.merge(out, on="date", how="left")


def _first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _safe_name(text: str) -> str:
    return str(text).lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")


def _panel_with_outcomes(panel: pd.DataFrame, outcome_columns: tuple[str, ...]) -> pd.DataFrame:
    keep = set(outcome_columns)
    missing = [c for c in outcome_columns if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing requested outcome columns: {', '.join(missing)}")
    drop = [spec.column for spec in OUTCOME_SPECS if spec.column in panel.columns and spec.column not in keep]
    return panel.drop(columns=drop)


def _outcome_columns_for_labels(labels: list[str]) -> list[str]:
    lookup = {spec.label: spec.column for spec in OUTCOME_SPECS}
    return [lookup.get(label, label) for label in labels]


def _normalize_header_frame(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip().lower() for c in raw.columns]
    if any(c in {"date", "year", "month"} or "surprise" in c or c == "mps" for c in cols):
        return raw.copy()
    head = raw.head(12)
    for i in range(len(head)):
        vals = [str(x).strip() for x in head.iloc[i].tolist()]
        low = [v.lower() for v in vals]
        score = sum(v in {"date", "year", "month", "mps", "mps_orth"} or "surprise" in v for v in low)
        if score >= 2:
            out = raw.iloc[i + 1 :].copy()
            out.columns = vals
            return out
    return raw.copy()


def _read_excel_frames(path: Path) -> list[tuple[str, pd.DataFrame]]:
    try:
        xl = pd.ExcelFile(path)
        frames = []
        for sheet in xl.sheet_names:
            try:
                frames.append((sheet, _normalize_header_frame(pd.read_excel(path, sheet_name=sheet))))
            except Exception:
                continue
        return frames
    except Exception:
        try:
            from ovk_data import _read_xlsx_xml_table  # type: ignore

            return [("first_xml_sheet", _normalize_header_frame(_read_xlsx_xml_table(path)))]
        except Exception:
            return []


def _to_datetime_maybe_excel(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.dropna().between(20000, 80000).mean() > 0.5:
            return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def _clean_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.dropna(how="all")
    return out


def _score_sf_column(name: str, kind: str) -> int:
    low = name.lower().strip()
    if low in {"year", "month", "date", "day"} or "date" in low:
        return -100
    score = 0
    surprise_like = any(x in low for x in ["mps", "surprise", "mp", "monetary", "target", "path", "ff4"])
    if surprise_like:
        score += 2
    if "mps" in low:
        score += 3
    if "surprise" in low:
        score += 2
    if "target" in low:
        score += 1
    if any(x in low for x in ["info", "cbi", "fg", "forward guidance", "news"]):
        score -= 4
    if kind == "orth":
        score += 8 if any(x in low for x in ["orth", "orthogonal"]) else -5
    elif kind == "raw":
        if any(x in low for x in ["orth", "orthogonal"]):
            score -= 7
        if low in {"mps", "mp_surprise", "monetary_policy_surprise"}:
            score += 8
    elif kind == "path":
        score += 8 if any(x in low for x in ["gk", "ff4", "path"]) else -5
    return score


def _choose_numeric_column(raw: pd.DataFrame, kind: str) -> tuple[str | None, pd.Series | None, int]:
    best: tuple[int, str | None, pd.Series | None] = (-10_000, None, None)
    for col in raw.columns:
        num = pd.to_numeric(raw[col], errors="coerce")
        finite = int(num.notna().sum())
        if finite == 0:
            continue
        score = _score_sf_column(str(col), kind) + min(finite, 500) // 50
        if score > best[0]:
            best = (score, str(col), num)
    if best[0] <= -5:
        return None, None, best[0]
    return best[1], best[2], best[0]


def _parse_sf_frame(raw: pd.DataFrame, path: Path, sheet_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _clean_columns(raw)
    cols_lower = {str(c).strip().lower(): c for c in raw.columns}
    if {"year", "month"}.issubset(cols_lower):
        date = pd.to_datetime(
            dict(
                year=pd.to_numeric(raw[cols_lower["year"]], errors="coerce"),
                month=pd.to_numeric(raw[cols_lower["month"]], errors="coerce"),
                day=1,
            ),
            errors="coerce",
        )
        date_decision = "year/month columns"
    else:
        date_col = None
        for needle in ["date", "fomc", "meeting"]:
            matches = [orig for low, orig in cols_lower.items() if needle in low]
            if matches:
                date_col = matches[0]
                break
        if date_col is None and len(raw.columns):
            date_col = raw.columns[0]
        if date_col is None:
            return pd.DataFrame(), {"sheet": sheet_name, "error": "no date column"}
        date = _to_datetime_maybe_excel(raw[date_col])
        date_decision = f"date column {date_col}"

    raw_col, raw_values, raw_score = _choose_numeric_column(raw, "raw")
    orth_col, orth_values, orth_score = _choose_numeric_column(raw, "orth")
    path_col, path_values, path_score = _choose_numeric_column(raw, "path")
    if raw_col is None and orth_col is None:
        return pd.DataFrame(), {"sheet": sheet_name, "error": "no raw or orthogonalized surprise column"}

    out = pd.DataFrame({"date": _month_start(pd.Series(date))})
    if raw_values is not None:
        out["iv_z_raw"] = pd.to_numeric(raw_values, errors="coerce")
    if orth_values is not None:
        out["iv_z_orth"] = pd.to_numeric(orth_values, errors="coerce")
    if path_values is not None:
        out["iv_z_gk_path"] = pd.to_numeric(path_values, errors="coerce")
    out = out.dropna(subset=["date"])
    numeric_cols = [c for c in ["iv_z_raw", "iv_z_orth", "iv_z_gk_path"] if c in out.columns]
    if not numeric_cols:
        return pd.DataFrame(), {"sheet": sheet_name, "error": "numeric columns empty after date parsing"}
    out = out.groupby("date", as_index=False)[numeric_cols].sum(min_count=1).sort_values("date").reset_index(drop=True)
    meta = {
        "sheet": sheet_name,
        "date_decision": date_decision,
        "raw_column": raw_col or "",
        "orth_column": orth_col or "",
        "gk_path_column": path_col or "",
        "raw_score": raw_score,
        "orth_score": orth_score,
        "gk_path_score": path_score,
        "source_file": str(path),
        "monthly_rows": int(len(out)),
        "raw_finite": int(out["iv_z_raw"].notna().sum()) if "iv_z_raw" in out.columns else 0,
        "orth_finite": int(out["iv_z_orth"].notna().sum()) if "iv_z_orth" in out.columns else 0,
        "gk_path_finite": int(out["iv_z_gk_path"].notna().sum()) if "iv_z_gk_path" in out.columns else 0,
    }
    out["iv_z_source_file"] = str(path)
    out["iv_z_raw_source_column"] = raw_col or ""
    out["iv_z_orth_source_column"] = orth_col or ""
    out["iv_z_gk_path_source_column"] = path_col or ""
    return out, meta


def parse_sf_fed_surprises(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse SF Fed/Bauer-Swanson surprises into month-start IV columns.

    The parser keeps raw, source-orthogonalized, and explicitly GK/FF4/path-like
    columns when available. Event-level files are summed within month. Months
    with no event are filled later, after merging to the baseline monthly panel.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "iv_z_raw", "iv_z_orth"]), {"available": False, "error": "file not found", "source_file": str(path)}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frames = _read_excel_frames(path)
    else:
        frames = [(path.stem, _normalize_header_frame(pd.read_csv(path)))]
    candidates = []
    decisions = []
    for sheet, raw in frames:
        parsed, meta = _parse_sf_frame(raw, path, sheet)
        decisions.append(meta)
        if parsed.empty:
            continue
        score = meta.get("raw_finite", 0) + 2 * meta.get("orth_finite", 0) + meta.get("gk_path_finite", 0)
        candidates.append((score, parsed, meta))
    if not candidates:
        return pd.DataFrame(columns=["date", "iv_z_raw", "iv_z_orth"]), {
            "available": False,
            "source_file": str(path),
            "parser_decisions": decisions,
            "error": "no parseable sheet/frame",
        }
    _, parsed, meta = max(candidates, key=lambda item: item[0])
    meta = {**meta, "available": True, "parser_decisions": decisions}
    return parsed, meta


def ensure_iv_data(raw_dir: Path = IV_RAW_DIR) -> dict[str, Any]:
    return download_iv_data(raw_dir)


def build_iv_policy_panel(
    baseline_panel_path: Path | None = None,
    raw_dir: Path = IV_RAW_DIR,
    out_path: Path = PROCESSED_PANEL,
    allow_download: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_panel_path = baseline_panel_path or _first_existing(BASELINE_PANEL_CANDIDATES)
    if baseline_panel_path is None:
        raise FileNotFoundError("Could not find a baseline processed panel for IV merge.")
    if allow_download:
        data_meta = ensure_iv_data(raw_dir)
    else:
        meta_path = raw_dir / "iv_data_sources.json"
        data_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"sources": []}

    panel = pd.read_csv(baseline_panel_path, parse_dates=["date"])
    panel = _complete_monthly_calendar(panel)
    dgs1_path = raw_dir / "dgs1_monthly_eom.csv"
    if not dgs1_path.exists():
        raise FileNotFoundError(f"Processed DGS1 monthly file not found: {dgs1_path}")
    dgs1 = pd.read_csv(dgs1_path, parse_dates=["date"])
    dgs1["date"] = _month_start(dgs1["date"])

    sf_path: Path | None = Path(data_meta["sf_fed_path"]) if isinstance(data_meta, dict) and data_meta.get("sf_fed_path") else None
    if sf_path is None and isinstance(data_meta, dict):
        for source in data_meta.get("sources", []):
            if source.get("source") == "sf_fed_monetary_policy_surprises" and source.get("path"):
                sf_path = Path(source["path"])
                break
    if sf_path is None or not sf_path.exists() or sf_path.is_dir():
        sf_path = _first_existing([raw_dir / "sf_fed_monetary_policy_surprises.xlsx", raw_dir / "sf_fed_monetary_policy_surprises.csv"]) or sf_path
    if sf_path is None:
        raise FileNotFoundError("SF Fed monetary-policy surprise file not found in IV raw directory.")
    sf_monthly, sf_meta = parse_sf_fed_surprises(sf_path)
    if sf_monthly.empty:
        raise RuntimeError(f"Could not parse a usable SF Fed monetary-policy surprise file: {sf_path}")
    sf_monthly["date"] = _month_start(sf_monthly["date"])

    out = panel.merge(dgs1, on="date", how="left").merge(sf_monthly, on="date", how="left")
    source_start = sf_monthly["date"].min()
    source_end = sf_monthly["date"].max()
    in_source_range = out["date"].between(source_start, source_end)
    for col in ["iv_z_raw", "iv_z_orth", "iv_z_gk_path"]:
        if col in out.columns:
            out.loc[in_source_range, col] = pd.to_numeric(out.loc[in_source_range, col], errors="coerce").fillna(0.0)
        else:
            out[col] = np.nan

    raw_finite = int(np.isfinite(pd.to_numeric(out["iv_z_raw"], errors="coerce")).sum())
    orth_finite = int(np.isfinite(pd.to_numeric(out["iv_z_orth"], errors="coerce")).sum())
    if orth_finite >= IV_MIN_INSTRUMENT_OBS:
        preferred = "iv_z_orth"
        preferred_source_col = sf_meta.get("orth_column", "")
    elif raw_finite >= IV_MIN_INSTRUMENT_OBS:
        preferred = "iv_z_raw"
        preferred_source_col = sf_meta.get("raw_column", "")
    else:
        preferred = "iv_z_orth" if orth_finite >= raw_finite else "iv_z_raw"
        preferred_source_col = sf_meta.get("orth_column" if preferred == "iv_z_orth" else "raw_column", "")
    out["iv_z_preferred"] = pd.to_numeric(out[preferred], errors="coerce")
    out["iv_z_source_column"] = preferred_source_col
    out["iv_z_source_file"] = str(sf_path)
    out["iv_z_preferred_kind"] = preferred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    meta = {
        "baseline_panel_path": str(baseline_panel_path),
        "processed_panel_path": str(out_path),
        "sf_fed": sf_meta,
        "data_sources": data_meta,
        "instrument_preferred": preferred,
        "instrument_source_column": preferred_source_col,
        "source_start": source_start.strftime("%Y-%m-%d"),
        "source_end": source_end.strftime("%Y-%m-%d"),
        "raw_finite_after_merge": raw_finite,
        "orth_finite_after_merge": orth_finite,
        "panel_rows": int(len(out)),
    }
    return out, meta


def _residualize(C: np.ndarray, A: np.ndarray, ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, float)
    A = np.asarray(A, float)
    XtX = C.T @ C
    XtA = C.T @ A
    coef = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtA)
    return A - C @ coef, coef


def first_stage_diagnostics_from_residuals(
    x_res: np.ndarray,
    z_res: np.ndarray,
    dates: pd.Series,
    x_col: str,
    z_col: str,
    cbi_col: str | None,
    q_zx: float | None = None,
) -> dict[str, Any]:
    x_res = np.asarray(x_res, float)
    z_res = np.asarray(z_res, float)
    mask = np.isfinite(x_res) & np.isfinite(z_res)
    x = x_res[mask]
    z = z_res[mask]
    if len(x) == 0:
        raise ValueError("No finite first-stage residual observations.")
    q = float(np.mean(z * x)) if q_zx is None else float(q_zx)
    zz = float(np.sum(z * z))
    pi_hat = float(np.sum(z * x) / zz) if zz > 0 else np.nan
    fitted = pi_hat * z if np.isfinite(pi_hat) else np.full_like(z, np.nan)
    resid = x - fitted
    sst = float(np.sum((x - x.mean()) ** 2))
    sse = float(np.sum(resid**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    corr = float(np.corrcoef(z, x)[0, 1]) if len(x) > 1 and np.nanstd(z) > 0 and np.nanstd(x) > 0 else np.nan
    sigma2 = sse / max(len(x) - 1, 1)
    se_pi = math.sqrt(sigma2 / zz) if zz > 0 and sigma2 >= 0 else np.nan
    f_stat = float((pi_hat / se_pi) ** 2) if np.isfinite(pi_hat) and np.isfinite(se_pi) and se_pi > 0 else np.nan
    partial_r2 = float(corr**2) if np.isfinite(corr) else r2
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    return {
        "nobs": int(len(x)),
        "sample_start": dates.iloc[0].strftime("%Y-%m-%d") if len(dates) else "",
        "sample_end": dates.iloc[-1].strftime("%Y-%m-%d") if len(dates) else "",
        "x_col": x_col,
        "z_col": z_col,
        "cbi_col": cbi_col or "",
        "q_zx": q,
        "corr_zx": corr,
        "pi_hat": pi_hat,
        "first_stage_r2": r2,
        "first_stage_partial_r2": partial_r2,
        "first_stage_f_stat": f_stat,
        "x_res_sd": float(np.nanstd(x, ddof=0)),
        "z_res_sd": float(np.nanstd(z, ddof=0)),
        "weak_iv_warning": bool(np.isfinite(f_stat) and f_stat < 10.0),
    }


def iv_lp_scores_from_design(
    C_design: np.ndarray,
    x_vector: np.ndarray,
    z_vector: np.ndarray,
    Y_resp: np.ndarray,
    ix: np.ndarray | None = None,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct exactly identified scalar-IV influence score surfaces.

    The OLS-style LP score used elsewhere in the pipeline is approximately
    ``psi_ols_t = x_res[t] * u[t, :] / mean(x_res**2)``. For scalar exactly
    identified IV, this function uses
    ``psi_iv_t = z_res[t] * u_iv[t, :] / mean(z_res * x_res)``, where
    ``beta_iv = mean(z_res[:, None] * Y_res, axis=0) / mean(z_res * x_res)``
    and ``u_iv = Y_res - x_res[:, None] * beta_iv[None, :]``.
    """
    C = np.asarray(C_design, float)
    x = np.asarray(x_vector, float)
    z = np.asarray(z_vector, float)
    Y = np.asarray(Y_resp, float)
    if ix is not None:
        ix = np.asarray(ix, dtype=int)
        C = C[ix]
        x = x[ix]
        z = z[ix]
        Y = Y[ix]
    resid, coef = _residualize(C, np.column_stack([x, z, Y]), ridge=ridge)
    x_res = resid[:, 0]
    z_res = resid[:, 1]
    Y_res = resid[:, 2:]
    q_zx = float(np.mean(z_res * x_res))
    if not np.isfinite(q_zx) or abs(q_zx) < 1e-12:
        raise RuntimeError(f"IV first-stage covariance is too close to zero: {q_zx}")
    beta_iv = np.mean(z_res[:, None] * Y_res, axis=0) / q_zx
    u = Y_res - x_res[:, None] * beta_iv[None, :]
    psi_iv = z_res[:, None] * u / q_zx
    zz = float(np.sum(z_res * z_res))
    pi_hat = float(np.sum(z_res * x_res) / zz) if zz > 0 else np.nan
    info = {
        "beta_iv": beta_iv,
        "x_res": x_res,
        "z_res": z_res,
        "Y_res": Y_res,
        "u_iv": u,
        "q_zx": q_zx,
        "pi_hat": pi_hat,
        "first_stage_fitted": pi_hat * z_res if np.isfinite(pi_hat) else np.full_like(z_res, np.nan),
        "control_coef": coef,
    }
    return psi_iv, info


def build_iv_lp_scores(
    panel: pd.DataFrame,
    x_col: str = SPEC.x_col,
    z_col: str = SPEC.z_col,
    cbi_col: str | None = SPEC.cbi_col,
    H: int = H,
    L: int = L,
    outcome_columns: tuple[str, ...] | None = None,
    sample_dates: pd.Series | np.ndarray | list[Any] | None = None,
) -> dict[str, Any]:
    """Build IV-LP score surfaces from 2SLS/GMM influence contributions.

    The endogenous policy indicator ``x_col`` and excluded instrument ``z_col``
    are residualized on the common LP control matrix. The returned ``Q_scores``
    are IV influence contributions, not unrestricted time-varying structural
    IRFs. The OVK object is ``K_iv = E[psi_iv_t psi_iv_t']`` and the retained
    ``A_t_iv`` path is a time-varying covariance amplification within that
    score-surface eigenspace.
    """
    panel = _complete_monthly_calendar(panel)
    panel = _panel_with_outcomes(panel, tuple(outcome_columns or BASE_OUTCOME_COLUMNS))
    Ybase = build_outcome_frame(panel)
    labels = list(Ybase.columns)
    Yarr = Ybase.to_numpy(float)
    dY = np.vstack([np.full((1, Yarr.shape[1]), np.nan), np.diff(Yarr, axis=0)])
    xvals = pd.to_numeric(panel[x_col], errors="coerce").to_numpy(float)
    zvals = pd.to_numeric(panel[z_col], errors="coerce").to_numpy(float)
    if cbi_col is not None and cbi_col in panel.columns:
        cvals = pd.to_numeric(panel[cbi_col], errors="coerce").to_numpy(float)
    elif cbi_col is None:
        cvals = np.full(len(panel), np.nan)
    else:
        raise ValueError(f"Requested CBI control column is missing: {cbi_col}")

    sample_months: set[pd.Timestamp] | None = None
    if sample_dates is not None:
        sample_months = set(pd.to_datetime(pd.Series(sample_dates)).dt.to_period("M").dt.to_timestamp())

    valid: list[int] = []
    for t in range(len(panel)):
        if t - L < 0 or t - 1 < 0 or t + H >= len(panel):
            continue
        if sample_months is not None and panel["date"].iloc[t].to_period("M").to_timestamp() not in sample_months:
            continue
        checks = [
            np.isfinite(xvals[t]),
            np.isfinite(zvals[t]),
            np.isfinite(xvals[t - L : t]).all(),
            np.isfinite(Yarr[t - 1 : t + H + 1, :]).all(),
            np.isfinite(Yarr[t - L : t, :]).all(),
            np.isfinite(dY[t - L : t, :]).all(),
        ]
        if cbi_col is not None:
            checks += [np.isfinite(cvals[t]), np.isfinite(cvals[t - L : t]).all()]
        if all(checks):
            valid.append(t)
    valid_idx = np.asarray(valid, dtype=int)
    if len(valid_idx) < max(30, H + L + 5):
        raise RuntimeError(f"IV sample is too short after controls/horizons: {len(valid_idx)} observations.")
    dates = panel["date"].iloc[valid_idx].reset_index(drop=True)

    controls: list[np.ndarray] = [np.ones(len(valid_idx)), valid_idx.astype(float)]
    if cbi_col is not None:
        controls.append(cvals[valid_idx])
    for lag in range(1, L + 1):
        controls.append(xvals[valid_idx - lag])
        if cbi_col is not None:
            controls.append(cvals[valid_idx - lag])
    for lag in range(1, L + 1):
        controls += [Yarr[valid_idx - lag, :], dY[valid_idx - lag, :]]
    C = np.hstack([np.asarray(a)[:, None] if np.asarray(a).ndim == 1 else np.asarray(a) for a in controls])
    C_scaled = C.copy()
    mu = C_scaled[:, 1:].mean(axis=0)
    sd = C_scaled[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    C_scaled[:, 1:] = (C_scaled[:, 1:] - mu) / sd
    Y_resp = np.hstack([Yarr[valid_idx + hh, :] - Yarr[valid_idx - 1, :] for hh in range(H + 1)])
    x_vector = xvals[valid_idx]
    z_vector = zvals[valid_idx]
    Q_scores, info = iv_lp_scores_from_design(C_scaled, x_vector, z_vector, Y_resp)
    diagnostics = first_stage_diagnostics_from_residuals(
        info["x_res"],
        info["z_res"],
        dates,
        x_col=x_col,
        z_col=z_col,
        cbi_col=cbi_col,
        q_zx=info["q_zx"],
    )
    return {
        "Q_scores": Q_scores,
        "dates": dates,
        "valid_idx": valid_idx,
        "C_design": C_scaled,
        "x_vector": x_vector,
        "z_vector": z_vector,
        "Y_resp": Y_resp,
        "x_res": info["x_res"],
        "z_res": info["z_res"],
        "Y_res": info["Y_res"],
        "u_iv": info["u_iv"],
        "first_stage_fitted": info["first_stage_fitted"],
        "pi_hat": info["pi_hat"],
        "q_zx": info["q_zx"],
        "beta_iv": info["beta_iv"],
        "first_stage": diagnostics,
        "outcome_labels": labels,
        "outcome_columns": _outcome_columns_for_labels(labels),
        "score_type": "iv",
        "x_col": x_col,
        "z_col": z_col,
        "cbi_col": cbi_col or "",
        "score_transform": {"kind": "raw"},
        "H": H,
        "L": L,
        "pvars": len(labels),
    }


def savefig(fig: plt.Figure, name: str) -> Path:
    path = CHARTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def write_first_stage_outputs(scores: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    diag = pd.DataFrame([scores["first_stage"]])
    diag.to_csv(TABLES / "iv_first_stage_summary.csv", index=False)
    fig = plt.figure(figsize=(7.2, 5.0))
    plt.scatter(scores["z_res"], scores["x_res"], s=18, alpha=0.65)
    z = np.asarray(scores["z_res"], float)
    fitted = np.asarray(scores["first_stage_fitted"], float)
    order = np.argsort(z)
    plt.plot(z[order], fitted[order], color="black", linewidth=1.4)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Residualized external instrument")
    plt.ylabel("Residualized DGS1 policy indicator")
    plt.title("IV first stage: residualized DGS1 on SF Fed surprise")
    chart = savefig(fig, "iv_first_stage_scatter.png")
    return diag, chart


def write_average_iv_responses(scores: dict[str, Any]) -> pd.DataFrame:
    labels = scores["outcome_labels"]
    beta = np.asarray(scores["beta_iv"], float).reshape(H + 1, len(labels))
    df = pd.DataFrame(beta, columns=labels)
    df.insert(0, "horizon_months", np.arange(H + 1))
    df.to_csv(TABLES / "iv_average_lp_responses.csv", index=False)
    fig = plt.figure(figsize=(8.6, 5.2))
    for label in labels:
        plt.plot(df["horizon_months"], df[label], marker="o", linewidth=1.2, markersize=2.8, label=label)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Horizon (months)")
    plt.ylabel("IV LP response")
    plt.title("Average proxy-IV LP responses")
    plt.legend(ncol=2, fontsize=8)
    savefig(fig, "iv_average_lp_responses.png")
    return df


def run_rank_models(scores: dict[str, Any], spec: IVVariantSpec = SPEC) -> dict[int, Any]:
    results: dict[int, Any] = {}
    ranks = tuple(int(r) for r in spec.run_ranks if int(r) > 0)
    if HEADLINE_R not in ranks:
        ranks = tuple(sorted(set(ranks + (HEADLINE_R,))))
    for rank in ranks:
        results[rank] = estimate_rank_model(
            scores["Q_scores"],
            scores["dates"],
            spec.key,
            spec.label,
            rank,
            em_iters=EM_ITERS,
            outcome_labels=scores["outcome_labels"],
        )
    return results


def write_rank_and_tau_outputs(results: dict[int, Any], scores: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [rank_summary_row(res) for _, res in sorted(results.items())]
    rank_df = pd.DataFrame(rows)
    rank_df.insert(1, "score_type", "iv")
    rank_df.to_csv(TABLES / "iv_rank_summary.csv", index=False)
    headline = results[HEADLINE_R]
    dates = pd.to_datetime(headline.dates)
    tau_df = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "tau": headline.tau})
    for r in range(HEADLINE_R):
        tau_df[f"A{r+1}{r+1}"] = headline.A[:, r, r]
    tau_df["robust_observation_weight"] = headline.fit.weights
    tau_df.to_csv(TABLES / "iv_tau_path.csv", index=False)

    top_idx = np.argsort(headline.tau)[::-1][:20]
    top_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(top_idx) + 1),
            "date": dates.iloc[top_idx].dt.strftime("%Y-%m-%d").to_numpy(),
            "tau": headline.tau[top_idx],
        }
    )
    for r in range(HEADLINE_R):
        top_df[f"A{r+1}{r+1}"] = headline.A[top_idx, r, r]
    top_df.to_csv(TABLES / "iv_top_amplification_months.csv", index=False)

    labels = scores["outcome_labels"]
    surface = surface_shape_from_A(headline.A, headline.V, H, len(labels), labels)
    metric_rows = []
    for rank, idx in enumerate(top_idx[:10], start=1):
        row = {
            "rank": rank,
            "date": dates.iloc[idx].strftime("%Y-%m-%d"),
            "tau": float(headline.tau[idx]),
            "surface_shape_rms_log_relative": float(surface["surface_shape_rms_log_relative"][idx]),
        }
        for name, values in surface["metrics"].items():
            if name.startswith("variable_share_") or name in {
                "macro_variable_share",
                "financial_variable_share",
                "short_horizon_share",
                "medium_horizon_share",
                "long_horizon_share",
                "cell_effective_support",
            }:
                row[name] = float(values[idx])
        metric_rows.append(row)
    shape_df = pd.DataFrame(metric_rows)
    shape_df.to_csv(TABLES / "iv_shape_decomposition_top_months.csv", index=False)

    fig = plt.figure(figsize=(8.0, 4.8))
    shares = np.asarray(headline.shares[: min(20, len(headline.shares))], float)
    plt.bar(np.arange(1, len(shares) + 1), shares)
    plt.axvline(HEADLINE_R + 0.5, color="black", linewidth=0.9)
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Trace share")
    plt.title("IV OVK eigenspectrum")
    savefig(fig, "iv_eigenspectrum.png")

    fig = plt.figure(figsize=(9.2, 5.0))
    plt.plot(dates, headline.tau, color="black", linewidth=1.4)
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t = trace(A_t)/5")
    plt.title("Proxy-IV rank-five covariance amplification")
    savefig(fig, "iv_tau_path.png")

    plot_basis_surfaces(headline.V, labels, H, HEADLINE_R)
    return rank_df, tau_df


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, float)).rank(pct=True, method="average").to_numpy(float)


def write_score_energy_decomposition(scores: dict[str, Any], headline: Any) -> dict[str, Any]:
    """Decompose retained IV score energy into instrument and residual drivers.

    For the retained rank-R basis V_R, the raw IV score coordinate is
    ``z_res[t] * (u_iv[t] @ V_R) / q_zx``. Since the scalar first stage gives
    ``first_stage_fitted[t] = pi_hat * z_res[t]``, the same coordinate can be
    written as ``first_stage_fitted[t] * (u_iv[t] @ V_R) / (pi_hat * q_zx)``.
    The max absolute difference between the two forms is reported as an
    implementation identity check. Its energy separates into first-stage
    leverage and retained residual energy.
    """
    dates = pd.to_datetime(scores["dates"]).reset_index(drop=True)
    z_res = np.asarray(scores["z_res"], float)
    fitted = np.asarray(scores["first_stage_fitted"], float)
    u_iv = np.asarray(scores["u_iv"], float)
    V = np.asarray(headline.V[:, : headline.rank], float)
    q_zx = float(scores["q_zx"])
    pi_hat = float(scores.get("pi_hat", np.nan))
    eps = 1e-14

    retained_residual = u_iv @ V
    retained_residual_energy = np.sum(retained_residual**2, axis=1)
    z_score_coords = z_res[:, None] * retained_residual / q_zx
    if np.isfinite(pi_hat) and abs(pi_hat) > eps:
        fitted_score_coords = fitted[:, None] * retained_residual / (pi_hat * q_zx)
    else:
        fitted_score_coords = np.full_like(z_score_coords, np.nan)
    z_energy = np.sum(z_score_coords**2, axis=1)
    fitted_energy = np.sum(fitted_score_coords**2, axis=1)
    max_abs_score_diff = float(np.nanmax(np.abs(z_score_coords - fitted_score_coords)))
    max_abs_energy_diff = float(np.nanmax(np.abs(z_energy - fitted_energy)))

    instrument_leverage = z_res**2 / max(float(np.nanmean(z_res**2)), eps)
    first_stage_leverage = fitted**2 / max(float(np.nanmean(fitted**2)), eps)
    normalized_residual_energy = retained_residual_energy / max(float(np.nanmean(retained_residual_energy)), eps)
    log_first_stage_leverage = np.log(np.maximum(first_stage_leverage, eps))
    log_instrument_leverage = np.log(np.maximum(instrument_leverage, eps))
    log_retained_residual_energy = np.log(np.maximum(normalized_residual_energy, eps))
    leverage_pct = _percentile_rank(log_first_stage_leverage)
    residual_pct = _percentile_rank(log_retained_residual_energy)
    pct_diff = leverage_pct - residual_pct
    driver_label = np.where(
        pct_diff > 0.10,
        "first-stage-driven",
        np.where(pct_diff < -0.10, "residual-driven", "mixed"),
    )

    path_df = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "tau": headline.tau,
            "z_res": z_res,
            "x_res": scores["x_res"],
            "first_stage_fitted": fitted,
            "instrument_leverage": instrument_leverage,
            "first_stage_leverage": first_stage_leverage,
            "retained_residual_energy": retained_residual_energy,
            "normalized_retained_residual_energy": normalized_residual_energy,
            "retained_score_energy_z_form": z_energy,
            "retained_score_energy_fitted_form": fitted_energy,
            "abs_diff_z_vs_fitted_score_form": np.max(np.abs(z_score_coords - fitted_score_coords), axis=1),
            "abs_diff_z_vs_fitted_energy_form": np.abs(z_energy - fitted_energy),
            "log_first_stage_leverage": log_first_stage_leverage,
            "log_instrument_leverage": log_instrument_leverage,
            "log_retained_residual_energy": log_retained_residual_energy,
            "first_stage_leverage_percentile": leverage_pct,
            "retained_residual_energy_percentile": residual_pct,
            "driver_percentile_diff": pct_diff,
            "driver_label": driver_label,
        }
    )
    top_idx = np.argsort(np.asarray(headline.tau, float))[::-1][:10]
    top_df = path_df.iloc[top_idx].copy()
    top_df.insert(0, "tau_rank", np.arange(1, len(top_df) + 1))
    counts = top_df["driver_label"].value_counts().to_dict()
    max_idx = int(np.argmax(headline.tau))
    corr_first = float(path_df["tau"].corr(path_df["log_first_stage_leverage"]))
    corr_resid = float(path_df["tau"].corr(path_df["log_retained_residual_energy"]))
    summary = pd.DataFrame(
        [
            {
                "max_abs_diff_between_z_score_and_fitted_score_forms": max_abs_score_diff,
                "max_abs_diff_between_z_energy_and_fitted_energy_forms": max_abs_energy_diff,
                "max_tau_month": dates.iloc[max_idx].strftime("%Y-%m-%d"),
                "max_tau": float(headline.tau[max_idx]),
                "driver_label_for_max_tau_month": str(path_df.iloc[max_idx]["driver_label"]),
                "top10_first_stage_driven_count": int(counts.get("first-stage-driven", 0)),
                "top10_residual_driven_count": int(counts.get("residual-driven", 0)),
                "top10_mixed_count": int(counts.get("mixed", 0)),
                "corr_tau_log_first_stage_leverage": corr_first,
                "corr_tau_log_retained_residual_energy": corr_resid,
                "path_csv": str(TABLES / "iv_score_energy_decomposition_path.csv"),
                "top_tau_csv": str(TABLES / "iv_top_tau_decomposition.csv"),
                "summary_csv": str(TABLES / "iv_score_decomposition_summary.csv"),
                "chart_png": str(CHARTS / "iv_score_decomposition_top_tau.png"),
            }
        ]
    )
    path_csv = TABLES / "iv_score_energy_decomposition_path.csv"
    top_csv = TABLES / "iv_top_tau_decomposition.csv"
    summary_csv = TABLES / "iv_score_decomposition_summary.csv"
    path_df.to_csv(path_csv, index=False)
    top_df.to_csv(top_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig = plt.figure(figsize=(9.4, 5.2))
    x = np.arange(len(top_df))
    width = 0.38
    plt.bar(x - width / 2, top_df["first_stage_leverage_percentile"], width=width, label="First-stage leverage percentile")
    plt.bar(x + width / 2, top_df["retained_residual_energy_percentile"], width=width, label="Residual energy percentile")
    plt.xticks(x, pd.to_datetime(top_df["date"]).dt.strftime("%Y-%m"), rotation=45, ha="right")
    for i, label in enumerate(top_df["driver_label"]):
        plt.text(i, 1.03, str(label).replace("-driven", ""), ha="center", va="bottom", fontsize=7, rotation=45)
    plt.ylim(0, 1.16)
    plt.ylabel("Within-sample percentile")
    plt.title("Score-energy drivers for top 10 proxy-IV tau months")
    plt.legend(fontsize=8)
    chart = savefig(fig, "iv_score_decomposition_top_tau.png")

    return {
        "path": path_df,
        "top_tau": top_df,
        "summary": summary,
        "chart": chart,
    }


def plot_basis_surfaces(V: np.ndarray, labels: list[str], H: int, rank: int) -> Path:
    fig, axes = plt.subplots(1, rank, figsize=(3.1 * rank, 5.0), squeeze=False)
    vmax = np.nanmax(np.abs(V[:, :rank])) if V.size else 1.0
    for r in range(rank):
        ax = axes[0, r]
        surface = V[:, r].reshape(H + 1, len(labels))
        im = ax.imshow(surface, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(f"Basis {r+1}")
        ax.set_xlabel("Variable")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        if r == 0:
            ax.set_ylabel("Horizon")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72)
    return savefig(fig, "iv_top5_basis_surfaces.png")


def trace_decomposition_from_augmented_covariance(Sigma_aug: np.ndarray, R: int) -> dict[str, np.ndarray]:
    """Return the block trace decomposition for stacked (reduced-form, FS) states."""
    Sigma_aug = np.asarray(Sigma_aug, float)
    if Sigma_aug.ndim != 3 or Sigma_aug.shape[1:] != (2 * R, 2 * R):
        raise ValueError(f"Expected Sigma_aug with shape (T, {2 * R}, {2 * R}), got {Sigma_aug.shape}")
    Sigma_RR = Sigma_aug[:, :R, :R]
    Sigma_RF = Sigma_aug[:, :R, R:]
    Sigma_FR = Sigma_aug[:, R:, :R]
    Sigma_FF = Sigma_aug[:, R:, R:]
    tau_rf = np.trace(Sigma_RR, axis1=1, axis2=2) / R
    tau_fs = np.trace(Sigma_FF, axis1=1, axis2=2) / R
    tau_cross = np.trace(Sigma_FR + Sigma_RF, axis1=1, axis2=2) / R
    tau_total_decomp = tau_rf + tau_fs + tau_cross
    A_aug_implied = Sigma_RR + Sigma_FF + Sigma_RF + Sigma_FR
    tau_aug_implied = np.trace(A_aug_implied, axis1=1, axis2=2) / R
    return {
        "tau_rf": tau_rf,
        "tau_fs": tau_fs,
        "tau_res": tau_rf,
        "tau_cross": tau_cross,
        "tau_total_decomp": tau_total_decomp,
        "tau_total_dec": tau_total_decomp,
        "tau_aug_implied": tau_aug_implied,
        "A_aug_implied": A_aug_implied,
    }


def _spd_sqrt(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    w, U = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return U @ np.diag(np.sqrt(w)) @ U.T


def _spd_invsqrt_local(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    w, U = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return U @ np.diag(1.0 / np.sqrt(w)) @ U.T


def _recover_unlifted_covariance_from_state(Z: np.ndarray, fit: StateFit) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover Sigma_t from the same alpha ridge lift used by the log-SPD model."""
    Z = np.asarray(Z, float)
    alpha = float(fit.alpha)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"Augmented decomposition requires alpha in (0, 1); got {alpha}")
    d = Z.shape[1]
    eye = np.eye(d)
    G_obs = alpha * eye[None, :, :] + (1.0 - alpha) * np.einsum("ti,tj->tij", Z, Z, optimize=True)
    Gbar = 0.5 * (G_obs.mean(axis=0) + G_obs.mean(axis=0).T)
    Gbar_sqrt = _spd_sqrt(Gbar, eps=1e-10)
    B_tilde_raw = batched_spd_exp(smat(np.asarray(fit.xs, float), d))
    Bbar_raw = 0.5 * (B_tilde_raw.mean(axis=0) + B_tilde_raw.mean(axis=0).T)
    Bbar_invsqrt = _spd_invsqrt_local(Bbar_raw, eps=1e-10)
    # Match the standalone OVK reconstruction: smooth in log-SPD space, then
    # normalize the smoothed SPD path to have sample mean identity before
    # returning it to the ridge-lift scale.
    B_tilde_hat = np.einsum("ij,tjk,kl->til", Bbar_invsqrt, B_tilde_raw, Bbar_invsqrt, optimize=True)
    G_hat = np.einsum("ij,tjk,kl->til", Gbar_sqrt, B_tilde_hat, Gbar_sqrt, optimize=True)
    Sigma_hat = (G_hat - alpha * eye[None, :, :]) / (1.0 - alpha)
    Sigma_hat = 0.5 * (Sigma_hat + np.swapaxes(Sigma_hat, 1, 2))
    recon = {
        "used_correct_unnormalization": True,
        "post_smoothing_log_spd_mean_normalization": True,
        "mean_abs_B_tilde_hat_minus_identity": float(np.mean(np.abs(B_tilde_hat.mean(axis=0) - eye))),
        "mean_abs_G_hat_bar_minus_Gbar": float(np.mean(np.abs(G_hat.mean(axis=0) - Gbar))),
    }
    return Sigma_hat, G_hat, recon


def _component_retained_scores(scores: dict[str, Any], headline: Any) -> dict[str, np.ndarray | float]:
    """Construct delta-method ratio IF components in the headline whitened basis."""
    M = np.asarray(scores["z_res"], float)
    X = np.asarray(scores["x_res"], float)
    Y = np.asarray(scores["Y_res"], float)
    Q_scores = np.asarray(scores["Q_scores"], float)
    gamma_hat = np.mean(M[:, None] * Y, axis=0)
    pi_hat = float(np.mean(M * X))
    if not np.isfinite(pi_hat) or abs(pi_hat) < 1e-12:
        raise ValueError(f"Cannot decompose retained IV scores with degenerate pi_hat={pi_hat}")
    theta_hat = gamma_hat / pi_hat
    psi_rf = (M[:, None] * Y - gamma_hat[None, :]) / pi_hat
    psi_fs = -(theta_hat[None, :] * (M * X - pi_hat)[:, None]) / pi_hat
    psi_iv_total_from_components = psi_rf + psi_fs
    psi_iv_total_direct = M[:, None] * (Y - X[:, None] * theta_hat[None, :]) / pi_hat
    if_error = psi_iv_total_from_components - psi_iv_total_direct
    max_abs_if_error = float(np.max(np.abs(if_error)))
    rel_fro_if_error = float(
        np.linalg.norm(if_error, "fro") / max(np.linalg.norm(psi_iv_total_direct, "fro"), 1e-12)
    )
    if max_abs_if_error >= 1e-8 or rel_fro_if_error >= 1e-8:
        raise ValueError(
            "Delta-method IV ratio influence components do not sum to the direct total: "
            f"max_abs_if_error={max_abs_if_error:.3g}, rel_fro_if_error={rel_fro_if_error:.3g}"
        )

    stored_if_error = Q_scores - psi_iv_total_direct
    max_abs_stored_if_error = float(np.max(np.abs(stored_if_error)))
    rel_fro_stored_if_error = float(
        np.linalg.norm(stored_if_error, "fro") / max(np.linalg.norm(psi_iv_total_direct, "fro"), 1e-12)
    )
    allow_mismatch = os.environ.get("OVK_IV_ALLOW_IF_MISMATCH", "0").lower() in {"1", "true", "yes", "on"}
    if (max_abs_stored_if_error >= 1e-8 or rel_fro_stored_if_error >= 1e-8) and not allow_mismatch:
        raise ValueError(
            "Existing stored IV influence score differs from the delta-method direct proxy-IV IF. "
            "Set OVK_IV_ALLOW_IF_MISMATCH=1 only for explicit diagnostic override. "
            f"max_abs_stored_if_error={max_abs_stored_if_error:.3g}, "
            f"rel_fro_stored_if_error={rel_fro_stored_if_error:.3g}, "
            f"stored_norm={np.linalg.norm(Q_scores, 'fro'):.6g}, direct_norm={np.linalg.norm(psi_iv_total_direct, 'fro'):.6g}"
        )

    R = int(headline.rank)
    if getattr(headline, "whitening_map", None) is not None:
        W_R = np.asarray(headline.whitening_map[:, :R], float).T
    else:
        V_R = np.asarray(headline.V[:, :R], float)
        lam_R = np.asarray(headline.eigvals[:R], float)
        W_R = np.diag(1.0 / np.sqrt(np.maximum(lam_R, 1e-12))) @ V_R.T
    z_iv_retained_direct = psi_iv_total_direct @ W_R.T
    z_rf_retained = psi_rf @ W_R.T
    z_fs_retained = psi_fs @ W_R.T
    z_iv_retained = np.asarray(headline.Z[:, :R], float)
    retained_direct_error = z_iv_retained - z_iv_retained_direct
    max_abs_retained_direct_error = float(np.max(np.abs(retained_direct_error)))
    rel_fro_retained_direct_error = float(
        np.linalg.norm(retained_direct_error, "fro") / max(np.linalg.norm(z_iv_retained, "fro"), 1e-12)
    )
    if (max_abs_retained_direct_error >= 1e-8 or rel_fro_retained_direct_error >= 1e-8) and not allow_mismatch:
        raise ValueError(
            "Headline retained IV score differs from delta-method direct IF whitened with the total-IV covariance map. "
            f"max_abs_retained_direct_error={max_abs_retained_direct_error:.3g}, "
            f"rel_fro_retained_direct_error={rel_fro_retained_direct_error:.3g}"
        )

    score_additive_error = z_iv_retained_direct - (z_rf_retained + z_fs_retained)
    max_abs_score_error = float(np.max(np.abs(score_additive_error)))
    rel_fro_score_error = float(
        np.linalg.norm(score_additive_error, "fro") / max(np.linalg.norm(z_iv_retained_direct, "fro"), 1e-12)
    )
    if max_abs_score_error >= 1e-8 or rel_fro_score_error >= 1e-8:
        raise ValueError(
            "Retained IV score additive identity failed: "
            f"max_abs_score_error={max_abs_score_error:.3g}, "
            f"rel_fro_score_error={rel_fro_score_error:.3g}"
        )
    return {
        "z_iv_retained": z_iv_retained_direct,
        "z_iv_retained_existing": z_iv_retained,
        "z_rf_retained": z_rf_retained,
        "z_fs_retained": z_fs_retained,
        "z_res_retained": z_rf_retained,
        "score_additive_error": score_additive_error,
        "gamma_hat": gamma_hat,
        "pi_hat": pi_hat,
        "theta_hat": theta_hat,
        "psi_rf": psi_rf,
        "psi_fs": psi_fs,
        "psi_iv_total_direct": psi_iv_total_direct,
        "psi_iv_total_from_components": psi_iv_total_from_components,
        "W_R": W_R,
        "max_abs_if_error": max_abs_if_error,
        "rel_fro_if_error": rel_fro_if_error,
        "max_abs_stored_if_error": max_abs_stored_if_error,
        "rel_fro_stored_if_error": rel_fro_stored_if_error,
        "max_abs_retained_direct_error": max_abs_retained_direct_error,
        "rel_fro_retained_direct_error": rel_fro_retained_direct_error,
        "max_abs_score_error": max_abs_score_error,
        "rel_fro_score_error": rel_fro_score_error,
        "max_abs_surface_error": max_abs_stored_if_error,
        "rel_fro_surface_error": rel_fro_stored_if_error,
    }


def plot_iv_tau_exact_decomposition_area(df: pd.DataFrame) -> tuple[Path, Path]:
    dates = pd.to_datetime(df["date"])
    components = [
        ("tau_rf", "Reduced-form/numerator trace contribution", "#16a34a", 0.62),
        ("tau_fs", "First-stage trace contribution", "#2563eb", 0.66),
        ("tau_cross", "Cross trace contribution", "#f97316", 0.64),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    pos_base = np.zeros(len(df))
    neg_base = np.zeros(len(df))
    for col, label, color, alpha in components:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        pos = np.where(values > 0, values, 0.0)
        neg = np.where(values < 0, values, 0.0)
        ax.fill_between(dates, pos_base, pos_base + pos, color=color, alpha=alpha, linewidth=0, label=label)
        ax.fill_between(dates, neg_base, neg_base + neg, color=color, alpha=alpha, linewidth=0)
        pos_base += pos
        neg_base += neg
    ax.plot(dates, df["tau_total_decomp"], color="black", linewidth=1.45, label="Augmented total tau")
    if "tau_existing_iv" in df.columns and pd.to_numeric(df["tau_existing_iv"], errors="coerce").notna().any():
        ax.plot(
            dates,
            df["tau_existing_iv"],
            color="#6b7280",
            linestyle="--",
            linewidth=1.15,
            label="Existing standalone tau",
        )
    ax.axhline(1.0, color="#111827", linewidth=0.8, alpha=0.65)
    ax.axhline(0.0, color="#111827", linewidth=0.7, alpha=0.45)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylabel("Trace contribution to tau")
    ax.set_title("Exact trace decomposition of proxy-IV OVK amplification")
    ax.legend(loc="upper left", fontsize=8.5, ncol=1, frameon=True)
    fig.tight_layout()
    png = FIGURES / "iv_tau_exact_decomposition_area.png"
    pdf = FIGURES / "iv_tau_exact_decomposition_area.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_iv_tau_exact_decomposition_excess_area(df: pd.DataFrame) -> tuple[Path, Path]:
    dates = pd.to_datetime(df["date"])
    components = [
        ("tau_rf_excess", "Reduced-form/numerator mean-centered contribution", "#16a34a", 0.62),
        ("tau_fs_excess", "First-stage mean-centered contribution", "#2563eb", 0.66),
        ("tau_cross_excess", "Cross mean-centered contribution", "#f97316", 0.64),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    pos_base = np.zeros(len(df))
    neg_base = np.zeros(len(df))
    for col, label, color, alpha in components:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        pos = np.where(values > 0, values, 0.0)
        neg = np.where(values < 0, values, 0.0)
        ax.fill_between(dates, pos_base, pos_base + pos, color=color, alpha=alpha, linewidth=0, label=label)
        ax.fill_between(dates, neg_base, neg_base + neg, color=color, alpha=alpha, linewidth=0)
        pos_base += pos
        neg_base += neg
    ax.plot(
        dates,
        df["tau_component_excess_sum"],
        color="black",
        linewidth=1.45,
        label="Component sum relative to sample mean",
    )
    if "tau_existing_excess" in df.columns and pd.to_numeric(df["tau_existing_excess"], errors="coerce").notna().any():
        ax.plot(
            dates,
            df["tau_existing_excess"],
            color="#6b7280",
            linestyle="--",
            linewidth=1.15,
            label="Existing standalone tau minus 1",
        )
    ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylabel("Mean-centered trace contribution")
    ax.set_title("Mean-centered diagnostic decomposition of proxy-IV OVK amplification")
    ax.legend(loc="upper left", fontsize=8.5, ncol=1, frameon=True)
    fig.tight_layout()
    png = FIGURES / "iv_tau_exact_decomposition_excess_area.png"
    pdf = FIGURES / "iv_tau_exact_decomposition_excess_area.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_iv_tau_exact_decomposition_caption() -> Path:
    caption = r"""The figure reports an exact trace decomposition of the rank-five proxy-IV OVK amplification index implied by the augmented state-space covariance model. The augmented model is estimated for the stacked retained delta-method ratio score \(x_t=(z^{RF\prime}_t,z^{FS\prime}_t)'\), where the whitening map is built from the total proxy-IV influence-function covariance. The reduced-form/numerator, first-stage/denominator, and cross components are \(R^{-1}\operatorname{tr}(\widehat\Sigma_{RR,t})\), \(R^{-1}\operatorname{tr}(\widehat\Sigma_{FF,t})\), and \(R^{-1}\operatorname{tr}(\widehat\Sigma_{RF,t}+\widehat\Sigma_{FR,t})\), respectively. These components sum exactly to the augmented model's implied \(\widehat\tau_t\). The dashed line, when shown, is the standalone proxy-IV \(\widehat\tau_t\) estimated from the original rank-\(R\) state-space model; it is included as a comparison rather than as an algebraic constraint. The cross term can be negative, indicating dates at which numerator and first-stage components offset each other in the retained OVK directions.
"""
    path = FIGURES / "iv_tau_exact_decomposition_area_caption.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption, encoding="utf-8")
    return path


def run_iv_tau_exact_decomposition(scores: dict[str, Any], headline: Any, em_iters: int = EM_ITERS) -> dict[str, Any]:
    """Estimate the full-sample exact augmented tau trace decomposition."""
    R = int(headline.rank)
    dates = pd.to_datetime(scores["dates"]).reset_index(drop=True)
    components = _component_retained_scores(scores, headline)
    x_aug = np.concatenate(
        [
            np.asarray(components["z_rf_retained"], float),
            np.asarray(components["z_fs_retained"], float),
        ],
        axis=1,
    )
    alpha = np.nan
    if getattr(headline, "estimator_mode", "log_spd_legacy") == "arithmetic_outer_product":
        _, y_aug = arithmetic_outer_product_observations(x_aug)
        fit_aug = fit_em_state_space(
            y_aug,
            rank=2 * R,
            em_iters=em_iters,
            nu=ROBUST_NU,
            estimator_mode="arithmetic_outer_product",
            observation_kind="arithmetic_outer_product_augmented",
        )
        Sigma_aug = arithmetic_matrix_series_from_state(fit_aug.xs, 2 * R)
        G_aug = Sigma_aug
        reconstruction_diagnostics = {
            "estimator_mode": "arithmetic_outer_product",
            "used_correct_unnormalization": True,
            "post_smoothing_log_spd_mean_normalization": False,
            "note": "Augmented RF/FS decomposition smoothed direct arithmetic outer products in headline retained coordinates.",
        }
    else:
        alpha = float(headline.fit.alpha)
        fit_aug = estimate_alpha_and_state(
            x_aug,
            rank=2 * R,
            em_iters=em_iters,
            nu=ROBUST_NU,
            alpha_grid=np.asarray([alpha], dtype=float),
        )
        Sigma_aug, G_aug, reconstruction_diagnostics = _recover_unlifted_covariance_from_state(x_aug, fit_aug)
    S = np.concatenate([np.eye(R), np.eye(R)], axis=1)
    Sigma_x_raw = (x_aug.T @ x_aug) / max(x_aug.shape[0], 1)
    A_raw_implied = S @ Sigma_x_raw @ S.T
    raw_aug_tau_bar = float(np.trace(A_raw_implied) / R)
    if getattr(headline, "estimator_mode", "log_spd_legacy") == "arithmetic_outer_product":
        current_aug_tau_bar = float(np.mean(np.trace(S @ Sigma_aug @ S.T, axis1=1, axis2=2) / R))
        if np.isfinite(current_aug_tau_bar) and current_aug_tau_bar > 1e-12:
            Sigma_aug = Sigma_aug * (raw_aug_tau_bar / current_aug_tau_bar)
            G_aug = Sigma_aug
            reconstruction_diagnostics["arithmetic_scalar_reference_normalization"] = raw_aug_tau_bar / current_aug_tau_bar
    trace = trace_decomposition_from_augmented_covariance(Sigma_aug, R)
    tau_sum_error = trace["tau_total_decomp"] - trace["tau_aug_implied"]
    max_abs_tau_sum_error = float(np.max(np.abs(tau_sum_error)))
    mean_abs_tau_sum_error = float(np.mean(np.abs(tau_sum_error)))
    if max_abs_tau_sum_error >= 1e-7:
        raise ValueError(f"Augmented tau trace decomposition failed: max_abs_tau_sum_error={max_abs_tau_sum_error:.3g}")

    z_iv_retained = np.asarray(components["z_iv_retained"], float)
    z_sum = np.asarray(components["z_rf_retained"], float) + np.asarray(components["z_fs_retained"], float)
    raw_tau_existing_from_scores = np.sum(z_iv_retained**2, axis=1) / R
    raw_tau_sum_from_components = np.sum(z_sum**2, axis=1) / R
    raw_tau_existing_mean = float(np.mean(raw_tau_existing_from_scores))
    raw_tau_sum_mean = float(np.mean(raw_tau_sum_from_components))
    raw_tol = 5e-8
    if abs(raw_tau_existing_mean - 1.0) > raw_tol or abs(raw_tau_sum_mean - 1.0) > raw_tol or abs(raw_aug_tau_bar - 1.0) > raw_tol:
        raise ValueError(
            "Raw retained-score tau normalization failed: "
            f"raw_tau_existing_mean={raw_tau_existing_mean:.12g}, "
            f"raw_tau_sum_mean={raw_tau_sum_mean:.12g}, raw_aug_tau_bar={raw_aug_tau_bar:.12g}"
        )

    existing_tau = np.asarray(getattr(headline, "tau", np.full(len(dates), np.nan)), float)
    if existing_tau.shape[0] == len(dates) and np.isfinite(existing_tau).any():
        tau_existing_diff = trace["tau_total_decomp"] - existing_tau
        tau_existing_excess = existing_tau - 1.0
        mean_existing_tau = float(np.nanmean(existing_tau))
        mean_augmented_total_tau = float(np.nanmean(trace["tau_total_decomp"]))
        mean_augmented_total_tau_plus_one = float(np.nanmean(trace["tau_total_decomp"] + 1.0))
        corr_existing_augmented_total = float(np.corrcoef(existing_tau, trace["tau_total_decomp"])[0, 1])
        corr_existing_augmented_total_plus_one = float(np.corrcoef(existing_tau, trace["tau_total_decomp"] + 1.0)[0, 1])
        rmse_existing_augmented_total = float(np.sqrt(np.nanmean((existing_tau - trace["tau_total_decomp"]) ** 2)))
        rmse_existing_augmented_total_plus_one = float(np.sqrt(np.nanmean((existing_tau - (trace["tau_total_decomp"] + 1.0)) ** 2)))
        tau_existing_max_abs_diff = float(np.max(np.abs(tau_existing_diff)))
    else:
        existing_tau = np.full(len(dates), np.nan)
        tau_existing_diff = np.full(len(dates), np.nan)
        tau_existing_excess = np.full(len(dates), np.nan)
        mean_existing_tau = np.nan
        mean_augmented_total_tau = float(np.nanmean(trace["tau_total_decomp"]))
        mean_augmented_total_tau_plus_one = float(np.nanmean(trace["tau_total_decomp"] + 1.0))
        corr_existing_augmented_total = np.nan
        corr_existing_augmented_total_plus_one = np.nan
        rmse_existing_augmented_total = np.nan
        rmse_existing_augmented_total_plus_one = np.nan
        tau_existing_max_abs_diff = np.nan

    row_error_norm = np.linalg.norm(np.asarray(components["score_additive_error"], float), axis=1)
    tau_rf_excess = trace["tau_rf"] - np.mean(trace["tau_rf"])
    tau_fs_excess = trace["tau_fs"] - np.mean(trace["tau_fs"])
    tau_cross_excess = trace["tau_cross"] - np.mean(trace["tau_cross"])
    tau_component_excess_sum = tau_rf_excess + tau_fs_excess + tau_cross_excess
    tau_total_excess = trace["tau_total_decomp"] - 1.0
    df = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "tau_rf": trace["tau_rf"],
            "tau_fs": trace["tau_fs"],
            "tau_cross": trace["tau_cross"],
            "tau_total_dec": trace["tau_total_dec"],
            "tau_total_decomp": trace["tau_total_decomp"],
            "tau_aug_implied": trace["tau_aug_implied"],
            "tau_sum_error": tau_sum_error,
            "standalone_tau_if_available": existing_tau,
            "tau_existing_iv": existing_tau,
            "tau_existing_diff": tau_existing_diff,
            "raw_tau_existing_from_scores": raw_tau_existing_from_scores,
            "raw_tau_sum_from_components": raw_tau_sum_from_components,
            "max_abs_if_error": float(components["max_abs_if_error"]),
            "rel_fro_if_error": float(components["rel_fro_if_error"]),
            "max_abs_stored_if_error": float(components["max_abs_stored_if_error"]),
            "rel_fro_stored_if_error": float(components["rel_fro_stored_if_error"]),
            "max_abs_retained_direct_error": float(components["max_abs_retained_direct_error"]),
            "rel_fro_retained_direct_error": float(components["rel_fro_retained_direct_error"]),
            "score_additive_error_norm": row_error_norm,
            "tau_rf_excess": tau_rf_excess,
            "tau_fs_excess": tau_fs_excess,
            "tau_cross_excess": tau_cross_excess,
            "tau_component_excess_sum": tau_component_excess_sum,
            "tau_total_excess": tau_total_excess,
            "tau_existing_excess": tau_existing_excess,
        }
    )
    table_path = TABLES / "iv_tau_exact_decomposition_timeseries.csv"
    df.to_csv(table_path, index=False)
    figure_png, figure_pdf = plot_iv_tau_exact_decomposition_area(df)
    excess_figure_png, excess_figure_pdf = plot_iv_tau_exact_decomposition_excess_area(df)
    caption_path = write_iv_tau_exact_decomposition_caption()
    diagnostics = {
        "R": R,
        "augmented_dimension": int(x_aug.shape[1]),
        "alpha": alpha,
        "n_obs": int(len(dates)),
        "max_abs_surface_error": float(components["max_abs_surface_error"]),
        "rel_fro_surface_error": float(components["rel_fro_surface_error"]),
        "max_abs_if_error": float(components["max_abs_if_error"]),
        "rel_fro_if_error": float(components["rel_fro_if_error"]),
        "max_abs_stored_if_error": float(components["max_abs_stored_if_error"]),
        "rel_fro_stored_if_error": float(components["rel_fro_stored_if_error"]),
        "max_abs_retained_direct_error": float(components["max_abs_retained_direct_error"]),
        "rel_fro_retained_direct_error": float(components["rel_fro_retained_direct_error"]),
        "max_abs_score_error": float(components["max_abs_score_error"]),
        "rel_fro_score_error": float(components["rel_fro_score_error"]),
        "pi_hat": float(components["pi_hat"]),
        "raw_tau_existing_mean": raw_tau_existing_mean,
        "raw_tau_sum_mean": raw_tau_sum_mean,
        "raw_tau_existing_min": float(np.min(raw_tau_existing_from_scores)),
        "raw_tau_existing_max": float(np.max(raw_tau_existing_from_scores)),
        "raw_tau_sum_min": float(np.min(raw_tau_sum_from_components)),
        "raw_tau_sum_max": float(np.max(raw_tau_sum_from_components)),
        "raw_aug_tau_bar": raw_aug_tau_bar,
        "mean_existing_tau": mean_existing_tau,
        "mean_augmented_total_tau": mean_augmented_total_tau,
        "mean_augmented_total_tau_plus_one": mean_augmented_total_tau_plus_one,
        "corr_existing_augmented_total": corr_existing_augmented_total,
        "corr_existing_augmented_total_plus_one": corr_existing_augmented_total_plus_one,
        "rmse_existing_augmented_total": rmse_existing_augmented_total,
        "rmse_existing_augmented_total_plus_one": rmse_existing_augmented_total_plus_one,
        "max_abs_tau_sum_error": max_abs_tau_sum_error,
        "mean_abs_tau_sum_error": mean_abs_tau_sum_error,
        "tau_existing_corr": corr_existing_augmented_total,
        "tau_existing_rmse": rmse_existing_augmented_total,
        "tau_existing_max_abs_diff": tau_existing_max_abs_diff,
            "used_correct_unnormalization": bool(reconstruction_diagnostics.get("used_correct_unnormalization", False)),
            **reconstruction_diagnostics,
        "augmented_state_dim": int(fit_aug.xs.shape[1]),
        "augmented_robust_loglik": float(fit_aug.robust_loglik),
        "augmented_factor_log_score": float(fit_aug.factor_log_score),
        "min_eigenvalue_G_aug_hat": float(np.min(np.linalg.eigvalsh(G_aug))),
        "sigma_projection": "none; decomposition identities are computed from the final symmetrized unlifted covariance matrices",
        "score_component_definition": (
            "Delta-method just-identified proxy-IV ratio IF: psi_rf=(M*Y-gamma_hat)/pi_hat, "
            "psi_fs=-theta_hat*(M*X-pi_hat)/pi_hat, and psi_total=M*(Y-X*theta_hat)/pi_hat. "
            "The whitening map W_R is built from the total direct IV IF covariance and applied to all components."
        ),
        "notes": (
                "The decomposed total is the tau path implied by the augmented state-space model and is not forced "
                "to equal the standalone rank-R IV tau path estimated from a separate model. "
                "In arithmetic_outer_product mode the augmented path is smoothed directly from retained RF/FS outer products; "
                "in log_spd_legacy mode the smoothed augmented SPD path is normalized before Gbar unnormalization."
            ),
        "table_csv": str(table_path),
        "figure_png": str(figure_png),
        "figure_pdf": str(figure_pdf),
        "excess_figure_png": str(excess_figure_png),
        "excess_figure_pdf": str(excess_figure_pdf),
        "caption_tex": str(caption_path),
    }
    diagnostics_path = TABLES / "iv_tau_exact_decomposition_diagnostics.json"
    diagnostics["diagnostics_json"] = str(diagnostics_path)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    return {
        "path": df,
        "diagnostics": diagnostics,
        "diagnostics_df": pd.DataFrame([diagnostics]),
        "table_path": table_path,
        "diagnostics_path": diagnostics_path,
        "figure_png": figure_png,
        "figure_pdf": figure_pdf,
        "excess_figure_png": excess_figure_png,
        "excess_figure_pdf": excess_figure_pdf,
        "caption_path": caption_path,
        "fit": fit_aug,
    }


def _print_exact_decomposition_summary(diagnostics: dict[str, Any]) -> None:
    print("Exact IV tau decomposition debug/fix complete.")
    print(f"Max additive score error: {float(diagnostics['max_abs_score_error']):.6g}")
    print(f"Relative Frobenius score error: {float(diagnostics['rel_fro_score_error']):.6g}")
    print(f"Raw existing score tau mean: {float(diagnostics['raw_tau_existing_mean']):.6g}")
    print(f"Raw augmented implied tau mean: {float(diagnostics['raw_aug_tau_bar']):.6g}")
    print(f"Corrected augmented total tau mean: {float(diagnostics['mean_augmented_total_tau']):.6g}")
    if np.isfinite(float(diagnostics.get("mean_existing_tau", np.nan))):
        print(f"Existing standalone tau mean: {float(diagnostics['mean_existing_tau']):.6g}")
    print(f"Max tau sum error: {float(diagnostics['max_abs_tau_sum_error']):.6g}")
    if np.isfinite(float(diagnostics.get("corr_existing_augmented_total", np.nan))):
        print(f"Correlation with existing standalone tau: {float(diagnostics['corr_existing_augmented_total']):.6g}")
    print(f"Saved corrected table: {diagnostics['table_csv']}")
    print(f"Saved diagnostics: {diagnostics['diagnostics_json']}")
    print(f"Saved total-level figure: {diagnostics['figure_png']}")
    print(f"Saved excess diagnostic figure: {diagnostics['excess_figure_png']}")


def debug_iv_tau_exact_decomposition(scores: dict[str, Any], headline: Any, em_iters: int = EM_ITERS) -> dict[str, Any]:
    """Run the exact decomposition with verbose diagnostics and saved artifacts."""
    out = run_iv_tau_exact_decomposition(scores, headline, em_iters=em_iters)
    _print_exact_decomposition_summary(out["diagnostics"])
    return out


def run_iv_tau_multiplicative_driver_diagnostic(scores: dict[str, Any], headline: Any) -> dict[str, Any]:
    """Write the proxy-IV tau multiplicative-driver diagnostic.

    The full-coordinate proxy-IV score is chi_proxy[t]=(M_t/kappa_hat)u_proxy[t].
    Since the full-coordinate temporal kernel is linear in score outer products,
    the period-level factors below exactly decompose the fitted tau_soft path.
    """
    dates = pd.to_datetime(scores["dates"]).reset_index(drop=True)
    M = np.asarray(scores["z_res"], float)
    X = np.asarray(scores["x_res"], float)
    Y = np.asarray(scores["Y_res"], float)
    gamma_hat = np.mean(M[:, None] * Y, axis=0)
    kappa_hat = float(np.mean(M * X))
    if not np.isfinite(kappa_hat) or abs(kappa_hat) < 1e-12:
        raise ValueError(f"Cannot build multiplicative driver diagnostic with degenerate kappa_hat={kappa_hat}")
    theta_hat = gamma_hat / kappa_hat
    u = Y - X[:, None] * theta_hat[None, :]

    chi_proxy = (M / kappa_hat)[:, None] * u
    full_iv = estimate_full_coordinate_kernel_model(
        chi_proxy,
        dates,
        variant=SPEC.key,
        label=SPEC.label,
        outcome_labels=list(scores["outcome_labels"]),
    )
    np.savez_compressed(
        TABLES / "iv_full_coordinate_covariance_components.npz",
        chi_proxy=chi_proxy,
        C_hat=full_iv.C_hat,
        D_rho=full_iv.D_rho,
        K_hat=full_iv.K_hat,
        A_hat=full_iv.A_hat,
        tau_soft=full_iv.tau_soft,
        temporal_weights=full_iv.temporal_weights,
        proxy_M=M,
        proxied_X=X,
        residual_u=u,
        kappa_hat=np.asarray([kappa_hat]),
        rho=np.asarray([full_iv.rho]),
        d_rho=np.asarray([full_iv.d_rho]),
        kernel_eta=np.asarray([full_iv.kernel_eta]),
    )
    cD, lowerD = cho_factor(full_iv.D_rho, lower=True, check_finite=False)
    residual_solved = cho_solve((cD, lowerD), u.T, check_finite=False).T
    score_solved = cho_solve((cD, lowerD), chi_proxy.T, check_finite=False).T
    residual_energy_soft = np.einsum("ti,ti->t", u, residual_solved, optimize=True) / full_iv.d_rho
    score_energy_soft = np.einsum("ti,ti->t", chi_proxy, score_solved, optimize=True) / full_iv.d_rho
    proxy_exposure_a2 = (M / kappa_hat) ** 2
    multiplicative_energy_product = proxy_exposure_a2 * residual_energy_soft
    product_error = score_energy_soft - multiplicative_energy_product
    max_abs_product_error = float(np.max(np.abs(product_error)))
    max_rel_product_error = float(np.max(np.abs(product_error) / np.maximum(np.abs(score_energy_soft), IV_FACTOR_ATOL)))
    rel_fro_product_error = float(
        np.linalg.norm(product_error) / max(np.linalg.norm(score_energy_soft), 1e-12)
    )
    if not np.allclose(score_energy_soft, multiplicative_energy_product, atol=IV_FACTOR_ATOL, rtol=IV_FACTOR_RTOL):
        raise ValueError(
            "Full-coordinate ridge-whitened energy multiplicative identity failed: "
            f"max_abs_product_error={max_abs_product_error:.3g}, max_rel_product_error={max_rel_product_error:.3g}, "
            f"rel_fro_product_error={rel_fro_product_error:.3g}"
        )

    tau = np.asarray(full_iv.tau_soft, float)
    W = np.asarray(full_iv.temporal_weights, float)
    row_sums = W.sum(axis=1)
    row_sum_error = row_sums - 1.0
    max_abs_row_sum_error = float(np.max(np.abs(row_sum_error)))
    if max_abs_row_sum_error > max(IV_FACTOR_ATOL, IV_FACTOR_RTOL):
        raise ValueError(f"Full-coordinate temporal weights do not sum to one: max_abs_row_sum_error={max_abs_row_sum_error:.3g}")
    negative_weight_count = int(np.sum(W < -IV_FACTOR_ATOL))
    any_negative_weights = bool(negative_weight_count > 0)
    maximum_weight = np.max(W, axis=1)
    weight_square_sum = np.sum(W * W, axis=1)
    effective_weight_count = np.where(weight_square_sum > 0.0, row_sums * row_sums / weight_square_sum, 0.0)

    tau_reconstructed = W @ multiplicative_energy_product
    reconstruction_error = tau - tau_reconstructed
    reconstruction_abs_error = np.abs(reconstruction_error)
    reconstruction_rel_error = reconstruction_abs_error / np.maximum(np.abs(tau), IV_FACTOR_ATOL)
    max_abs_reconstruction_error = float(np.max(reconstruction_abs_error))
    max_rel_reconstruction_error = float(np.max(reconstruction_rel_error))
    if not np.allclose(tau, tau_reconstructed, atol=IV_FACTOR_ATOL, rtol=IV_FACTOR_RTOL):
        raise ValueError(
            "Full-coordinate tau reconstruction from temporal weights failed: "
            f"max_abs={max_abs_reconstruction_error:.3g}, max_rel={max_rel_reconstruction_error:.3g}"
        )

    E_bar = W @ proxy_exposure_a2
    numerator = tau_reconstructed
    R_tilde = np.zeros_like(numerator)
    nonzero_E = E_bar > IV_FACTOR_ZERO_TOL
    if np.any(~nonzero_E & (np.abs(numerator) > IV_FACTOR_ZERO_TOL)):
        bad = int(np.flatnonzero(~nonzero_E & (np.abs(numerator) > IV_FACTOR_ZERO_TOL))[0])
        raise ValueError(
            "Smoothed exposure factor is numerically zero while tau numerator is not: "
            f"index={bad}, E_bar={E_bar[bad]:.3g}, numerator={numerator[bad]:.3g}"
        )
    R_tilde[nonzero_E] = numerator[nonzero_E] / E_bar[nonzero_E]
    E_ref = float(np.mean(E_bar[nonzero_E])) if np.any(nonzero_E) else float("nan")
    if not np.isfinite(E_ref) or E_ref <= 0.0:
        raise ValueError(f"Exposure reference scale must be positive and finite; got E_ref={E_ref}.")
    exposure_factor = E_bar / E_ref
    residual_factor = E_ref * R_tilde
    factor_product = exposure_factor * residual_factor
    factor_product_error = tau - factor_product
    factor_product_abs_error = np.abs(factor_product_error)
    factor_product_rel_error = factor_product_abs_error / np.maximum(np.abs(tau), IV_FACTOR_ATOL)
    max_abs_factor_product_error = float(np.max(factor_product_abs_error))
    max_rel_factor_product_error = float(np.max(factor_product_rel_error))
    if not np.allclose(tau, factor_product, atol=IV_FACTOR_ATOL, rtol=IV_FACTOR_RTOL):
        raise ValueError(
            "Period-level exposure/residual factor product failed: "
            f"max_abs={max_abs_factor_product_error:.3g}, max_rel={max_rel_factor_product_error:.3g}"
        )

    proxy_pct = _percentile_rank(np.log(np.maximum(exposure_factor, 1e-300)))
    residual_pct = _percentile_rank(np.log(np.maximum(residual_factor, 1e-300)))
    product_pct = _percentile_rank(np.log(np.maximum(factor_product, 1e-300)))
    tau_pct = _percentile_rank(tau)
    high_tau_threshold = float(HIGH_TAU_QUANTILE)
    driver_threshold = float(DRIVER_FACTOR_QUANTILE)
    high_tau_90 = tau_pct >= 0.90
    high_tau_95 = tau_pct >= 0.95

    def classify(p_proxy: float, p_resid: float) -> str:
        if p_proxy >= driver_threshold and p_resid >= driver_threshold:
            return "mixed"
        if p_proxy >= driver_threshold and p_resid < driver_threshold:
            return "proxy-exposure"
        if p_proxy < driver_threshold and p_resid >= driver_threshold:
            return "residual-energy"
        return "unclassified"

    driver_label = np.asarray([classify(float(pa), float(pr)) for pa, pr in zip(proxy_pct, residual_pct)], dtype=object)
    high_tau_main = tau_pct >= high_tau_threshold
    high_tau_threshold_value = float(np.nanmin(tau[high_tau_main])) if bool(np.any(high_tau_main)) else float("nan")
    rel_product_error_global = rel_fro_product_error
    monthly = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "tau_iv": tau,
            "p_tau": tau_pct,
            "proxy_exposure": exposure_factor,
            "p_proxy": proxy_pct,
            "resid_energy": residual_factor,
            "p_resid": residual_pct,
            "instant_product_energy": score_energy_soft,
            "p_product": product_pct,
            "high_tau": high_tau_main,
            "high_tau_90": high_tau_90,
            "high_tau_95": high_tau_95,
            "high_tau_quantile": high_tau_threshold,
            "high_tau_threshold": high_tau_threshold_value,
            "driver_threshold": driver_threshold,
            "driver_label": driver_label,
            "instant_score_energy": score_energy_soft,
            "rel_product_error": rel_product_error_global,
            "tau_soft": tau,
            "tau_soft_existing": tau,
            "tau_soft_reconstructed": tau_reconstructed,
            "factor_product": factor_product,
            "reconstruction_abs_error": reconstruction_abs_error,
            "reconstruction_rel_error": reconstruction_rel_error,
            "factor_product_abs_error": factor_product_abs_error,
            "factor_product_rel_error": factor_product_rel_error,
            "E_raw_smoothed": E_bar,
            "R_exposure_weighted_raw": R_tilde,
            "exposure_factor": exposure_factor,
            "residual_factor": residual_factor,
            "source_proxy_exposure_E": proxy_exposure_a2,
            "source_residual_energy_R": residual_energy_soft,
            "source_score_energy_ER": score_energy_soft,
            "weight_row_sum": row_sums,
            "effective_weight_count": effective_weight_count,
            "maximum_weight": maximum_weight,
            "residual_energy_soft": residual_energy_soft,
            "score_energy_soft": score_energy_soft,
            "ridge_rho": full_iv.rho,
            "d_rho": full_iv.d_rho,
            "kernel_eta": full_iv.kernel_eta,
            # Back-compatible aliases used by earlier diagnostics.
            "tau": tau,
            "proxy_exposure_a2": proxy_exposure_a2,
            "retained_residual_energy_s": residual_energy_soft,
            "raw_retained_score_energy": score_energy_soft,
            "full_coordinate_residual_energy_soft_percentile": residual_pct,
            "multiplicative_energy_product": multiplicative_energy_product,
            "multiplicative_energy_product_error": product_error,
            "proxy_exposure_percentile": proxy_pct,
            "retained_residual_energy_percentile": residual_pct,
        }
    )
    monthly_path = TABLES / "iv_tau_driver_diagnostic_monthly.csv"
    monthly.to_csv(monthly_path, index=False)
    legacy_table_path = TABLES / "iv_tau_multiplicative_driver_timeseries.csv"
    monthly.to_csv(legacy_table_path, index=False)
    audit_path = TABLES / "iv_tau_factor_decomposition_audit.csv"
    audit_cols = [
        "date",
        "tau_soft_existing",
        "tau_soft_reconstructed",
        "E_raw_smoothed",
        "R_exposure_weighted_raw",
        "exposure_factor",
        "residual_factor",
        "factor_product",
        "reconstruction_abs_error",
        "high_tau",
        "high_tau_threshold",
        "weight_row_sum",
        "effective_weight_count",
        "maximum_weight",
    ]
    monthly[audit_cols].to_csv(audit_path, index=False)

    order = np.argsort(tau)[::-1]
    top_idx = order[: min(20, len(order))]
    top_months = pd.DataFrame(
        {
            "rank": np.arange(1, len(top_idx) + 1),
            "date": dates.iloc[top_idx].dt.strftime("%Y-%m-%d").to_numpy(),
            "tau_iv": tau[top_idx],
            "p_tau": tau_pct[top_idx],
            "p_proxy": proxy_pct[top_idx],
            "p_resid": residual_pct[top_idx],
            "p_product": product_pct[top_idx],
            "exposure_factor": exposure_factor[top_idx],
            "residual_factor": residual_factor[top_idx],
            "factor_product": factor_product[top_idx],
            "driver_label": driver_label[top_idx],
        }
    )
    top_months_path = TABLES / "iv_tau_driver_top_months.csv"
    top_months.to_csv(top_months_path, index=False)

    episode_rows: list[dict[str, Any]] = []
    episode_id = 0
    start: int | None = None
    flags = np.asarray(high_tau_90, bool)
    for i, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            end = i - 1
            idx = np.arange(start, end + 1)
            episode_id += 1
            local_max_pos = int(idx[np.argmax(tau[idx])])
            labels = pd.Series(driver_label[idx])
            counts = labels.value_counts()
            tied = set(counts[counts.eq(counts.max())].index)
            label_at_max = str(driver_label[local_max_pos])
            dominant = label_at_max if label_at_max in tied else str(counts.index[0])
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "start_date": dates.iloc[start].strftime("%Y-%m-%d"),
                    "end_date": dates.iloc[end].strftime("%Y-%m-%d"),
                    "n_months": int(len(idx)),
                    "max_tau": float(tau[local_max_pos]),
                    "date_of_max_tau": dates.iloc[local_max_pos].strftime("%Y-%m-%d"),
                    "mean_tau": float(np.mean(tau[idx])),
                    "median_p_proxy": float(np.median(proxy_pct[idx])),
                    "median_p_resid": float(np.median(residual_pct[idx])),
                    "median_p_product": float(np.median(product_pct[idx])),
                    "driver_label_at_max_tau": label_at_max,
                    "dominant_driver_label": dominant,
                }
            )
            start = None
    episodes = pd.DataFrame(episode_rows)
    episodes_path = TABLES / "iv_tau_driver_high_tau_episodes.csv"
    episodes.to_csv(episodes_path, index=False)

    colors = {
        "proxy-exposure": "#2563eb",
        "residual-energy": "#16a34a",
        "mixed": "#7e22ce",
        "unclassified": "#6b7280",
    }
    month_number = dates.dt.year.to_numpy(int) * 12 + dates.dt.month.to_numpy(int)

    def spaced_top_indices(order_values: np.ndarray, max_labels: int = 6, min_month_gap: int = 6) -> list[int]:
        chosen: list[int] = []
        for idx_value in order_values:
            idx_int = int(idx_value)
            if all(abs(int(month_number[idx_int]) - int(month_number[j])) >= min_month_gap for j in chosen):
                chosen.append(idx_int)
            if len(chosen) >= max_labels:
                break
        return chosen

    label_idx = spaced_top_indices(order, max_labels=6, min_month_gap=6)
    fig, (ax_ts, ax_sc) = plt.subplots(2, 1, figsize=(10.8, 8.2), gridspec_kw={"height_ratios": [1.0, 1.05]})
    ax_ts.plot(dates, tau, color="black", linewidth=1.55, label="Full-coordinate tau_soft")
    ax_ts.plot(dates, exposure_factor, color="#2563eb", linewidth=1.25, alpha=0.9, label="Smoothed proxy exposure factor")
    ax_ts.plot(dates, residual_factor, color="#16a34a", linewidth=1.25, alpha=0.9, label="Exposure-weighted residual-energy factor")
    ax_ts.axhline(1.0, color="#111827", linewidth=0.8, alpha=0.65)
    for label, color in colors.items():
        mask = high_tau_main & (driver_label == label)
        if mask.any():
            ax_ts.scatter(dates[mask], tau[mask], s=28, color=color, edgecolor="white", linewidth=0.4, label=label)
    for j, idx in enumerate(label_idx):
        ax_ts.annotate(
            dates.iloc[idx].strftime("%Y-%m"),
            (dates.iloc[idx], tau[idx]),
            xytext=(4, 8 + (j % 2) * 9),
            textcoords="offset points",
            fontsize=6.5,
            color="#111827",
        )
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax_ts.xaxis.set_major_locator(locator)
    ax_ts.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plotted_lines = [tau, exposure_factor, residual_factor]
    if all(np.all(np.asarray(line, float) > 0.0) for line in plotted_lines):
        ax_ts.set_yscale("log")
    ax_ts.set_ylabel("Dimensionless factor / tau_soft")
    ax_ts.set_title("Proxy-IV full-coordinate ridge-soft amplification and exact multiplicative factors")
    ax_ts.legend(loc="upper left", fontsize=8.0, frameon=True, ncol=3)

    tau_span = max(float(np.nanmax(tau) - np.nanmin(tau)), 1e-12)
    sizes = 20.0 + 80.0 * (tau - float(np.nanmin(tau))) / tau_span
    ax_sc.scatter(proxy_pct, residual_pct, s=18, color="#d1d5db", alpha=0.55, label="all months")
    for label, color in colors.items():
        mask = high_tau_main & (driver_label == label)
        if mask.any():
            ax_sc.scatter(
                proxy_pct[mask],
                residual_pct[mask],
                s=sizes[mask],
                color=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.45,
                label=f"{label} high-tau",
            )
    ax_sc.axvline(driver_threshold, color="#111827", linewidth=0.9, alpha=0.7)
    ax_sc.axhline(driver_threshold, color="#111827", linewidth=0.9, alpha=0.7)
    ax_sc.axvline(0.90, color="#6b7280", linewidth=0.75, linestyle=":", alpha=0.6)
    ax_sc.axhline(0.90, color="#6b7280", linewidth=0.75, linestyle=":", alpha=0.6)
    ax_sc.set_xlim(0, 1.02)
    ax_sc.set_ylim(0, 1.02)
    ax_sc.set_xlabel("Smoothed proxy exposure factor percentile")
    ax_sc.set_ylabel("Exposure-weighted residual-energy factor percentile")
    ax_sc.set_title("High-tau months in exact target-month factor space")
    ax_sc.legend(loc="lower right", fontsize=8.0, frameon=True)
    fig.tight_layout()
    figure_png = FIGURES / "iv_tau_driver_diagnostic.png"
    figure_pdf = FIGURES / "iv_tau_driver_diagnostic.pdf"
    fig.savefig(figure_png, dpi=180, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    fig_hm, ax_hm = plt.subplots(figsize=(10.5, 2.5))
    heat = np.vstack([tau_pct, proxy_pct, residual_pct])
    im = ax_hm.imshow(heat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax_hm.set_yticks([0, 1, 2])
    ax_hm.set_yticklabels(["tau_soft percentile", "exposure factor", "residual factor"])
    tick_idx = np.linspace(0, len(dates) - 1, min(9, len(dates))).astype(int)
    ax_hm.set_xticks(tick_idx)
    ax_hm.set_xticklabels([dates.iloc[i].strftime("%Y") for i in tick_idx], rotation=0)
    ax_hm.set_title("Proxy-IV tau_soft and exact factor percentile heatmap")
    fig_hm.colorbar(im, ax=ax_hm, shrink=0.82, label="Within-sample percentile")
    fig_hm.tight_layout()
    heatmap_png = FIGURES / "iv_tau_driver_heatmap.png"
    heatmap_pdf = FIGURES / "iv_tau_driver_heatmap.pdf"
    fig_hm.savefig(heatmap_png, dpi=180, bbox_inches="tight")
    fig_hm.savefig(heatmap_pdf, bbox_inches="tight")
    plt.close(fig_hm)

    caption = rf"""The figure characterizes an exact multiplicative decomposition of proxy-IV OVK amplification on the full coordinate grid. Panel A plots the soft ridge-whitened proxy-IV amplification index \(\widehat\tau_{{\mathrm{{soft}},t}}\), the smoothed proxy-exposure factor, and the exposure-weighted residual-energy factor; the two colored factor lines multiply to \(\widehat\tau_{{\mathrm{{soft}},t}}\) at every plotted month up to numerical tolerance. High-\(\tau\) months are defined by within-sample percentile rank at least {high_tau_threshold:.2f} and are colored by whether the target-month exposure factor, the target-month residual factor, or both are unusually high. Panel B plots all months in this exact target-month factor space. The source-month score identity is \(\chi_{{\mathrm{{proxy}},r}}=(M_r/\widehat\kappa)u_r\) with \(\widehat\kappa=n^{{-1}}\sum_r M_rX_r\), so \(\chi_{{\mathrm{{proxy}},r}}'D_\rho^{{-1}}\chi_{{\mathrm{{proxy}},r}}/d_\rho=E_rR_r\). For the fitted linear temporal covariance smoother, \(\widehat\tau_{{\mathrm{{soft}},t}}=\sum_r \widehat w_{{tr}}E_rR_r=\mathrm{{exposure\ factor}}_t\times\mathrm{{residual\ factor}}_t\). The interaction between exposure and residual energy is assigned to the residual factor through exposure-tilted weights; this ordering is useful but not unique.
"""
    caption_path = FIGURES / "iv_tau_driver_diagnostic_caption.tex"
    caption_path.write_text(caption, encoding="utf-8")
    high = monthly["high_tau"].to_numpy(bool)
    high_count = max(int(high.sum()), 1)
    high_labels = monthly.loc[high, "driver_label"]
    shares = high_labels.value_counts(normalize=True).to_dict() if high.any() else {}
    diagnostics = {
        "coordinate_dimension": int(chi_proxy.shape[1]),
        "backend": "full_coordinate_temporal_kernel",
        "covariance_estimator_linear": True,
        "smoothing_weights_source": "estimate_full_coordinate_kernel_model(...).temporal_weights",
        "weighted_covariance_representation": "K_hat[t] = sum_r temporal_weights[t,r] chi_proxy[r] chi_proxy[r]'",
        "additive_matrix_component": "none",
        "ridge_rho": float(full_iv.rho),
        "d_rho": float(full_iv.d_rho),
        "kernel_eta": float(full_iv.kernel_eta),
        "kappa_hat": kappa_hat,
        "pi_hat": kappa_hat,
        "n_obs": int(len(dates)),
        "driver_threshold": driver_threshold,
        "high_tau_quantile": high_tau_threshold,
        "high_tau_threshold": high_tau_threshold_value,
        "tau_min": float(np.nanmin(tau)),
        "tau_mean": float(np.nanmean(tau)),
        "tau_max": float(np.nanmax(tau)),
        "valid_dates": int(len(dates)),
        "weight_row_sum_min": float(np.min(row_sums)),
        "weight_row_sum_max": float(np.max(row_sums)),
        "weight_row_sum_max_abs_error": max_abs_row_sum_error,
        "negative_weight_count": negative_weight_count,
        "negative_weights_occur": any_negative_weights,
        "minimum_weight": float(np.min(W)),
        "maximum_weight": float(np.max(W)),
        "minimum_effective_weight_count": float(np.min(effective_weight_count)),
        "maximum_effective_weight_count": float(np.max(effective_weight_count)),
        "E_ref": E_ref,
        "E_ref_method": "arithmetic mean of E_raw_smoothed over valid plotted dates; no separate Figure 3 target-date reference weights are used",
        "share_high_tau_proxy_exposure": float(shares.get("proxy-exposure", 0.0)),
        "share_high_tau_residual_energy": float(shares.get("residual-energy", 0.0)),
        "share_high_tau_mixed": float(shares.get("mixed", 0.0)),
        "share_high_tau_unclassified": float(shares.get("unclassified", 0.0)),
        "corr_tau_proxy_percentile": float(pd.Series(tau).corr(pd.Series(proxy_pct))),
        "corr_tau_resid_percentile": float(pd.Series(tau).corr(pd.Series(residual_pct))),
        "corr_tau_product_percentile": float(pd.Series(tau).corr(pd.Series(product_pct))),
        "rel_product_error": rel_product_error_global,
        "max_abs_multiplicative_energy_product_error": max_abs_product_error,
        "max_rel_multiplicative_energy_product_error": max_rel_product_error,
        "rel_fro_multiplicative_energy_product_error": rel_fro_product_error,
        "max_abs_reconstruction_error": max_abs_reconstruction_error,
        "max_rel_reconstruction_error": max_rel_reconstruction_error,
        "max_abs_factor_product_error": max_abs_factor_product_error,
        "max_rel_factor_product_error": max_rel_factor_product_error,
        "mean_tau": float(np.mean(tau)),
        "mean_proxy_exposure_a2": float(np.mean(proxy_exposure_a2)),
        "mean_residual_energy_soft": float(np.mean(residual_energy_soft)),
        "mean_score_energy_soft": float(np.mean(score_energy_soft)),
        "mean_E_raw_smoothed": float(np.mean(E_bar)),
        "mean_R_exposure_weighted_raw": float(np.mean(R_tilde)),
        "mean_exposure_factor": float(np.mean(exposure_factor)),
        "mean_residual_factor": float(np.mean(residual_factor)),
        "corr_tau_log_proxy_exposure": float(pd.Series(tau).corr(pd.Series(np.log(np.maximum(exposure_factor, 1e-300))))),
        "corr_tau_log_residual_energy_soft": float(
            pd.Series(tau).corr(pd.Series(np.log(np.maximum(residual_factor, 1e-300))))
        ),
        "notes": (
            "The plotted factor lines are exact target-month factors, not contemporaneous source-month diagnostics. "
            "Full-coordinate source score_energy_soft equals (M_r^2/kappa_hat^2) times residual_energy_soft, "
            "and the temporal-kernel weights map those source products exactly into tau_soft."
        ),
        "monthly_csv": str(monthly_path),
        "audit_csv": str(audit_path),
        "top_months_csv": str(top_months_path),
        "episodes_csv": str(episodes_path),
        "table_csv": str(monthly_path),
        "legacy_table_csv": str(legacy_table_path),
        "figure_png": str(figure_png),
        "figure_pdf": str(figure_pdf),
        "heatmap_png": str(heatmap_png),
        "heatmap_pdf": str(heatmap_pdf),
        "caption_tex": str(caption_path),
    }
    diagnostics_path = TABLES / "iv_tau_driver_diagnostic_summary.json"
    diagnostics["diagnostics_json"] = str(diagnostics_path)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    legacy_diagnostics_path = TABLES / "iv_tau_multiplicative_driver_diagnostics.json"
    legacy_diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    return {
        "path": monthly,
        "top_months": top_months,
        "episodes": episodes,
        "diagnostics": diagnostics,
        "diagnostics_df": pd.DataFrame([diagnostics]),
        "table_path": monthly_path,
        "audit_path": audit_path,
        "top_months_path": top_months_path,
        "episodes_path": episodes_path,
        "diagnostics_path": diagnostics_path,
        "figure_png": figure_png,
        "figure_pdf": figure_pdf,
        "heatmap_png": heatmap_png,
        "heatmap_pdf": heatmap_pdf,
        "caption_path": caption_path,
    }


def read_baseline_tau() -> pd.DataFrame:
    candidates = [
        BASE_DIR / "publication_grade_ovk" / "outputs" / "tables" / "publication_grade_headline_state_path.csv",
        BASE_DIR / "top5_headline" / "outputs" / "tables" / "state_space_A_t_top5_drift_estimates_with_bands.csv",
        BASE_DIR / "top5_full_appended_results_pack" / "top5_baseline_state_space_results" / "tables" / "state_space_A_t_top5_drift_estimates_with_bands.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            tau_col = "tau" if "tau" in df.columns else "trace_A_over_R" if "trace_A_over_R" in df.columns else None
            if tau_col is None:
                continue
            return df[["date", tau_col]].rename(columns={tau_col: "baseline_tau"})
    return pd.DataFrame(columns=["date", "baseline_tau"])


def write_baseline_overlay(headline: Any) -> dict[str, Any]:
    iv = pd.DataFrame({"date": pd.to_datetime(headline.dates), "iv_tau": headline.tau})
    base = read_baseline_tau()
    if base.empty:
        return {"baseline_available": False}
    merged = iv.merge(base, on="date", how="inner").dropna()
    if merged.empty:
        return {"baseline_available": False}
    corr = float(merged["iv_tau"].corr(merged["baseline_tau"]))
    iv_top10 = set(merged.nlargest(10, "iv_tau")["date"].dt.strftime("%Y-%m"))
    base_top10 = set(merged.nlargest(10, "baseline_tau")["date"].dt.strftime("%Y-%m"))
    stats = {
        "baseline_available": True,
        "overlap_observations": int(len(merged)),
        "tau_correlation": corr,
        "top10_overlap": int(len(iv_top10 & base_top10)),
        "iv_max_month": merged.loc[merged["iv_tau"].idxmax(), "date"].strftime("%Y-%m"),
        "iv_max_tau": float(merged["iv_tau"].max()),
        "baseline_max_month": merged.loc[merged["baseline_tau"].idxmax(), "date"].strftime("%Y-%m"),
        "baseline_max_tau": float(merged["baseline_tau"].max()),
    }
    pd.DataFrame([stats]).to_csv(TABLES / "iv_vs_baseline_tau_comparison.csv", index=False)
    fig = plt.figure(figsize=(9.2, 5.0))
    plt.plot(merged["date"], merged["iv_tau"], label="Proxy-IV tau_t", color="black")
    plt.plot(merged["date"], merged["baseline_tau"], label="Baseline base5 tau_t", alpha=0.8)
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t")
    plt.title("Proxy-IV versus baseline covariance amplification")
    plt.legend()
    savefig(fig, "iv_tau_overlay_baseline.png")
    return stats


def run_nested_comparison_iv(scores: dict[str, Any]) -> dict[str, Any]:
    Q = np.asarray(scores["Q_scores"], float)
    dates = pd.to_datetime(scores["dates"]).reset_index(drop=True)
    N, M = Q.shape
    train_end = 180 if N > 204 else max(50, min(N - 24, int(0.65 * N)))
    val_start = max(20, min(100, train_end // 2))
    if N <= train_end + 12:
        train_end = max(30, int(0.6 * N))
        val_start = max(10, train_end // 2)
    if N <= train_end + 8:
        raise RuntimeError("IV sample too short for nested mean-covariance comparison.")

    eig_train = covariance_eigenbasis(Q, train_end=train_end)
    bases = build_mean_and_eval_basis(eig_train["beta"], eig_train["V"], d_eval=min(10, M))
    W = bases["W"]
    Bbeta = bases["Bbeta"]
    Bc = W.T @ Bbeta
    y = Q @ W
    models = ["M0", "M1", "M2", "M3"]
    params: dict[str, ModelParams] = {}
    outputs: dict[str, dict[str, Any]] = {}
    score_dict: dict[str, np.ndarray] = {}
    for model in models:
        params[model] = tune_model(y, Bc, model, train_end=train_end, val_start=val_start)
        outputs[model] = run_predictive_scores(y, Bc, model, train_end, N, params[model])
        score_dict[model] = outputs[model]["log_scores"]

    model_summary = pd.DataFrame(
        [
            {
                "model": m,
                "avg_log_score": float(np.mean(score_dict[m])),
                "sum_log_score": float(np.sum(score_dict[m])),
                "kmean": params[m].kmean,
                "kcov": params[m].kcov,
                "phi": params[m].phi,
                "cov_shrink": params[m].cov_shrink,
                "cov_target_weight": params[m].cov_target_weight,
            }
            for m in models
        ]
    )
    pairs = [
        ("M1", "M0", "Dynamic covariance vs fixed benchmark"),
        ("M2", "M0", "Dynamic mean vs fixed benchmark"),
        ("M3", "M0", "Joint mean-covariance vs fixed benchmark"),
        ("M3", "M1", "Does joint model beat dynamic covariance only?"),
        ("M3", "M2", "Does joint model beat dynamic mean only?"),
        ("M2", "M1", "Mean-only vs covariance-only"),
    ]
    comp_rows = []
    for a, b, meaning in pairs:
        diff = score_dict[a] - score_dict[b]
        mean, lo, hi, prob = block_bootstrap_ci(
            diff,
            block_len=IV_NESTED_BOOT_BLOCK,
            B=IV_NESTED_BOOT_DRAWS,
            seed=1234,
        )
        comp_rows.append(
            {
                "comparison": f"{a} - {b}",
                "meaning": meaning,
                "avg_log_score_diff": mean,
                "p05": lo,
                "p95": hi,
                "prob_diff_gt_0": prob,
                "sum_log_score_diff": float(np.sum(diff)),
            }
        )
    comparison = pd.DataFrame(comp_rows)
    comparison.to_csv(TABLES / "iv_nested_log_score_comparison.csv", index=False)
    model_summary.to_csv(TABLES / "iv_nested_model_summary.csv", index=False)

    eig_full = covariance_eigenbasis(Q, train_end=None)
    bases_full = build_mean_and_eval_basis(eig_full["beta"], eig_full["V"], d_eval=min(10, M))
    W_full = bases_full["W"]
    Bc_full = W_full.T @ bases_full["Bbeta"]
    y_full = Q @ W_full
    mu_full = y_full.mean(axis=0)
    kmean_m3 = params["M3"].kmean if params["M3"].kmean > 0 else params["M2"].kmean
    gamma_full = filtered_gamma_path(y_full - mu_full, Bc_full, kmean=kmean_m3, phi=params["M3"].phi)
    mean_drift_q = (gamma_full @ Bc_full.T) @ W_full.T
    V_R = eig_full["V"][:, :HEADLINE_R]
    lam_R = eig_full["evals"][:HEADLINE_R]
    lam_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(lam_R, 1e-12)))
    z_m1 = (Q - eig_full["beta"]) @ V_R @ lam_inv_sqrt
    z_m3 = (Q - eig_full["beta"] - mean_drift_q) @ V_R @ lam_inv_sqrt
    upgraded_m1 = upgraded_state_space_A_from_z(z_m1, HEADLINE_R)
    upgraded_m3 = upgraded_state_space_A_from_z(z_m3, HEADLINE_R)
    trace_m1 = upgraded_m1["tau"]
    trace_m3 = upgraded_m3["tau"]
    top_n = min(12, len(trace_m1))
    top_idx_m1 = np.argsort(trace_m1)[::-1][:top_n]
    top_idx_m3 = np.argsort(trace_m3)[::-1][:top_n]
    overlap = len(set(top_idx_m1) & set(top_idx_m3))
    survival = pd.DataFrame(
        {
            "quantity": [
                "Correlation of trace(A_t) paths, M1 vs M3",
                "Top-12 spike overlap count",
                "Top-12 spike overlap share",
                "Mean trace(A_t), M1",
                "Mean trace(A_t), M3",
                "SD trace(A_t), M1",
                "SD trace(A_t), M3",
                "Max trace(A_t), M1",
                "Max trace(A_t), M3",
                "Date of max M1 amplification",
                "Date of max M3 amplification",
            ],
            "value": [
                float(np.corrcoef(trace_m1, trace_m3)[0, 1]),
                int(overlap),
                float(overlap / top_n) if top_n else np.nan,
                float(np.mean(trace_m1)),
                float(np.mean(trace_m3)),
                float(np.std(trace_m1, ddof=0)),
                float(np.std(trace_m3, ddof=0)),
                float(np.max(trace_m1)),
                float(np.max(trace_m3)),
                dates.iloc[int(np.argmax(trace_m1))].strftime("%Y-%m-%d"),
                dates.iloc[int(np.argmax(trace_m3))].strftime("%Y-%m-%d"),
            ],
        }
    )
    survival.to_csv(TABLES / "iv_a_survival_after_mean_adjustment.csv", index=False)
    paths = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "traceA_M1_fixed_mean": trace_m1,
            "traceA_M3_after_mean_adjustment": trace_m3,
            "mean_drift_norm": np.linalg.norm(mean_drift_q, axis=1),
        }
    )
    paths.to_csv(TABLES / "iv_a_survival_trace_paths.csv", index=False)

    fig = plt.figure(figsize=(7.0, 4.6))
    plt.bar(model_summary["model"], model_summary["avg_log_score"])
    base = model_summary.loc[model_summary["model"].eq("M0"), "avg_log_score"].iloc[0]
    plt.axhline(base, linestyle="--", linewidth=1.0, color="black")
    plt.ylabel("Average one-step log score")
    plt.title("IV nested mean-covariance log scores")
    savefig(fig, "iv_nested_log_score_bar.png")

    fig = plt.figure(figsize=(9.0, 5.0))
    plt.plot(dates, trace_m1, label="M1 fixed mean")
    plt.plot(dates, trace_m3, label="M3 after mean adjustment")
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("trace(A_t)/5")
    plt.title("IV A_t survival after mean adjustment")
    plt.legend()
    savefig(fig, "iv_a_survival_trace_overlay.png")

    return {
        "model_summary": model_summary,
        "comparison": comparison,
        "survival": survival,
        "paths": paths,
    }


def init_iv_bootstrap_worker(context: dict[str, Any]) -> None:
    global _IV_BOOTSTRAP_CONTEXT
    _IV_BOOTSTRAP_CONTEXT = context


def iv_bootstrap_draw_task(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    ctx = _IV_BOOTSTRAP_CONTEXT
    draw = int(task["draw"])
    ix = np.asarray(ctx["indices"][draw], dtype=int)
    try:
        Qb, info = iv_lp_scores_from_design(ctx["C_design"], ctx["x_vector"], ctx["z_vector"], ctx["Y_resp"], ix=ix)
        dates = pd.Series(pd.to_datetime(ctx["dates_ns"].astype("datetime64[ns]")))
        rb = estimate_rank_model(
            Qb,
            dates,
            ctx["variant"],
            "IV bootstrap",
            int(ctx["rank"]),
            em_iters=int(ctx["em_iters"]),
            outcome_labels=list(ctx["outcome_labels"]),
        )
        diag = first_stage_diagnostics_from_residuals(
            info["x_res"],
            info["z_res"],
            dates,
            ctx["x_col"],
            ctx["z_col"],
            ctx["cbi_col"],
            q_zx=info["q_zx"],
        )
        row = {
            "draw": draw + 1,
            "alpha_hat": rb.fit.alpha,
            "retained_trace_share": float(rb.shares[: int(ctx["rank"])].sum()),
            "max_tau": float(rb.tau.max()),
            "max_month": dates.iloc[int(np.argmax(rb.tau))].strftime("%Y-%m"),
            "first_stage_f_stat": diag["first_stage_f_stat"],
            "q_zx": diag["q_zx"],
            "weak_iv_warning": diag["weak_iv_warning"],
            "error": "",
        }
        payload = {"tau": rb.tau, "row": row}
    except Exception as exc:
        payload = {
            "tau": np.full(len(ctx["dates_ns"]), np.nan),
            "row": {"draw": draw + 1, "error": str(exc)},
        }
    return draw, payload


def run_iv_bootstrap(scores: dict[str, Any], headline: Any, workers: int = PUBLICATION_WORKERS) -> dict[str, Any]:
    n = len(scores["dates"])
    rng = np.random.default_rng(BOOT_SEED)
    indices = np.asarray([circular_block_indices(n, BOOT_BLOCK_LEN, rng) for _ in range(IV_BOOT_DRAWS)], dtype=np.int64)
    context = {
        "C_design": np.asarray(scores["C_design"], float),
        "x_vector": np.asarray(scores["x_vector"], float),
        "z_vector": np.asarray(scores["z_vector"], float),
        "Y_resp": np.asarray(scores["Y_resp"], float),
        "indices": indices,
        "dates_ns": pd.to_datetime(scores["dates"]).to_numpy(dtype="datetime64[ns]"),
        "variant": SPEC.key,
        "rank": int(headline.rank),
        "em_iters": BOOT_EM_ITERS,
        "outcome_labels": list(scores["outcome_labels"]),
        "x_col": scores["x_col"],
        "z_col": scores["z_col"],
        "cbi_col": scores.get("cbi_col", ""),
    }
    tau_boot = np.empty((IV_BOOT_DRAWS, n))
    rows = []
    tasks = [{"draw": b} for b in range(IV_BOOT_DRAWS)]
    results = run_parallel_tasks(
        iv_bootstrap_draw_task,
        tasks,
        workers,
        initializer=init_iv_bootstrap_worker,
        initargs=(context,),
    )
    for draw, payload in results:
        tau_boot[draw] = payload["tau"]
        rows.append(payload["row"])
    boot_df = pd.DataFrame(sorted(rows, key=lambda row: row.get("draw", 0)))
    boot_df.to_csv(TABLES / "iv_bootstrap_top5_trace_share.csv", index=False)
    valid = np.isfinite(tau_boot).all(axis=1)
    tau_valid = tau_boot[valid]
    if len(tau_valid):
        band = positive_simultaneous_band(tau_valid, headline.tau, level=0.90)
    else:
        band = positive_simultaneous_band(np.asarray([headline.tau]), headline.tau, level=0.90)
    band_df = pd.DataFrame(
        {
            "date": pd.to_datetime(scores["dates"]).dt.strftime("%Y-%m-%d"),
            "tau": headline.tau,
            "tau_p05": band["point_low"],
            "tau_p50": band["point_med"],
            "tau_p95": band["point_high"],
            "tau_simul_p05": band["sim_low"],
            "tau_simul_p95": band["sim_high"],
        }
    )
    band_df.to_csv(TABLES / "iv_bootstrap_tau_bands.csv", index=False)
    fig = plt.figure(figsize=(9.2, 5.0))
    dates = pd.to_datetime(scores["dates"])
    plt.plot(dates, headline.tau, color="black", label="tau_t")
    plt.fill_between(dates, band_df["tau_p05"], band_df["tau_p95"], alpha=0.25, label="90% pointwise full-pipeline band")
    plt.plot(dates, band_df["tau_simul_p05"], linestyle="--", linewidth=1.0, label="90% log-sim lower")
    plt.plot(dates, band_df["tau_simul_p95"], linestyle="--", linewidth=1.0, label="90% log-sim upper")
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t = trace(A_t)/5")
    plt.title("Proxy-IV full-pipeline bootstrap tau bands")
    plt.legend(fontsize=8)
    savefig(fig, "iv_bootstrap_tau_bands.png")
    return {"boot_df": boot_df, "tau_boot": tau_boot, "tau_boot_valid": tau_valid, "band": band, "band_df": band_df}


def _date_range_text(dates: pd.Series | pd.Index | np.ndarray | list[Any]) -> str:
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if parsed.empty:
        return "not available"
    return f"{parsed.min().strftime('%Y-%m-%d')} to {parsed.max().strftime('%Y-%m-%d')}"


def _finite_date_range_text(panel: pd.DataFrame, columns: list[str], require_all: bool = True) -> tuple[int, str]:
    present = [c for c in columns if c in panel.columns]
    if not present or "date" not in panel.columns:
        return 0, "not available"
    finite = pd.DataFrame({c: pd.to_numeric(panel[c], errors="coerce").notna() for c in present})
    mask = finite.all(axis=1) if require_all else finite.any(axis=1)
    if not mask.any():
        return 0, "not available"
    dates = pd.to_datetime(panel.loc[mask, "date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return 0, "not available"
    return int(mask.sum()), _date_range_text(dates)


def _incomplete_dates_text(panel: pd.DataFrame, columns: list[str], max_show: int = 8) -> str:
    present = [c for c in columns if c in panel.columns]
    if not present or "date" not in panel.columns:
        return "not available"
    finite = pd.DataFrame({c: pd.to_numeric(panel[c], errors="coerce").notna() for c in present})
    dates = pd.to_datetime(panel["date"], errors="coerce")
    missing_dates = dates.loc[dates.notna() & ~finite.all(axis=1)]
    if missing_dates.empty:
        return "none"
    shown = [d.strftime("%Y-%m-%d") for d in missing_dates.head(max_show)]
    suffix = "" if len(missing_dates) <= max_show else f"; plus {len(missing_dates) - max_show} more"
    unit = "month" if len(missing_dates) == 1 else "months"
    return f"{', '.join(shown)} ({len(missing_dates)} {unit}{suffix})"


def _sample_value(sample_summary: pd.DataFrame, item: str) -> str:
    if sample_summary is None or sample_summary.empty or "item" not in sample_summary or "value" not in sample_summary:
        return ""
    values = sample_summary.loc[sample_summary["item"].eq(item), "value"]
    return "" if values.empty else str(values.iloc[0])


def write_sample_summary(panel: pd.DataFrame, scores: dict[str, Any], merge_meta: dict[str, Any]) -> pd.DataFrame:
    dates = pd.to_datetime(scores["dates"])
    panel_dates = pd.to_datetime(panel["date"], errors="coerce")
    h = int(scores.get("H", H))
    lag_count = int(scores.get("L", L))
    pvars = int(scores.get("pvars", len(scores.get("outcome_labels", []))))
    horizon_count = h + 1
    score_dim = int(scores["Q_scores"].shape[1])
    outcome_cols = list(scores.get("outcome_columns", _outcome_columns_for_labels(scores["outcome_labels"])))
    outcome_n, outcome_range = _finite_date_range_text(panel, outcome_cols, require_all=True)
    outcome_incomplete = _incomplete_dates_text(panel, outcome_cols)
    shock_cols = [c for c in ["MP_median_fallback", "CBI_median_fallback"] if c in panel.columns]
    shock_n, shock_range = _finite_date_range_text(panel, shock_cols, require_all=True)
    policy_n, policy_range = _finite_date_range_text(panel, [scores["x_col"]], require_all=True)
    instrument_n, instrument_range = _finite_date_range_text(panel, [scores["z_col"]], require_all=True)
    cbi_col = str(scores.get("cbi_col", ""))
    cbi_n, cbi_range = _finite_date_range_text(panel, [cbi_col], require_all=True) if cbi_col else (0, "not used")
    valid_idx = np.asarray(scores.get("valid_idx", []), dtype=int)
    if len(valid_idx):
        raw_start_idx = max(int(valid_idx.min()) - lag_count - 1, 0)
        raw_end_idx = min(int(valid_idx.max()) + h, len(panel) - 1)
        raw_required = _date_range_text(panel["date"].iloc[[raw_start_idx, raw_end_idx]])
    else:
        raw_required = "not available"
    common_score_range = f"{dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}"
    q_zx = float(scores.get("q_zx", np.nan))
    rows = [
        {"item": "Merged IV panel calendar coverage", "value": f"{_date_range_text(panel_dates)} ({len(panel)} monthly rows)"},
        {"item": "Headline outcomes finite coverage", "value": f"{outcome_range} ({outcome_n} rows with all {pvars} outcomes finite)"},
        {"item": "Headline outcome incomplete months", "value": outcome_incomplete},
        {"item": "Conventional shock-date coverage", "value": f"{shock_range} ({shock_n} rows with MP and CBI finite)"},
        {"item": "External instrument shock-date coverage", "value": f"{instrument_range} ({instrument_n} finite {scores['z_col']} rows)"},
        {"item": "Endogenous policy indicator coverage", "value": f"{policy_range} ({policy_n} finite {scores['x_col']} rows)"},
        {"item": "CBI control coverage", "value": f"{cbi_range} ({cbi_n} finite {cbi_col} rows)" if cbi_col else cbi_range},
        {"item": "IV score observations", "value": len(dates)},
        {"item": "Score-surface dimension", "value": score_dim},
        {"item": "Coordinate grid", "value": f"{pvars} outcomes x {horizon_count} horizons = {score_dim} coordinates"},
        {"item": "Common complete-coordinate score coverage", "value": common_score_range},
        {"item": "Usable IV sample range", "value": common_score_range},
        {"item": "State index attached to", "value": "base month t (the plotted date)"},
        {"item": "Raw outcome window consumed by score sample", "value": raw_required},
        {"item": "Horizons", "value": f"0 to {h} months"},
        {"item": "Lagged controls", "value": f"{lag_count} monthly lags of policy/control/outcomes and outcome differences"},
        {
            "item": "First-stage denominator sample",
            "value": f"A single q_zx is estimated once on the common {len(dates)}-row complete-coordinate sample",
        },
        {"item": "First-stage denominator q_zx", "value": q_zx},
        {"item": "Outcomes", "value": ", ".join(scores["outcome_labels"])},
        {"item": "Endogenous policy indicator", "value": scores["x_col"]},
        {"item": "Excluded external instrument", "value": scores["z_col"]},
        {"item": "Instrument preferred source", "value": merge_meta.get("instrument_preferred", "")},
        {"item": "Instrument source column", "value": merge_meta.get("instrument_source_column", "")},
        {"item": "CBI control", "value": scores.get("cbi_col", "")},
        {"item": "Finite DGS1 observations in panel", "value": int(np.isfinite(pd.to_numeric(panel["dgs1_eom"], errors="coerce")).sum())},
        {"item": "Finite preferred instrument observations in panel", "value": int(np.isfinite(pd.to_numeric(panel["iv_z_preferred"], errors="coerce")).sum())},
        {"item": "First-stage F-stat", "value": scores["first_stage"]["first_stage_f_stat"]},
        {"item": "Weak-IV warning", "value": scores["first_stage"]["weak_iv_warning"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "iv_lp_sample_summary.csv", index=False)
    return df


def write_file_manifest() -> pd.DataFrame:
    rows = []
    roots = [IV_ROOT, FINAL_PDF, FINAL_HTML, FINAL_ZIP, PROCESSED_PANEL]
    seen: set[Path] = set()
    manifest_path = (TABLES / "iv_file_manifest.csv").resolve()
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            path = path.resolve()
            if path in seen or path == manifest_path:
                continue
            seen.add(path)
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            rows.append(
                {
                    "relative_path": str(rel),
                    "bytes": path.stat().st_size,
                    "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                    "sha256": sha256_file(path),
                }
            )
    df = pd.DataFrame(rows).sort_values("relative_path")
    df.to_csv(TABLES / "iv_file_manifest.csv", index=False)
    return df


def _table_for_report(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 7) -> Table:
    view = df.iloc[:max_rows, :max_cols].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    rows = [[Paragraph(f"<b>{html.escape(str(c))}</b>", getSampleStyleSheet()["BodyText"]) for c in view.columns]]
    for _, row in view.iterrows():
        rows.append([Paragraph(html.escape(str(v)), getSampleStyleSheet()["BodyText"]) for v in row.tolist()])
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEDED")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build_pdf_report(
    sample_summary: pd.DataFrame,
    first_stage: pd.DataFrame,
    avg_resp: pd.DataFrame,
    rank_df: pd.DataFrame,
    top_df: pd.DataFrame,
    decomposition: dict[str, Any],
    exact_decomposition: dict[str, Any],
    driver_diagnostic: dict[str, Any],
    nested: dict[str, Any],
    baseline_stats: dict[str, Any],
    bootstrap: dict[str, Any],
    merge_meta: dict[str, Any],
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontSize=17, leading=21, spaceAfter=8))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=12.5, leading=15, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.8, leading=11.5))

    def P(text: str, style: str = "Bodyx") -> Paragraph:
        return Paragraph(html.escape(str(text)), styles[style])

    story: list[Any] = []
    story.append(P("Proxy-IV LP/OVK Appendix", "TitleX"))
    story.append(P("This appendix constructs LP score surfaces from 2SLS/external-instrument influence contributions. The endogenous policy indicator is DGS1 and the excluded instrument is the SF Fed monetary-policy surprise."))
    story.append(P("The object is time-varying covariance of IV-LP response-score surfaces, not an unrestricted time-varying structural IRF estimate."))

    story.append(P("Data Provenance", "H1x"))
    source_cols = ["source", "used_cached", "file_size", "parser_decision", "error"]
    sources_report = pd.read_csv(TABLES / "iv_data_sources.csv")
    story.append(_table_for_report(sources_report[[c for c in source_cols if c in sources_report.columns]], max_rows=5))
    story.append(P(f"Processed IV panel: {PROCESSED_PANEL}"))
    story.append(P(f"SF Fed source column used: {merge_meta.get('instrument_source_column', '')}; sample source range {merge_meta.get('source_start', '')} to {merge_meta.get('source_end', '')}."))
    story.append(_table_for_report(sample_summary, max_rows=30))
    complete_range = _sample_value(sample_summary, "Common complete-coordinate score coverage")
    raw_window = _sample_value(sample_summary, "Raw outcome window consumed by score sample")
    denom_sample = _sample_value(sample_summary, "First-stage denominator sample")
    story.append(
        P(
            "The state index is attached to the base month t. "
            f"In this run the common complete-coordinate score coverage is {complete_range}; "
            f"the LP construction consumes raw outcome data over {raw_window}. "
            f"{denom_sample}, so the 125 coordinates share the same scalar proxy-IV first-stage denominator."
        )
    )

    story.append(P("First Stage", "H1x"))
    fs_cols = ["nobs", "sample_start", "sample_end", "q_zx", "corr_zx", "first_stage_partial_r2", "first_stage_f_stat", "weak_iv_warning"]
    story.append(_table_for_report(first_stage[[c for c in fs_cols if c in first_stage.columns]], max_rows=1))
    story.append(Image(str(CHARTS / "iv_first_stage_scatter.png"), width=5.9 * inch, height=3.9 * inch))

    story.append(PageBreak())
    story.append(P("Average IV LP Responses", "H1x"))
    story.append(P("The response table reports horizons 0-24 for the five headline outcomes."))
    story.append(Image(str(CHARTS / "iv_average_lp_responses.png"), width=6.4 * inch, height=3.9 * inch))
    story.append(_table_for_report(avg_resp, max_rows=8, max_cols=6))

    story.append(P("OVK Geometry", "H1x"))
    rank_cols = ["rank", "alpha_hat", "retained_trace_share", "eigengap_R_to_Rplus1", "tau_mean", "tau_sd", "tau_max", "tau_max_month"]
    story.append(_table_for_report(rank_df[[c for c in rank_cols if c in rank_df.columns]], max_rows=10))
    story.append(Image(str(CHARTS / "iv_eigenspectrum.png"), width=5.9 * inch, height=3.5 * inch))
    story.append(Image(str(CHARTS / "iv_tau_path.png"), width=6.4 * inch, height=3.7 * inch))

    story.append(PageBreak())
    story.append(P("Top Amplification Months and Shape", "H1x"))
    story.append(_table_for_report(top_df, max_rows=10, max_cols=8))
    story.append(Image(str(CHARTS / "iv_top5_basis_surfaces.png"), width=7.1 * inch, height=3.3 * inch))

    story.append(P("Score-Energy Decomposition", "H1x"))
    story.append(P("The retained IV score energy is decomposed into first-stage leverage and retained structural-residual energy. The z-score and fitted-first-stage score forms are algebraically equivalent; the reported max difference is an implementation check."))
    decomp_summary_cols = [
        "max_abs_diff_between_z_score_and_fitted_score_forms",
        "driver_label_for_max_tau_month",
        "top10_first_stage_driven_count",
        "top10_residual_driven_count",
        "top10_mixed_count",
        "corr_tau_log_first_stage_leverage",
        "corr_tau_log_retained_residual_energy",
    ]
    decomp_summary = decomposition["summary"]
    story.append(_table_for_report(decomp_summary[[c for c in decomp_summary_cols if c in decomp_summary.columns]], max_rows=1))
    top_decomp_cols = [
        "tau_rank",
        "date",
        "tau",
        "driver_label",
        "first_stage_leverage_percentile",
        "retained_residual_energy_percentile",
        "log_first_stage_leverage",
        "log_retained_residual_energy",
    ]
    story.append(_table_for_report(decomposition["top_tau"][[c for c in top_decomp_cols if c in decomposition["top_tau"].columns]], max_rows=10))
    story.append(Image(str(CHARTS / "iv_score_decomposition_top_tau.png"), width=6.5 * inch, height=3.7 * inch))

    story.append(P("Multiplicative Tau Drivers", "H1x"))
    story.append(P("The proxy-IV full-coordinate score is chi_proxy[r]=(M_r/kappa_hat) u_r, with kappa_hat=mean(M_r X_r). For the linear temporal-kernel covariance smoother, tau_soft[t] is exactly the weighted average of E_r R_r and is plotted as the product of a smoothed proxy exposure factor and an exposure-weighted residual-energy factor."))
    driver_cols = [
        "coordinate_dimension",
        "kappa_hat",
        "n_obs",
        "ridge_rho",
        "d_rho",
        "max_abs_reconstruction_error",
        "max_rel_reconstruction_error",
        "weight_row_sum_min",
        "weight_row_sum_max",
    ]
    driver_diag = driver_diagnostic["diagnostics_df"]
    story.append(_table_for_report(driver_diag[[c for c in driver_cols if c in driver_diag.columns]], max_rows=1, max_cols=8))
    story.append(Image(str(driver_diagnostic["figure_png"]), width=6.5 * inch, height=3.7 * inch))

    story.append(P("Comparison to Baseline", "H1x"))
    if baseline_stats.get("baseline_available"):
        base_cols = ["overlap_observations", "tau_correlation", "top10_overlap", "iv_max_month", "iv_max_tau", "baseline_max_month", "baseline_max_tau"]
        base_df = pd.DataFrame([baseline_stats])
        story.append(_table_for_report(base_df[[c for c in base_cols if c in base_df.columns]], max_rows=1))
        story.append(Image(str(CHARTS / "iv_tau_overlay_baseline.png"), width=6.4 * inch, height=3.7 * inch))
    else:
        story.append(P("Baseline tau path was not available, so the overlay comparison was skipped."))

    story.append(PageBreak())
    story.append(P("Nested Mean-Covariance Comparison", "H1x"))
    story.append(P("The same four models are compared using one-step Gaussian quasi-log scores in W coordinates. Dynamic covariance wins if M1-M0 is positive and M2-M0 is not positive."))
    nested_cols = ["comparison", "avg_log_score_diff", "p05", "p95", "prob_diff_gt_0", "sum_log_score_diff"]
    story.append(_table_for_report(nested["comparison"][[c for c in nested_cols if c in nested["comparison"].columns]], max_rows=6))
    story.append(Image(str(CHARTS / "iv_nested_log_score_bar.png"), width=5.9 * inch, height=3.7 * inch))
    story.append(Image(str(CHARTS / "iv_a_survival_trace_overlay.png"), width=6.4 * inch, height=3.7 * inch))

    story.append(P("Full-Pipeline Bootstrap", "H1x"))
    story.append(P("Each bootstrap draw resamples valid rows in circular moving blocks, re-estimates residualization, the first stage, IV LP coefficients, the eigensystem, alpha, state-space parameters, and the tau path."))
    boot_cols = ["draw", "alpha_hat", "retained_trace_share", "max_tau", "max_month", "first_stage_f_stat", "q_zx", "weak_iv_warning"]
    story.append(_table_for_report(bootstrap["boot_df"][[c for c in boot_cols if c in bootstrap["boot_df"].columns]], max_rows=8))
    story.append(Image(str(CHARTS / "iv_bootstrap_tau_bands.png"), width=6.4 * inch, height=3.7 * inch))

    story.append(P("Caveats", "H1x"))
    caveats = [
        "IV-OVK estimates time variation in the covariance of IV-LP response-score surfaces, not time-varying structural IRFs.",
        "IV score spikes can reflect response residuals, instrument leverage, or first-stage strength.",
        "Therefore first-stage diagnostics are part of the result; weak first stages are reported rather than hidden.",
    ]
    for caveat in caveats:
        story.append(P("- " + caveat))

    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(FINAL_PDF),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    doc.build(story)


def _df_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p><em>No rows.</em></p>"
    return df.head(max_rows).to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)


def build_html_report(
    sample_summary: pd.DataFrame,
    first_stage: pd.DataFrame,
    avg_resp: pd.DataFrame,
    rank_df: pd.DataFrame,
    top_df: pd.DataFrame,
    decomposition: dict[str, Any],
    exact_decomposition: dict[str, Any],
    driver_diagnostic: dict[str, Any],
    nested: dict[str, Any],
    baseline_stats: dict[str, Any],
) -> None:
    def img(name: str) -> str:
        path = os.path.relpath(CHARTS / name, FINAL_HTML.parent)
        return f'<img src="{html.escape(path)}" style="max-width:100%; margin:10px 0;">'

    def figure_img(path: Path) -> str:
        rel = os.path.relpath(path, FINAL_HTML.parent)
        return f'<img src="{html.escape(rel)}" style="max-width:100%; margin:10px 0;">'

    complete_range = _sample_value(sample_summary, "Common complete-coordinate score coverage")
    raw_window = _sample_value(sample_summary, "Raw outcome window consumed by score sample")
    denom_sample = _sample_value(sample_summary, "First-stage denominator sample")
    sample_note = html.escape(
        "The state index is attached to the base month t. "
        f"In this run the common complete-coordinate score coverage is {complete_range}; "
        f"the LP construction consumes raw outcome data over {raw_window}. "
        f"{denom_sample}, so the 125 coordinates share the same scalar proxy-IV first-stage denominator."
    )

    body = f"""
<html><head><meta charset="utf-8"><title>Proxy-IV LP/OVK Appendix</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.4; color: #222; }}
table {{ border-collapse: collapse; font-size: 12px; margin: 10px 0 18px 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: left; }}
th {{ background: #eee; }}
h1 {{ font-size: 24px; }}
h2 {{ margin-top: 28px; }}
</style></head><body>
<h1>Proxy-IV LP/OVK Appendix</h1>
<p>This appendix constructs LP score surfaces from 2SLS/external-instrument influence contributions. The endogenous policy indicator is DGS1 and the excluded instrument is the SF Fed monetary-policy surprise.</p>
<p><strong>Caveat:</strong> this is time-varying covariance of IV-LP score surfaces, not an unrestricted time-varying structural IRF estimate.</p>
<h2>Sample and Data</h2><p>{sample_note}</p>{_df_html(sample_summary, 30)}{_df_html(pd.read_csv(TABLES / "iv_data_sources.csv"), 10)}
<h2>First Stage</h2>{_df_html(first_stage, 5)}{img("iv_first_stage_scatter.png")}
<h2>Average IV LP Responses</h2>{img("iv_average_lp_responses.png")}{_df_html(avg_resp, 30)}
<h2>OVK Geometry</h2>{_df_html(rank_df, 10)}{img("iv_eigenspectrum.png")}{img("iv_tau_path.png")}{img("iv_top5_basis_surfaces.png")}
<h2>Score-Energy Decomposition</h2>{_df_html(decomposition["summary"], 5)}{img("iv_score_decomposition_top_tau.png")}{_df_html(decomposition["top_tau"], 12)}
<h2>Multiplicative Tau Drivers</h2><p>The proxy-IV full-coordinate score is chi_proxy[r]=(M_r/kappa_hat)u_r. With the fitted linear temporal-kernel covariance smoother, tau_soft[t] equals the weighted average of E_r R_r and is plotted as the exact product of a smoothed proxy exposure factor and an exposure-weighted residual-energy factor.</p>{_df_html(driver_diagnostic["diagnostics_df"], 1)}{figure_img(driver_diagnostic["figure_png"])}{_df_html(driver_diagnostic["path"], 12)}
<h2>Baseline Comparison</h2>{_df_html(pd.DataFrame([baseline_stats]), 3)}{img("iv_tau_overlay_baseline.png") if baseline_stats.get("baseline_available") else ""}
<h2>Nested Mean-Covariance</h2>{_df_html(nested["comparison"], 10)}{img("iv_nested_log_score_bar.png")}{img("iv_a_survival_trace_overlay.png")}
<h2>Top Amplification Months</h2>{_df_html(top_df, 20)}
</body></html>
"""
    FINAL_HTML.parent.mkdir(parents=True, exist_ok=True)
    FINAL_HTML.write_text(body, encoding="utf-8")


def build_bundle() -> None:
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root in [IV_ROOT]:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    z.write(f, arcname=str(Path("iv_ovk") / f.relative_to(IV_ROOT)))
        for f in [FINAL_PDF, FINAL_HTML, PROCESSED_PANEL]:
            if f.exists():
                z.write(f, arcname=f.name if f.parent == REPORTS else str(f.relative_to(BASE_DIR) if f.is_relative_to(BASE_DIR) else f.name))


def write_skip(reason: str) -> Path:
    path = IV_ROOT / "iv_skipped.json"
    path.write_text(json.dumps({"created_at": _now_iso(), "reason": reason}, indent=2), encoding="utf-8")
    return path


def run_iv_appendix(allow_download: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    stale_skip = IV_ROOT / "iv_skipped.json"
    if stale_skip.exists():
        stale_skip.unlink()
    panel, merge_meta = build_iv_policy_panel(allow_download=allow_download)
    sources_json = IV_RAW_DIR / "iv_data_sources.json"
    if sources_json.exists():
        write_sources_csv(sources_json, TABLES / "iv_data_sources.csv")
    else:
        pd.DataFrame(merge_meta.get("data_sources", {}).get("sources", [])).to_csv(TABLES / "iv_data_sources.csv", index=False)

    scores = build_iv_lp_scores(panel, outcome_columns=SPEC.outcome_columns)
    sample_summary = write_sample_summary(panel, scores, merge_meta)
    first_stage, _ = write_first_stage_outputs(scores)
    avg_resp = write_average_iv_responses(scores)
    rank_results = run_rank_models(scores)
    rank_df, tau_df = write_rank_and_tau_outputs(rank_results, scores)
    headline = rank_results[HEADLINE_R]
    decomposition = write_score_energy_decomposition(scores, headline)
    exact_decomposition = run_iv_tau_exact_decomposition(scores, headline)
    driver_diagnostic = run_iv_tau_multiplicative_driver_diagnostic(scores, headline)
    baseline_stats = write_baseline_overlay(headline)
    nested = run_nested_comparison_iv(scores)
    bootstrap = run_iv_bootstrap(scores, headline)

    top_df = pd.read_csv(TABLES / "iv_top_amplification_months.csv")
    state_draws = int(os.environ.get("OVK_PUBLICATION_STATE_DRAWS", "0"))
    if state_draws > 0:
        try:
            paths = ffbs_state_draws(headline.fit, min(state_draws, 250), int(os.environ.get("OVK_PUBLICATION_STATE_SEED", "9127")))
            tau_draws, _, _, _ = state_draw_scale_shape(
                paths,
                HEADLINE_R,
                estimator_mode=getattr(headline, "estimator_mode", "log_spd_legacy"),
            )
            state_band = positive_simultaneous_band(tau_draws, headline.tau, level=0.90)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(scores["dates"]).dt.strftime("%Y-%m-%d"),
                    "tau_state_p05": state_band["point_low"],
                    "tau_state_p50": state_band["point_med"],
                    "tau_state_p95": state_band["point_high"],
                }
            ).to_csv(TABLES / "iv_state_smoother_tau_bands.csv", index=False)
        except Exception as exc:
            (IV_ROOT / "iv_state_smoother_skipped.json").write_text(json.dumps({"reason": str(exc)}, indent=2), encoding="utf-8")

    build_pdf_report(
        sample_summary,
        first_stage,
        avg_resp,
        rank_df,
        top_df,
        decomposition,
        exact_decomposition,
        driver_diagnostic,
        nested,
        baseline_stats,
        bootstrap,
        merge_meta,
    )
    build_html_report(
        sample_summary,
        first_stage,
        avg_resp,
        rank_df,
        top_df,
        decomposition,
        exact_decomposition,
        driver_diagnostic,
        nested,
        baseline_stats,
    )
    shutil.copy2(Path(__file__), CODE_OUT / "iv_ovk.py")
    shutil.copy2(CODE_ROOT / "download_iv_data.py", CODE_OUT / "download_iv_data.py")
    if (CODE_ROOT / "ovk_data.py").exists():
        shutil.copy2(CODE_ROOT / "ovk_data.py", CODE_OUT / "ovk_data.py")
    build_bundle()
    manifest = write_file_manifest()
    elapsed = time.perf_counter() - t0
    metadata = {
        "created_at": _now_iso(),
        "elapsed_seconds": round(elapsed, 3),
        "variant": SPEC.__dict__,
        "H": H,
        "L": L,
        "rank": HEADLINE_R,
        "ranks": list(rank_results.keys()),
        "iv_bootstrap_draws": IV_BOOT_DRAWS,
        "iv_nested_bootstrap_draws": IV_NESTED_BOOT_DRAWS,
        "first_stage": scores["first_stage"],
        "sample_coverage": dict(zip(sample_summary["item"].astype(str), sample_summary["value"].astype(str))),
        "baseline_comparison": baseline_stats,
        "score_energy_decomposition": decomposition["summary"].iloc[0].to_dict(),
        "exact_tau_decomposition": exact_decomposition["diagnostics"],
        "multiplicative_driver_diagnostic": driver_diagnostic["diagnostics"],
        "outputs": {
            "iv_root": str(IV_ROOT),
            "processed_panel": str(PROCESSED_PANEL),
            "pdf": str(FINAL_PDF),
            "html": str(FINAL_HTML),
            "zip": str(FINAL_ZIP),
            "manifest": str(TABLES / "iv_file_manifest.csv"),
            "score_energy_decomposition_path": str(TABLES / "iv_score_energy_decomposition_path.csv"),
            "top_tau_decomposition": str(TABLES / "iv_top_tau_decomposition.csv"),
            "score_decomposition_summary": str(TABLES / "iv_score_decomposition_summary.csv"),
            "score_decomposition_chart": str(CHARTS / "iv_score_decomposition_top_tau.png"),
            "exact_tau_decomposition_table": str(exact_decomposition["table_path"]),
            "exact_tau_decomposition_diagnostics": str(exact_decomposition["diagnostics_path"]),
            "exact_tau_decomposition_png": str(exact_decomposition["figure_png"]),
            "exact_tau_decomposition_pdf": str(exact_decomposition["figure_pdf"]),
            "exact_tau_decomposition_excess_png": str(exact_decomposition["excess_figure_png"]),
            "exact_tau_decomposition_excess_pdf": str(exact_decomposition["excess_figure_pdf"]),
            "multiplicative_driver_table": str(driver_diagnostic["table_path"]),
            "multiplicative_driver_audit": str(driver_diagnostic["audit_path"]),
            "multiplicative_driver_top_months": str(driver_diagnostic["top_months_path"]),
            "multiplicative_driver_episodes": str(driver_diagnostic["episodes_path"]),
            "multiplicative_driver_diagnostics": str(driver_diagnostic["diagnostics_path"]),
            "multiplicative_driver_png": str(driver_diagnostic["figure_png"]),
            "multiplicative_driver_pdf": str(driver_diagnostic["figure_pdf"]),
            "multiplicative_driver_heatmap_png": str(driver_diagnostic["heatmap_png"]),
            "multiplicative_driver_heatmap_pdf": str(driver_diagnostic["heatmap_pdf"]),
            "multiplicative_driver_caption": str(driver_diagnostic["caption_path"]),
        },
    }
    (IV_ROOT / "iv_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {
        "metadata": metadata,
        "scores": scores,
        "rank_results": rank_results,
        "rank_df": rank_df,
        "tau_df": tau_df,
        "nested": nested,
        "bootstrap": bootstrap,
        "decomposition": decomposition,
        "exact_decomposition": exact_decomposition,
        "driver_diagnostic": driver_diagnostic,
        "manifest": manifest,
    }


def _comparison_value(table: pd.DataFrame, comparison: str) -> float:
    rows = table.loc[table["comparison"].eq(comparison), "avg_log_score_diff"]
    return float(rows.iloc[0]) if len(rows) else np.nan


def main() -> int:
    try:
        result = run_iv_appendix(allow_download=True)
    except Exception as exc:
        path = write_skip(str(exc))
        print(f"IV OVK skipped: {exc}")
        print(f"skip_metadata: {path}")
        return 0

    scores = result["scores"]
    rank_results = result["rank_results"]
    headline = rank_results[HEADLINE_R]
    nested = result["nested"]["comparison"]
    decomp = result["decomposition"]["summary"].iloc[0]
    exact = result["exact_decomposition"]["diagnostics"]
    print("IV OVK complete")
    print(f"downloaded_iv_data_dir: {IV_RAW_DIR}")
    print(f"sf_fed_file: {Path(scores.get('z_source_file', '')) if scores.get('z_source_file') else (IV_RAW_DIR / 'sf_fed_monetary_policy_surprises.xlsx')}")
    print(f"dgs1_monthly_file: {IV_RAW_DIR / 'dgs1_monthly_eom.csv'}")
    print(f"iv_first_stage_f_stat: {scores['first_stage']['first_stage_f_stat']:.6g}")
    print(f"iv_top5_trace_share: {float(headline.shares[:HEADLINE_R].sum()):.6g}")
    print(f"iv_max_tau_month: {pd.to_datetime(headline.dates.iloc[int(np.argmax(headline.tau))]).strftime('%Y-%m')}")
    print(f"iv_max_tau_value: {float(np.max(headline.tau)):.6g}")
    print(f"iv_nested_M1_minus_M0: {_comparison_value(nested, 'M1 - M0'):.6g}")
    print(f"iv_nested_M2_minus_M0: {_comparison_value(nested, 'M2 - M0'):.6g}")
    print(f"iv_nested_M3_minus_M1: {_comparison_value(nested, 'M3 - M1'):.6g}")
    print(f"max_abs_diff_between_z_score_and_fitted_score_forms: {float(decomp['max_abs_diff_between_z_score_and_fitted_score_forms']):.6g}")
    print(f"driver_label_for_max_tau_month: {decomp['driver_label_for_max_tau_month']}")
    print(f"top10_first_stage_driven_count: {int(decomp['top10_first_stage_driven_count'])}")
    print(f"top10_residual_driven_count: {int(decomp['top10_residual_driven_count'])}")
    print(f"top10_mixed_count: {int(decomp['top10_mixed_count'])}")
    print(f"corr_tau_log_first_stage_leverage: {float(decomp['corr_tau_log_first_stage_leverage']):.6g}")
    print(f"corr_tau_log_retained_residual_energy: {float(decomp['corr_tau_log_retained_residual_energy']):.6g}")
    print(f"iv_score_energy_decomposition_path_csv: {TABLES / 'iv_score_energy_decomposition_path.csv'}")
    print(f"iv_top_tau_decomposition_csv: {TABLES / 'iv_top_tau_decomposition.csv'}")
    print(f"iv_score_decomposition_summary_csv: {TABLES / 'iv_score_decomposition_summary.csv'}")
    print(f"iv_score_decomposition_top_tau_png: {CHARTS / 'iv_score_decomposition_top_tau.png'}")
    print("Exact IV tau decomposition debug/fix complete.")
    print(f"Saved table: {TABLES / 'iv_tau_exact_decomposition_timeseries.csv'}")
    print(f"Saved diagnostics: {TABLES / 'iv_tau_exact_decomposition_diagnostics.json'}")
    print(f"Saved figure: {FIGURES / 'iv_tau_exact_decomposition_area.png'}")
    print(f"Saved excess diagnostic figure: {FIGURES / 'iv_tau_exact_decomposition_excess_area.png'}")
    print(f"Max additive score error: {float(exact['max_abs_score_error']):.6g}")
    print(f"Relative Frobenius score error: {float(exact['rel_fro_score_error']):.6g}")
    print(f"Raw existing score tau mean: {float(exact['raw_tau_existing_mean']):.6g}")
    print(f"Raw augmented implied tau mean: {float(exact['raw_aug_tau_bar']):.6g}")
    print(f"Corrected augmented total tau mean: {float(exact['mean_augmented_total_tau']):.6g}")
    if np.isfinite(float(exact.get("mean_existing_tau", np.nan))):
        print(f"Existing standalone tau mean: {float(exact['mean_existing_tau']):.6g}")
    print(f"Max tau sum error: {float(exact['max_abs_tau_sum_error']):.6g}")
    if np.isfinite(float(exact.get("tau_existing_corr", np.nan))):
        print(f"Correlation with existing standalone tau: {float(exact['tau_existing_corr']):.6g}")
    print(f"iv_report_pdf: {FINAL_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
