#!/usr/bin/env python3
"""
Single CLI entrypoint for the monthly monetary-policy OVK results pipeline.

The headline result is the rank-five response-score covariance state-space algorithm. The nested
mean/covariance workflow is run as a robustness check against omitted
time-varying mean IRFs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ovk_data import merge_extra_outcome_data, outcome_columns_for_panel, outcome_labels_for_panel


REQUIRED_ZIP_MEMBERS = {
    "data/fred_macro_monthly.csv",
    "data/shocks_fed_jk_m.csv",
    "data/shocks_fed_jk_t.csv",
}
REQUIRED_FRED_COLUMNS = {"date", "ip", "cpi", "unrate", "gs2", "baa10y"}
REQUIRED_MONTHLY_SHOCK_COLUMNS = {
    "year",
    "month",
    "pc1_hf",
    "SP500_hf",
    "MP_pm",
    "CBI_pm",
    "MP_median",
    "CBI_median",
}
REQUIRED_EVENT_SHOCK_COLUMNS = {
    "start",
    "MP_pm",
    "CBI_pm",
    "MP_median",
    "CBI_median",
}
PIPELINE_STEP_CACHE_VERSION = "pipeline_log_tau_bands_v1"


@dataclass
class PipelineConfig:
    workspace: Path
    code_dir: Path
    data_zip: Path
    out_dir: Path
    overwrite: bool
    include_math_appendix: bool
    workers: int | None
    sec_bootstrap_draws: int
    cache_dir: Path
    disable_cache: bool
    benchmark_workers: bool
    headline_outcomes: str
    sf_fed_surprises: Path
    python_executable: str


@dataclass
class PipelinePaths:
    data_raw: Path
    data_processed: Path
    top5_root: Path
    robustness_root: Path
    nested_root: Path
    iv_root: Path
    reports: Path
    logs: Path
    full_pack: Path
    math_pack: Path
    pdf_render: Path
    sec_root: Path
    sec_final_pdf: Path
    sec_final_zip: Path
    sec_contact: Path
    iv_final_pdf: Path
    iv_final_html: Path
    iv_final_zip: Path
    manifest: Path
    metadata: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _safe_remove_output_dir(out_dir: Path, workspace: Path) -> None:
    out_dir = out_dir.resolve()
    workspace = workspace.resolve()
    blocked = {
        workspace,
        workspace / "code",
        workspace / "data_raw",
        workspace / "data_processed",
        workspace / "tables",
        workspace / "charts",
        out_dir.anchor and Path(out_dir.anchor),
    }
    if out_dir in blocked:
        raise RuntimeError(f"Refusing to remove unsafe output directory: {out_dir}")
    if len(out_dir.parts) <= 2:
        raise RuntimeError(f"Refusing to remove shallow output directory: {out_dir}")
    shutil.rmtree(out_dir)


def _paths(config: PipelineConfig) -> PipelinePaths:
    return PipelinePaths(
        data_raw=config.out_dir / "data_raw",
        data_processed=config.out_dir / "data_processed",
        top5_root=config.out_dir / "top5_headline",
        robustness_root=config.out_dir / "top5_shock_robustness",
        nested_root=config.out_dir / "nested_mean_cov_robustness",
        iv_root=config.out_dir / "iv_ovk",
        reports=config.out_dir / "reports",
        logs=config.out_dir / "logs",
        full_pack=config.out_dir / "top5_full_appended_results_pack",
        math_pack=config.out_dir / "top5_full_math_results_pack",
        pdf_render=config.out_dir / "pdf_render",
        sec_root=config.workspace / "sec_robustness_results",
        sec_final_pdf=config.workspace / "monthly_ovk_top5_with_SEC_robustness_report.pdf",
        sec_final_zip=config.workspace / "monthly_ovk_top5_with_SEC_robustness_full_pack.zip",
        sec_contact=config.workspace / "monthly_ovk_top5_with_SEC_robustness_contact_sheet.jpg",
        iv_final_pdf=config.out_dir / "reports" / "iv_ovk_report.pdf",
        iv_final_html=config.out_dir / "reports" / "iv_ovk_report.html",
        iv_final_zip=config.out_dir / "reports" / "iv_ovk_bundle.zip",
        manifest=config.out_dir / "manifest.csv",
        metadata=config.out_dir / "run_metadata.json",
    )


def _prepare_output_dir(config: PipelineConfig, paths: PipelinePaths) -> None:
    if config.out_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output directory exists; pass --overwrite to replace it: {config.out_dir}")
        _safe_remove_output_dir(config.out_dir, config.workspace)
    for path in [
        config.out_dir,
        paths.data_raw,
        paths.data_processed,
        paths.reports,
        paths.logs,
        paths.pdf_render,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def prepare_shared_data(config: PipelineConfig, paths: PipelinePaths) -> dict[str, Any]:
    if not config.data_zip.exists():
        raise FileNotFoundError(f"Data zip not found: {config.data_zip}")

    extracted = paths.data_raw / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.data_zip, paths.data_raw / "data.zip")

    with zipfile.ZipFile(config.data_zip) as zf:
        names = {name.replace("\\", "/") for name in zf.namelist()}
        missing = sorted(REQUIRED_ZIP_MEMBERS - names)
        if missing:
            raise ValueError(f"Data zip is missing required files: {', '.join(missing)}")
        zf.extractall(extracted)

    data_dir = extracted / "data"
    fred = pd.read_csv(data_dir / "fred_macro_monthly.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    monthly = pd.read_csv(data_dir / "shocks_fed_jk_m.csv")
    events = pd.read_csv(data_dir / "shocks_fed_jk_t.csv")
    _validate_columns(fred, REQUIRED_FRED_COLUMNS, "fred_macro_monthly.csv")
    _validate_columns(monthly, REQUIRED_MONTHLY_SHOCK_COLUMNS, "shocks_fed_jk_m.csv")
    _validate_columns(events, REQUIRED_EVENT_SHOCK_COLUMNS, "shocks_fed_jk_t.csv")
    fred = merge_extra_outcome_data(fred, data_dir=data_dir, balanced=True)
    outcome_columns = outcome_columns_for_panel(fred)
    outcome_labels = outcome_labels_for_panel(fred)

    monthly["date"] = pd.to_datetime(
        dict(year=monthly["year"].astype(int), month=monthly["month"].astype(int), day=1)
    )
    monthly["MP_used"] = monthly["MP_median"].fillna(monthly["MP_pm"])
    monthly["CBI_used"] = monthly["CBI_median"].fillna(monthly["CBI_pm"])
    monthly["used_pm_fallback"] = monthly["MP_median"].isna() | monthly["CBI_median"].isna()
    monthly["MP_median_fallback"] = monthly["MP_used"]
    monthly["CBI_median_fallback"] = monthly["CBI_used"]
    monthly["fallback_flag"] = monthly["used_pm_fallback"]

    shock_cols = [
        "date",
        "year",
        "month",
        "pc1_hf",
        "SP500_hf",
        "MP_pm",
        "CBI_pm",
        "MP_median",
        "CBI_median",
        "MP_used",
        "CBI_used",
        "used_pm_fallback",
    ]
    panel_full = fred.merge(monthly[shock_cols], on="date", how="left").sort_values("date").reset_index(drop=True)
    panel_overlap = (
        fred.merge(monthly[shock_cols], on="date", how="inner")
        .dropna(subset=outcome_columns + ["MP_used", "CBI_used"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    events["datetime"] = pd.to_datetime(events["start"])
    events["date"] = events["datetime"].dt.to_period("M").dt.to_timestamp()
    events["MP_event_used"] = events["MP_median"].fillna(events["MP_pm"])
    events["CBI_event_used"] = events["CBI_median"].fillna(events["CBI_pm"])
    events["fallback_event"] = events["MP_median"].isna() | events["CBI_median"].isna()
    event_agg = events.groupby("date", as_index=False).agg(
        MP_event_manual=("MP_event_used", "sum"),
        CBI_event_manual=("CBI_event_used", "sum"),
        MP_pm_sum=("MP_pm", "sum"),
        CBI_pm_sum=("CBI_pm", "sum"),
        fallback_event=("fallback_event", "max"),
        n_events=("MP_event_used", "size"),
        nonmissing_median_events=("MP_median", lambda x: int(x.notna().sum())),
        missing_median_events=("MP_median", lambda x: int(x.isna().sum())),
    )
    month_range = pd.DataFrame({"date": pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="MS")})
    event_monthly = month_range.merge(event_agg, on="date", how="left")
    fill_cols = [
        "MP_event_manual",
        "CBI_event_manual",
        "MP_pm_sum",
        "CBI_pm_sum",
        "n_events",
        "nonmissing_median_events",
        "missing_median_events",
    ]
    event_monthly[fill_cols] = event_monthly[fill_cols].fillna(0.0)
    event_monthly["fallback_event"] = event_monthly["fallback_event"].fillna(False).astype(bool)
    event_monthly["mixed_missing_and_nonmissing_events"] = (
        (event_monthly["missing_median_events"] > 0)
        & (event_monthly["nonmissing_median_events"] > 0)
    )

    panel_three = fred.merge(
        monthly[
            [
                "date",
                "MP_median_fallback",
                "CBI_median_fallback",
                "MP_pm",
                "CBI_pm",
                "fallback_flag",
            ]
        ],
        on="date",
        how="left",
    ).merge(
        event_monthly[
            [
                "date",
                "MP_event_manual",
                "CBI_event_manual",
                "n_events",
                "fallback_event",
                "mixed_missing_and_nonmissing_events",
            ]
        ],
        on="date",
        how="left",
    ).sort_values("date").reset_index(drop=True)

    panel_full.to_csv(paths.data_processed / "ovk_monetary_panel_monthly_fixed_full.csv", index=False)
    panel_overlap.to_csv(paths.data_processed / "ovk_monetary_panel_monthly_fixed_overlap.csv", index=False)
    panel_three.to_csv(paths.data_processed / "processed_panel_three_shock_definitions.csv", index=False)
    monthly.to_csv(paths.data_processed / "monthly_shocks_repaired.csv", index=False)
    events.to_csv(paths.data_processed / "event_shocks_with_manual_fields.csv", index=False)
    event_monthly.to_csv(paths.data_processed / "event_level_manual_monthly_aggregation.csv", index=False)

    return {
        "fred_rows": int(len(fred)),
        "monthly_shock_rows": int(len(monthly)),
        "event_shock_rows": int(len(events)),
        "full_panel_rows": int(len(panel_full)),
        "overlap_panel_rows": int(len(panel_overlap)),
        "outcome_columns": outcome_columns,
        "outcome_labels": outcome_labels,
        "data_processed_dir": str(paths.data_processed),
    }


def _pipeline_env(config: PipelineConfig, paths: PipelinePaths) -> dict[str, str]:
    env = os.environ.copy()
    reports = paths.reports
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "OVK_DATA_ZIP": str(config.data_zip),
            "OVK_BASE_DIR": str(config.out_dir),
            "OVK_REPORTS_DIR": str(reports),
            "OVK_GENERATE_SCRIPT": str(config.code_dir / "generate_top5_ovk_pack.py"),
            "OVK_FINALIZE_SCRIPT": str(config.code_dir / "finalize_top5_from_tables.py"),
            "OVK_ROBUST_SCRIPT": str(config.code_dir / "run_top5_robustness_point_estimates.py"),
            "OVK_FINAL_PACK_SCRIPT": str(config.code_dir / "make_final_appended_top5_pack.py"),
            "OVK_MATH_SCRIPT": str(config.code_dir / "create_full_math_appendix_pack.py"),
            "OVK_PUBLICATION_ROOT": str(config.out_dir / "publication_grade_ovk"),
            "OVK_PUBLICATION_FINAL_PDF": str(reports / "publication_grade_ovk_report.pdf"),
            "OVK_PUBLICATION_FINAL_HTML": str(reports / "publication_grade_ovk_report.html"),
            "OVK_PUBLICATION_FINAL_ZIP": str(reports / "publication_grade_ovk_bundle.zip"),
            "OVK_PUBLICATION_WORKERS": str(config.workers or ""),
            "OVK_PUBLICATION_BENCHMARK_WORKERS": "1" if config.benchmark_workers else "0",
            "OVK_HEADLINE_OUTCOMES": config.headline_outcomes,
            "OVK_SF_FED_SURPRISES": str(config.sf_fed_surprises),
            "OVK_CACHE_DIR": str(config.cache_dir),
            "OVK_DISABLE_CACHE": "1" if config.disable_cache else "0",
            "OVK_TOP5_COMPUTE_ONLY": "1",
            "OVK_TOP5_ROOT": str(paths.top5_root),
            "OVK_TOP5_OUT": str(paths.top5_root / "outputs"),
            "OVK_TOP5_FINAL_PDF": str(reports / "top5_headline_report.pdf"),
            "OVK_TOP5_FINAL_HTML": str(reports / "top5_headline_report.html"),
            "OVK_TOP5_FINAL_ZIP": str(reports / "top5_headline_bundle.zip"),
            "OVK_TOP5_FINAL_CONTACT": str(reports / "top5_headline_contact_sheet.jpg"),
            "OVK_ROBUST_ROOT": str(paths.robustness_root),
            "OVK_ROBUST_OUT": str(paths.robustness_root / "outputs"),
            "OVK_FULL_PACK": str(paths.full_pack),
            "OVK_FULL_FINAL_PDF": str(reports / "top5_full_appended_report.pdf"),
            "OVK_FULL_FINAL_HTML": str(reports / "top5_full_appended_report.html"),
            "OVK_FULL_FINAL_ZIP": str(reports / "top5_full_appended_results_pack.zip"),
            "OVK_FULL_FINAL_CONTACT": str(reports / "top5_full_appended_contact_sheet.jpg"),
            "OVK_NESTED_PANEL_PATH": str(paths.data_processed / "ovk_monetary_panel_monthly_fixed_full.csv"),
            "OVK_NESTED_OUTDIR": str(paths.nested_root),
            "OVK_NESTED_FINAL_PDF": str(reports / "nested_mean_cov_robustness_report.pdf"),
            "OVK_NESTED_FINAL_ZIP": str(reports / "nested_mean_cov_robustness_bundle.zip"),
            "OVK_NESTED_REPORT_TITLE": "Robustness: Nested Mean-Covariance LP/OVK Workflow",
            "OVK_NESTED_REPORT_SUBTITLE": (
                "Robustness check for omitted time-varying mean IRFs; headline result remains "
                "the rank-five dynamic covariance operator of LP score surfaces."
            ),
            "OVK_IV_ROOT": str(paths.iv_root),
            "OVK_IV_RAW_DIR": str(config.workspace / "data_raw" / "external" / "iv"),
            "OVK_IV_PANEL_PATH": str(paths.data_processed / "iv_proxy_policy_panel.csv"),
            "OVK_IV_BASELINE_PANEL": str(paths.data_processed / "processed_panel_three_shock_definitions.csv"),
            "OVK_IV_FINAL_PDF": str(paths.iv_final_pdf),
            "OVK_IV_FINAL_HTML": str(paths.iv_final_html),
            "OVK_IV_FINAL_ZIP": str(paths.iv_final_zip),
            "OVK_MATH_PACK": str(paths.math_pack),
            "OVK_MATH_FINAL_PDF": str(reports / "top5_full_appended_with_math_report.pdf"),
            "OVK_MATH_FINAL_ZIP": str(reports / "top5_full_appended_with_math_results_pack.zip"),
            "OVK_MATH_APPENDIX_PDF": str(reports / "top5_math_appendix.pdf"),
            "OVK_MATH_CONTACT": str(reports / "top5_full_appended_with_math_contact_sheet.jpg"),
            "OVK_MATH_RENDER_DIR": str(paths.pdf_render / "math"),
        }
    )
    return env


def _copy_artifact(src: Path, dst: Path, workspace: Path) -> None:
    if not src.exists():
        return
    dst = dst.resolve()
    workspace = workspace.resolve()
    if not dst.is_relative_to(workspace):
        raise RuntimeError(f"Refusing to copy artifact outside workspace: {dst}")
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _step_artifact_specs(name: str, paths: PipelinePaths) -> list[tuple[Path, Path]]:
    reports = paths.reports
    specs: dict[str, list[tuple[Path, Path]]] = {
        "01_publication_grade_ovk": [
            (paths.reports.parent / "publication_grade_ovk", Path("publication_grade_ovk")),
            (paths.top5_root, Path("top5_headline")),
            (paths.robustness_root, Path("top5_shock_robustness")),
            (reports / "publication_grade_ovk_report.pdf", Path("reports/publication_grade_ovk_report.pdf")),
            (reports / "publication_grade_ovk_report.html", Path("reports/publication_grade_ovk_report.html")),
            (reports / "publication_grade_ovk_bundle.zip", Path("reports/publication_grade_ovk_bundle.zip")),
            (reports / "top5_headline_report.pdf", Path("reports/top5_headline_report.pdf")),
            (reports / "top5_headline_report.html", Path("reports/top5_headline_report.html")),
            (reports / "top5_headline_bundle.zip", Path("reports/top5_headline_bundle.zip")),
        ],
        "01_top5_headline_generate": [
            (paths.top5_root, Path("top5_headline")),
        ],
        "02_top5_headline_finalize": [
            (paths.top5_root, Path("top5_headline")),
            (reports / "top5_headline_report.pdf", Path("reports/top5_headline_report.pdf")),
            (reports / "top5_headline_report.html", Path("reports/top5_headline_report.html")),
            (reports / "top5_headline_bundle.zip", Path("reports/top5_headline_bundle.zip")),
        ],
        "03_top5_shock_robustness": [
            (paths.robustness_root, Path("top5_shock_robustness")),
        ],
        "04_top5_full_appended_pack": [
            (paths.full_pack, Path("top5_full_appended_results_pack")),
            (reports / "top5_full_appended_report.pdf", Path("reports/top5_full_appended_report.pdf")),
            (reports / "top5_full_appended_report.html", Path("reports/top5_full_appended_report.html")),
            (reports / "top5_full_appended_results_pack.zip", Path("reports/top5_full_appended_results_pack.zip")),
        ],
        "02_top5_full_appended_pack": [
            (paths.full_pack, Path("top5_full_appended_results_pack")),
            (reports / "top5_full_appended_report.pdf", Path("reports/top5_full_appended_report.pdf")),
            (reports / "top5_full_appended_report.html", Path("reports/top5_full_appended_report.html")),
            (reports / "top5_full_appended_results_pack.zip", Path("reports/top5_full_appended_results_pack.zip")),
        ],
        "05_publication_grade_ovk": [
            (paths.reports.parent / "publication_grade_ovk", Path("publication_grade_ovk")),
            (reports / "publication_grade_ovk_report.pdf", Path("reports/publication_grade_ovk_report.pdf")),
            (reports / "publication_grade_ovk_report.html", Path("reports/publication_grade_ovk_report.html")),
            (reports / "publication_grade_ovk_bundle.zip", Path("reports/publication_grade_ovk_bundle.zip")),
        ],
        "06_nested_mean_cov_robustness": [
            (paths.nested_root, Path("nested_mean_cov_robustness")),
            (reports / "nested_mean_cov_robustness_report.pdf", Path("reports/nested_mean_cov_robustness_report.pdf")),
            (reports / "nested_mean_cov_robustness_bundle.zip", Path("reports/nested_mean_cov_robustness_bundle.zip")),
        ],
        "03_nested_mean_cov_robustness": [
            (paths.nested_root, Path("nested_mean_cov_robustness")),
            (reports / "nested_mean_cov_robustness_report.pdf", Path("reports/nested_mean_cov_robustness_report.pdf")),
            (reports / "nested_mean_cov_robustness_bundle.zip", Path("reports/nested_mean_cov_robustness_bundle.zip")),
        ],
        "04_sec_robustness": [
            (paths.sec_root, Path("sec_robustness_results")),
            (paths.sec_final_pdf, Path("monthly_ovk_top5_with_SEC_robustness_report.pdf")),
            (paths.sec_final_zip, Path("monthly_ovk_top5_with_SEC_robustness_full_pack.zip")),
            (paths.sec_contact, Path("monthly_ovk_top5_with_SEC_robustness_contact_sheet.jpg")),
        ],
        "07_math_appendix": [
            (paths.math_pack, Path("top5_full_math_results_pack")),
            (reports / "top5_full_appended_with_math_report.pdf", Path("reports/top5_full_appended_with_math_report.pdf")),
            (reports / "top5_full_appended_with_math_results_pack.zip", Path("reports/top5_full_appended_with_math_results_pack.zip")),
            (reports / "top5_math_appendix.pdf", Path("reports/top5_math_appendix.pdf")),
        ],
        "04_math_appendix": [
            (paths.math_pack, Path("top5_full_math_results_pack")),
            (reports / "top5_full_appended_with_math_report.pdf", Path("reports/top5_full_appended_with_math_report.pdf")),
            (reports / "top5_full_appended_with_math_results_pack.zip", Path("reports/top5_full_appended_with_math_results_pack.zip")),
            (reports / "top5_math_appendix.pdf", Path("reports/top5_math_appendix.pdf")),
        ],
        "05_math_appendix": [
            (paths.math_pack, Path("top5_full_math_results_pack")),
            (reports / "top5_full_appended_with_math_report.pdf", Path("reports/top5_full_appended_with_math_report.pdf")),
            (reports / "top5_full_appended_with_math_results_pack.zip", Path("reports/top5_full_appended_with_math_results_pack.zip")),
            (reports / "top5_math_appendix.pdf", Path("reports/top5_math_appendix.pdf")),
        ],
    }
    if name.endswith("iv_proxy_ovk"):
        return [
            (paths.iv_root, Path("iv_ovk")),
            (paths.data_processed / "iv_proxy_policy_panel.csv", Path("data_processed/iv_proxy_policy_panel.csv")),
            (paths.iv_final_pdf, Path("reports/iv_ovk_report.pdf")),
            (paths.iv_final_html, Path("reports/iv_ovk_report.html")),
            (paths.iv_final_zip, Path("reports/iv_ovk_bundle.zip")),
        ]
    return specs.get(name, [])


def _step_cache_key(
    name: str,
    script: Path,
    config: PipelineConfig,
    env: dict[str, str],
    data_hash: str,
    extra_args: list[str] | None = None,
) -> str:
    relevant_env = {
        key: env.get(key, "")
        for key in [
            "OVK_TOP5_COMPUTE_ONLY",
            "OVK_PUBLICATION_RANKS",
            "OVK_PUBLICATION_HEADLINE_R",
            "OVK_PUBLICATION_STATE_DRAWS",
            "OVK_PUBLICATION_BOOTSTRAP_DRAWS",
            "OVK_PUBLICATION_BOOT_BLOCK_LEN",
            "OVK_PUBLICATION_STUDENT_T_DF",
            "OVK_PUBLICATION_MIN_STUDENT_WEIGHT",
            "OVK_PUBLICATION_EM_ITERS",
            "OVK_PUBLICATION_BOOT_EM_ITERS",
            "OVK_PUBLICATION_WORKERS",
            "OVK_PUBLICATION_BENCHMARK_WORKERS",
            "OVK_RUN_IV",
            "OVK_IV_BOOTSTRAP_DRAWS",
            "OVK_IV_NESTED_BOOT_DRAWS",
            "OVK_IV_NESTED_BOOT_BLOCK_LEN",
            "OVK_IV_RAW_DIR",
            "OVK_IV_BASELINE_PANEL",
        ]
    }
    payload = {
        "version": PIPELINE_STEP_CACHE_VERSION,
        "name": name,
        "script_hash": _sha256(script),
        "args": extra_args or [],
        "data_hash": data_hash,
        "sec_bootstrap_draws": config.sec_bootstrap_draws,
        "include_math_appendix": config.include_math_appendix,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "env": relevant_env,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _restore_step_cache(
    cache_entry: Path,
    specs: list[tuple[Path, Path]],
    config: PipelineConfig,
) -> bool:
    if not specs or not cache_entry.exists():
        return False
    for _, rel in specs:
        if not (cache_entry / rel).exists():
            return False
    for target, rel in specs:
        _copy_artifact(cache_entry / rel, target, config.workspace)
    return True


def _save_step_cache(
    cache_entry: Path,
    specs: list[tuple[Path, Path]],
    config: PipelineConfig,
) -> None:
    if not specs:
        return
    tmp = cache_entry.parent / f".tmp_{os.getpid()}_{cache_entry.name[:8]}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for target, rel in specs:
        if not target.exists():
            continue
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.copytree(target, dst)
        else:
            shutil.copy2(target, dst)
    if cache_entry.exists():
        shutil.rmtree(cache_entry)
    shutil.move(str(tmp), str(cache_entry))


def _run_step(
    name: str,
    script: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    env: dict[str, str],
    data_hash: str,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    log_path = paths.logs / f"{name}.log"
    specs = _step_artifact_specs(name, paths)
    cache_entry = config.cache_dir / "pipeline_steps" / _step_cache_key(
        name, script, config, env, data_hash, extra_args
    )
    started = time.perf_counter()
    if not config.disable_cache and _restore_step_cache(cache_entry, specs, config):
        elapsed = time.perf_counter() - started
        log_path.write_text(f"cache hit: restored {name} from {cache_entry}\n", encoding="utf-8")
        status = {
            "name": name,
            "script": str(script),
            "args": extra_args or [],
            "returncode": 0,
            "elapsed_seconds": round(elapsed, 3),
            "log": str(log_path),
            "cache_hit": True,
            "cache_entry": str(cache_entry),
        }
        print(f"restored {name} from cache in {elapsed:.1f}s")
        return status
    cmd = [config.python_executable, str(script), *(extra_args or [])]
    print(f"running {name} ...")
    result = subprocess.run(
        cmd,
        cwd=config.workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(result.stdout, encoding="utf-8")
    status = {
        "name": name,
        "script": str(script),
        "args": extra_args or [],
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "log": str(log_path),
        "cache_hit": False,
    }
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"Step failed: {name}\nLog: {log_path}\n\n{tail}")
    if not config.disable_cache:
        cache_entry.parent.mkdir(parents=True, exist_ok=True)
        _save_step_cache(cache_entry, specs, config)
        status["cache_entry"] = str(cache_entry)
    print(f"finished {name} in {elapsed:.1f}s")
    return status


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, manifest_path: Path) -> None:
    rows = []
    for file_path in sorted(out_dir.rglob("*")):
        if not file_path.is_file() or file_path == manifest_path:
            continue
        stat = file_path.stat()
        rows.append(
            {
                "relative_path": str(file_path.relative_to(out_dir)),
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": _sha256(file_path),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "bytes", "modified_utc", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonable_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def collect_key_outputs(paths: PipelinePaths) -> dict[str, Any]:
    return {
        "top5_headline_pdf": str(paths.reports / "top5_headline_report.pdf"),
        "top5_headline_html": str(paths.reports / "top5_headline_report.html"),
        "top5_headline_zip": str(paths.reports / "top5_headline_bundle.zip"),
        "publication_grade_pdf": str(paths.reports / "publication_grade_ovk_report.pdf"),
        "publication_grade_html": str(paths.reports / "publication_grade_ovk_report.html"),
        "publication_grade_zip": str(paths.reports / "publication_grade_ovk_bundle.zip"),
        "top5_full_appended_pdf": str(paths.reports / "top5_full_appended_report.pdf"),
        "top5_full_appended_html": str(paths.reports / "top5_full_appended_report.html"),
        "top5_full_appended_zip": str(paths.reports / "top5_full_appended_results_pack.zip"),
        "nested_robustness_pdf": str(paths.reports / "nested_mean_cov_robustness_report.pdf"),
        "nested_robustness_zip": str(paths.reports / "nested_mean_cov_robustness_bundle.zip"),
        "sec_robustness_pdf": str(paths.sec_final_pdf),
        "sec_robustness_zip": str(paths.sec_final_zip),
        "sec_robustness_contact_sheet": str(paths.sec_contact),
        "sec_robustness_appendix_pdf": str(paths.sec_root / "reports" / "sec_robustness_appendix.pdf"),
        "iv_ovk_pdf": str(paths.iv_final_pdf),
        "iv_ovk_html": str(paths.iv_final_html),
        "iv_ovk_zip": str(paths.iv_final_zip),
        "iv_ovk_root": str(paths.iv_root),
        "iv_proxy_policy_panel": str(paths.data_processed / "iv_proxy_policy_panel.csv"),
        "manifest": str(paths.manifest),
        "metadata": str(paths.metadata),
    }


def _compare_numeric_csv(candidate: Path, reference: Path, tolerance: float = 1e-8) -> str | None:
    if not candidate.exists() or not reference.exists():
        return None
    left = pd.read_csv(candidate)
    right = pd.read_csv(reference)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return f"shape/column mismatch for {candidate.name}"
    numeric_cols = [c for c in left.columns if pd.api.types.is_numeric_dtype(left[c])]
    for col in numeric_cols:
        diff = (left[col].astype(float) - right[col].astype(float)).abs().max()
        if pd.notna(diff) and diff > tolerance:
            return f"{candidate.name}:{col} differs from reference by max {diff:g}"
    return None


def run_reference_comparisons(config: PipelineConfig, paths: PipelinePaths) -> list[str]:
    warnings: list[str] = []
    comparisons = [
        (
            paths.nested_root / "tables" / "model_summary.csv",
            config.workspace / "tables" / "model_summary.csv",
        ),
        (
            paths.nested_root / "tables" / "model_comparisons_block_bootstrap.csv",
            config.workspace / "tables" / "model_comparisons_block_bootstrap.csv",
        ),
        (
            paths.nested_root / "tables" / "A_t_survival_summary.csv",
            config.workspace / "tables" / "A_t_survival_summary.csv",
        ),
    ]
    for candidate, reference in comparisons:
        issue = _compare_numeric_csv(candidate, reference)
        if issue:
            warnings.append(f"Reference comparison warning: {issue}")
    return warnings


def maybe_render_pdf_contact_sheet(paths: PipelinePaths) -> list[str]:
    warnings: list[str] = []
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        warnings.append("PDF contact sheets skipped: pdftoppm was not found on PATH.")
        return warnings
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception as exc:  # pragma: no cover - depends on environment
        warnings.append(f"PDF contact sheets skipped: Pillow import failed ({exc}).")
        return warnings

    pdfs = [
        paths.reports / "top5_headline_report.pdf",
        paths.reports / "top5_full_appended_report.pdf",
        paths.reports / "nested_mean_cov_robustness_report.pdf",
        paths.iv_final_pdf,
    ]
    if (paths.reports / "top5_full_appended_with_math_report.pdf").exists():
        pdfs.append(paths.reports / "top5_full_appended_with_math_report.pdf")

    for pdf_path in pdfs:
        if not pdf_path.exists():
            continue
        render_dir = paths.pdf_render / pdf_path.stem
        render_dir.mkdir(parents=True, exist_ok=True)
        prefix = render_dir / "page"
        result = subprocess.run(
            [pdftoppm, "-png", "-r", "90", str(pdf_path), str(prefix)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            warnings.append(f"PDF contact sheet failed for {pdf_path.name}: {result.stdout.strip()}")
            continue
        images = []
        for image_path in sorted(render_dir.glob("page-*.png")):
            try:
                images.append(Image.open(image_path).convert("RGB"))
            except Exception:
                continue
        if not images:
            warnings.append(f"PDF contact sheet skipped for {pdf_path.name}: no rendered pages found.")
            continue
        thumbs = []
        for i, image in enumerate(images, start=1):
            thumb = ImageOps.contain(image, (220, 285))
            canvas = Image.new("RGB", (240, 320), "white")
            canvas.paste(thumb, ((240 - thumb.width) // 2, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"page {i}", fill="black")
            thumbs.append(canvas)
        cols = 3
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 240, rows * 320), "white")
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((i % cols) * 240, (i // cols) * 320))
        sheet.save(paths.reports / f"{pdf_path.stem}_contact_sheet.jpg", quality=88)
    return warnings


def write_metadata(
    config: PipelineConfig,
    paths: PipelinePaths,
    data_summary: dict[str, Any],
    steps: list[dict[str, Any]],
    warnings: list[str],
    started_at: str,
) -> None:
    metadata = {
        "started_at": started_at,
        "finished_at": _now_iso(),
        "config": {
            **asdict(config),
            "workspace": str(config.workspace),
            "code_dir": str(config.code_dir),
            "data_zip": str(config.data_zip),
            "out_dir": str(config.out_dir),
        },
        "data_summary": data_summary,
        "steps": steps,
        "warnings": warnings,
        "key_outputs": collect_key_outputs(paths),
        "headline_defaults": {
            "H": 24,
            "L": 12,
            "R": 5,
            "alpha": 0.25,
            "process_share": 0.25,
            "Bboot": 400,
            "Bstate": 250,
            "block_len": 18,
        },
        "publication_grade_defaults": {
            "headline_outcomes": config.headline_outcomes,
            "sf_fed_surprises": str(config.sf_fed_surprises),
            "ranks": os.environ.get("OVK_PUBLICATION_RANKS", "3,5,7"),
            "headline_rank": os.environ.get("OVK_PUBLICATION_HEADLINE_R", "5"),
            "workers": str(config.workers or "auto"),
            "benchmark_workers": str(config.benchmark_workers),
            "cache_dir": str(config.cache_dir),
            "cache_disabled": str(config.disable_cache),
            "state_draws": os.environ.get("OVK_PUBLICATION_STATE_DRAWS", "1000"),
            "full_pipeline_bootstrap_draws": os.environ.get("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "1000"),
            "bootstrap_block_length": os.environ.get("OVK_PUBLICATION_BOOT_BLOCK_LEN", "18"),
            "student_t_degrees_of_freedom": os.environ.get("OVK_PUBLICATION_STUDENT_T_DF", "7"),
            "minimum_student_t_weight": os.environ.get("OVK_PUBLICATION_MIN_STUDENT_WEIGHT", "0.25"),
            "em_iterations": os.environ.get("OVK_PUBLICATION_EM_ITERS", "5"),
            "bootstrap_em_iterations": os.environ.get("OVK_PUBLICATION_BOOT_EM_ITERS", "4"),
        },
        "nested_mean_covariance_defaults": {
            "score_difference_bootstrap_draws": "2000",
            "score_difference_bootstrap_block_length_months": "12",
        },
        "sec_robustness_defaults": {
            "bootstrap_draws": str(config.sec_bootstrap_draws),
            "bootstrap_block_length_months": "18",
        },
        "iv_proxy_defaults": {
            "enabled": os.environ.get("OVK_RUN_IV", "1"),
            "raw_data_dir": str(config.workspace / "data_raw" / "external" / "iv"),
            "bootstrap_draws": os.environ.get("OVK_IV_BOOTSTRAP_DRAWS", os.environ.get("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "1000")),
            "nested_bootstrap_draws": os.environ.get("OVK_IV_NESTED_BOOT_DRAWS", os.environ.get("OVK_NESTED_SCORE_BOOTSTRAP_DRAWS", "2000")),
            "x_col": "dgs1_eom",
            "z_col": "iv_z_preferred",
        },
        "publication_grade_upgrades": [
            "Alpha selected by robust factor-score predictive likelihood instead of fixed alpha=0.25.",
            "F, Q, and R estimated with EM-style state-space iterations instead of a fixed process-noise share.",
            "Structured transition shrinkage separates diagonal and off-diagonal log-covariance states and caps spectral radius.",
            "Student-t robust filtering downweights extreme log-covariance observations.",
            "Scale tau_t and surface-space shape allocation are reported separately.",
            "Rank R=3, R=5, and R=7 sensitivity is reported with eigengaps, log scores, and subspace angles.",
            "FFBS simulation-smoother draws replace independent marginal state draws.",
            "Full-pipeline moving-block bootstrap rebuilds LP scores, eigensystem, alpha, F, Q, R, and A_t.",
            "Pointwise and simultaneous tau_t path bands are reported.",
            "Shock-construction uncertainty is summarized as an envelope across median fallback, MP_pm, and event-manual definitions.",
            "Main robustness expansion adds same-sample expectations, K_std outcome-trace standardization, placebo shocks, SF Fed/Bauer-Swanson shocks, policy/CBI splits, smooth LPs, and episode-level spike uncertainty.",
        ],
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def run_all(args: argparse.Namespace) -> int:
    workspace = Path(__file__).resolve().parents[1]
    config = PipelineConfig(
        workspace=workspace,
        code_dir=workspace / "code",
        data_zip=_resolve(args.data_zip, workspace),
        out_dir=_resolve(args.out_dir, workspace),
        overwrite=bool(args.overwrite),
        include_math_appendix=bool(args.include_math_appendix),
        workers=args.workers,
        sec_bootstrap_draws=args.sec_bootstrap_draws,
        cache_dir=_resolve(args.cache_dir, workspace),
        disable_cache=bool(args.no_cache),
        benchmark_workers=bool(args.benchmark_workers),
        headline_outcomes=str(args.headline_outcomes),
        sf_fed_surprises=_resolve(args.sf_fed_surprises, workspace),
        python_executable=sys.executable,
    )
    paths = _paths(config)
    started_at = _now_iso()
    warnings: list[str] = []
    steps: list[dict[str, Any]] = []

    if not config.data_zip.exists():
        raise FileNotFoundError(f"Data zip not found: {config.data_zip}")
    data_hash = _sha256(config.data_zip)
    _prepare_output_dir(config, paths)
    data_summary = prepare_shared_data(config, paths)
    env = _pipeline_env(config, paths)

    step_scripts = [
        ("01_publication_grade_ovk", config.code_dir / "run_publication_grade_ovk.py", []),
        ("02_top5_full_appended_pack", config.code_dir / "make_final_appended_top5_pack.py", []),
        ("03_nested_mean_cov_robustness", config.code_dir / "ovk_nested_workflow.py", []),
        (
            "04_sec_robustness",
            config.code_dir / "run_sec_robustness.py",
            [
                "--repo-root",
                str(config.workspace),
                "--results-dir",
                str(config.out_dir),
                "--bootstrap-draws",
                str(config.sec_bootstrap_draws),
                "--clean",
            ],
        ),
    ]
    if config.include_math_appendix:
        step_scripts.append(("05_math_appendix", config.code_dir / "create_full_math_appendix_pack.py", []))
    run_iv = os.environ.get("OVK_RUN_IV", "1").lower() not in {"0", "false", "no", "off"}
    if run_iv:
        step_scripts.append((f"{len(step_scripts) + 1:02d}_iv_proxy_ovk", config.code_dir / "iv_ovk.py", []))

    try:
        for name, script, extra_args in step_scripts:
            if not script.exists():
                raise FileNotFoundError(f"Required script not found: {script}")
            steps.append(_run_step(name, script, config, paths, env, data_hash, extra_args))
        warnings.extend(run_reference_comparisons(config, paths))
        warnings.extend(maybe_render_pdf_contact_sheet(paths))
        write_metadata(config, paths, data_summary, steps, warnings, started_at)
        write_manifest(config.out_dir, paths.manifest)
    except Exception:
        write_metadata(config, paths, data_summary, steps, warnings, started_at)
        raise

    outputs = collect_key_outputs(paths)
    print("\nOVK pipeline complete")
    for key in [
        "publication_grade_pdf",
        "publication_grade_html",
        "publication_grade_zip",
        "top5_full_appended_pdf",
        "nested_robustness_pdf",
        "sec_robustness_pdf",
        "sec_robustness_zip",
        "iv_ovk_pdf",
        "iv_ovk_html",
        "iv_ovk_zip",
        "manifest",
        "metadata",
    ]:
        print(f"{key}: {outputs[key]}")
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full monthly monetary-policy OVK pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-all", help="Produce headline top-five OVK, robustness, reports, and manifest.")
    run.add_argument("--data-zip", required=True, help="Input data.zip containing the data/*.csv files.")
    run.add_argument("--out-dir", required=True, help="Directory where all pipeline outputs should be written.")
    run.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists.")
    run.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Publication-grade worker processes; default is min(available cores - 1, 4).",
    )
    run.add_argument(
        "--cache-dir",
        default=".ovk_cache",
        help="Directory for reusable publication-grade cache artifacts.",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reusable publication-grade cache reads and writes.",
    )
    run.add_argument(
        "--benchmark-workers",
        action="store_true",
        help="Run a tiny worker-count calibration before the publication-grade step when --workers is omitted.",
    )
    run.add_argument(
        "--sec-bootstrap-draws",
        type=int,
        default=1000,
        help="SEC robustness moving-block bootstrap draws; SEC runs by default.",
    )
    run.add_argument(
        "--include-math-appendix",
        action="store_true",
        help="Also build the optional mathematical appendix pack.",
    )
    run.add_argument(
        "--headline-outcomes",
        default="base5",
        choices=["base5"],
        help="Headline outcome set; base5 is the original five-outcome paper specification.",
    )
    run.add_argument(
        "--sf-fed-surprises",
        default="data_raw/external/sf_fed_monetary_policy_surprises.xlsx",
        help="Vendored SF Fed/Bauer-Swanson monetary-policy surprise snapshot used for the appendix shock.",
    )
    run.set_defaults(func=run_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
