#!/usr/bin/env python3
"""Download and process external-instrument data for the IV/proxy OVK appendix."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd


SF_FED_LANDING_URL = "https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/"
SF_FED_DIRECT_URL = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"
FRED_DGS1_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1"
FRED_GS1_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS1"


@dataclass(frozen=True)
class DownloadedSource:
    source: str
    url: str
    path: Path
    retrieved_at: str
    sha256: str
    file_size: int
    used_cached: bool
    parser_decision: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "path": str(self.path),
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "file_size": self.file_size,
            "used_cached": self.used_cached,
            "parser_decision": self.parser_decision,
            "error": self.error,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_bytes(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OVK-IV-proxy-data-downloader/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _file_suffix_from_url(url: str, default: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".xlsx", ".xls", ".csv"}:
        return suffix
    return default


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{hashlib.sha1(str(path).encode()).hexdigest()[:8]}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _source_from_file(source: str, url: str, path: Path, used_cached: bool, decision: str = "", error: str = "") -> DownloadedSource:
    return DownloadedSource(
        source=source,
        url=url,
        path=path,
        retrieved_at=_now_iso(),
        sha256=sha256_file(path) if path.exists() else "",
        file_size=path.stat().st_size if path.exists() else 0,
        used_cached=used_cached,
        parser_decision=decision,
        error=error,
    )


def _download_or_cache(source: str, url: str, path: Path, decision: str = "") -> DownloadedSource:
    try:
        payload = _download_bytes(url)
        if not payload:
            raise RuntimeError("empty response")
        _write_bytes(path, payload)
        return _source_from_file(source, url, path, used_cached=False, decision=decision)
    except Exception as exc:
        if path.exists():
            return _source_from_file(
                source,
                url,
                path,
                used_cached=True,
                decision=decision,
                error=f"download failed; using cached file: {exc}",
            )
        raise


def _scrape_sf_fed_file_url(landing_url: str) -> str:
    text = _download_bytes(landing_url).decode("utf-8", errors="replace")
    hrefs = re.findall(r"""href\s*=\s*["']([^"']+)["']""", text, flags=re.IGNORECASE)
    candidates: list[str] = []
    for href in hrefs:
        href = html.unescape(href)
        low = href.lower()
        if "monetary-policy-surprises" in low and (low.endswith(".xlsx") or low.endswith(".csv")):
            candidates.append(urljoin(landing_url, href))
    if not candidates:
        raise RuntimeError("landing page did not expose a monetary-policy-surprises xlsx/csv link")
    return candidates[0]


def download_sf_fed_surprises(
    raw_dir: Path,
    direct_url: str = SF_FED_DIRECT_URL,
    landing_url: str = SF_FED_LANDING_URL,
) -> DownloadedSource:
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        suffix = _file_suffix_from_url(direct_url, ".xlsx")
        path = raw_dir / f"sf_fed_monetary_policy_surprises{suffix}"
        return _download_or_cache("sf_fed_monetary_policy_surprises", direct_url, path, "direct URL")
    except Exception as direct_exc:
        try:
            scraped = _scrape_sf_fed_file_url(landing_url)
            suffix = _file_suffix_from_url(scraped, ".xlsx")
            path = raw_dir / f"sf_fed_monetary_policy_surprises{suffix}"
            source = _download_or_cache(
                "sf_fed_monetary_policy_surprises",
                scraped,
                path,
                "direct URL failed; scraped first landing-page xlsx/csv href",
            )
            return source
        except Exception as scrape_exc:
            for cached in [
                raw_dir / "sf_fed_monetary_policy_surprises.xlsx",
                raw_dir / "sf_fed_monetary_policy_surprises.csv",
            ]:
                if cached.exists():
                    return _source_from_file(
                        "sf_fed_monetary_policy_surprises",
                        direct_url,
                        cached,
                        used_cached=True,
                        decision="network failed; using cached SF Fed file",
                        error=f"direct: {direct_exc}; landing: {scrape_exc}",
                    )
            raise RuntimeError(f"SF Fed download failed. direct: {direct_exc}; landing: {scrape_exc}") from scrape_exc


def _fred_csv_to_monthly_eom(raw_csv: Path, value_col: str, output_col: str, monthly_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    date_col = "observation_date" if "observation_date" in raw.columns else "date"
    if date_col not in raw.columns or value_col not in raw.columns:
        raise ValueError(f"{raw_csv.name} must include {date_col!r} and {value_col!r}.")
    df = raw[[date_col, value_col]].rename(columns={date_col: "daily_date", value_col: output_col}).copy()
    df["daily_date"] = pd.to_datetime(df["daily_date"], errors="coerce")
    df[output_col] = pd.to_numeric(df[output_col].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=["daily_date", output_col]).sort_values("daily_date")
    if df.empty:
        raise ValueError(f"{raw_csv.name} has no finite {value_col} observations.")
    idx = df.groupby(df["daily_date"].dt.to_period("M"))["daily_date"].idxmax()
    monthly = df.loc[idx, ["daily_date", output_col]].sort_values("daily_date").reset_index(drop=True)
    monthly["date"] = monthly["daily_date"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly[["date", output_col]]
    monthly[f"{output_col}_diff"] = monthly[output_col] - monthly[output_col].shift(1)
    monthly_csv.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_csv, index=False)
    return monthly


def download_fred_policy_data(
    raw_dir: Path,
    dgs1_url: str = FRED_DGS1_URL,
    gs1_url: str = FRED_GS1_URL,
) -> tuple[DownloadedSource, DownloadedSource, pd.DataFrame, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dgs1_raw = raw_dir / "DGS1.csv"
    gs1_raw = raw_dir / "GS1.csv"
    dgs1_error = ""
    gs1_error = ""
    try:
        dgs1_source = _download_or_cache("fred_dgs1_daily", dgs1_url, dgs1_raw, "FRED daily DGS1 CSV")
    except Exception as exc:
        dgs1_error = str(exc)
        dgs1_source = DownloadedSource(
            source="fred_dgs1_daily",
            url=dgs1_url,
            path=dgs1_raw,
            retrieved_at=_now_iso(),
            sha256=sha256_file(dgs1_raw) if dgs1_raw.exists() else "",
            file_size=dgs1_raw.stat().st_size if dgs1_raw.exists() else 0,
            used_cached=dgs1_raw.exists(),
            parser_decision="FRED daily DGS1 CSV unavailable; will try GS1 fallback",
            error=dgs1_error,
        )
    try:
        gs1_source = _download_or_cache("fred_gs1_monthly", gs1_url, gs1_raw, "FRED monthly GS1 CSV; optional fallback")
    except Exception as exc:
        gs1_error = str(exc)
        gs1_source = DownloadedSource(
            source="fred_gs1_monthly",
            url=gs1_url,
            path=gs1_raw,
            retrieved_at=_now_iso(),
            sha256=sha256_file(gs1_raw) if gs1_raw.exists() else "",
            file_size=gs1_raw.stat().st_size if gs1_raw.exists() else 0,
            used_cached=gs1_raw.exists(),
            parser_decision="FRED monthly GS1 CSV unavailable",
            error=gs1_error,
        )
    monthly_path = raw_dir / "dgs1_monthly_eom.csv"
    try:
        if not dgs1_raw.exists():
            raise FileNotFoundError(f"DGS1 raw CSV unavailable: {dgs1_error}")
        monthly = _fred_csv_to_monthly_eom(dgs1_raw, "DGS1", "dgs1_eom", monthly_path)
        decision = "DGS1 daily converted to monthly last available observation"
    except Exception as dgs1_exc:
        if not gs1_raw.exists():
            raise RuntimeError(
                f"DGS1 conversion failed ({dgs1_exc}) and GS1 fallback is unavailable ({gs1_error})."
            )
        raw = pd.read_csv(gs1_raw)
        date_col = "observation_date" if "observation_date" in raw.columns else "date"
        if date_col not in raw.columns or "GS1" not in raw.columns:
            raise RuntimeError(f"DGS1 conversion failed ({dgs1_exc}) and GS1 fallback is malformed.")
        monthly = raw[[date_col, "GS1"]].rename(columns={date_col: "date", "GS1": "dgs1_eom"}).copy()
        monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        monthly["dgs1_eom"] = pd.to_numeric(monthly["dgs1_eom"].replace(".", np.nan), errors="coerce")
        monthly = monthly.dropna(subset=["date", "dgs1_eom"]).drop_duplicates("date", keep="last").sort_values("date")
        monthly["dgs1_eom_diff"] = monthly["dgs1_eom"] - monthly["dgs1_eom"].shift(1)
        monthly.to_csv(monthly_path, index=False)
        decision = f"DGS1 conversion failed; GS1 monthly fallback used: {dgs1_exc}"
    return dgs1_source, gs1_source, monthly, decision


def write_sources_json(raw_dir: Path, sources: list[DownloadedSource], extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "created_at": _now_iso(),
        "sources": [source.as_dict() for source in sources],
        "parser_decisions": extra or {},
    }
    path = raw_dir / "iv_data_sources.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_sources_csv(json_path: Path, csv_path: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("sources", [])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "url",
        "path",
        "retrieved_at",
        "sha256",
        "file_size",
        "used_cached",
        "parser_decision",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fieldnames} for row in rows)
    return csv_path


def download_iv_data(
    raw_dir: Path,
    sf_direct_url: str = SF_FED_DIRECT_URL,
    sf_landing_url: str = SF_FED_LANDING_URL,
    dgs1_url: str = FRED_DGS1_URL,
    gs1_url: str = FRED_GS1_URL,
) -> dict[str, Any]:
    """Download all raw IV/proxy data and write stable local artifacts.

    Network failures are tolerated only when an existing cached file is present.
    The processed DGS1 monthly file always uses a month-start ``date`` key so it
    merges with the existing monthly OVK panel.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources: list[DownloadedSource] = []
    decisions: dict[str, Any] = {}

    sf_source = download_sf_fed_surprises(raw_dir, direct_url=sf_direct_url, landing_url=sf_landing_url)
    sources.append(sf_source)

    dgs1_source, gs1_source, dgs1_monthly, policy_decision = download_fred_policy_data(
        raw_dir,
        dgs1_url=dgs1_url,
        gs1_url=gs1_url,
    )
    sources.extend([dgs1_source, gs1_source])
    decisions["policy_indicator_monthly_conversion"] = policy_decision
    decisions["dgs1_monthly_rows"] = int(len(dgs1_monthly))
    decisions["dgs1_monthly_start"] = dgs1_monthly["date"].min().strftime("%Y-%m-%d") if len(dgs1_monthly) else ""
    decisions["dgs1_monthly_end"] = dgs1_monthly["date"].max().strftime("%Y-%m-%d") if len(dgs1_monthly) else ""

    metadata_path = write_sources_json(raw_dir, sources, decisions)
    return {
        "raw_dir": str(raw_dir),
        "sf_fed_path": str(sf_source.path),
        "dgs1_raw_path": str(dgs1_source.path),
        "gs1_raw_path": str(gs1_source.path),
        "dgs1_monthly_path": str(raw_dir / "dgs1_monthly_eom.csv"),
        "metadata_path": str(metadata_path),
        "sources": [source.as_dict() for source in sources],
        "parser_decisions": decisions,
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    raw_dir = Path(__import__("os").environ.get("OVK_IV_RAW_DIR", str(repo / "data_raw" / "external" / "iv")))
    result = download_iv_data(raw_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
