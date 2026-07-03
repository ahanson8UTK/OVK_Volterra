"""Shared data loading and outcome-variable helpers for the OVK pipeline."""
from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutcomeSpec:
    column: str
    label: str
    transform: str
    group: str


OUTCOME_SPECS: tuple[OutcomeSpec, ...] = (
    OutcomeSpec("ip", "IP", "log100", "macro"),
    OutcomeSpec("cpi", "CPI", "log100", "macro"),
    OutcomeSpec("cpi10", "Median CPI10", "level", "macro"),
    OutcomeSpec("unrate", "Unemployment", "level", "macro"),
    OutcomeSpec("gs2", "2Y yield", "level", "financial"),
    OutcomeSpec("baa10y", "BAA-10Y spread", "level", "financial"),
    OutcomeSpec("expinf5yr", "5Y expected inflation", "level", "macro"),
    OutcomeSpec("mich", "Michigan inflation expectations", "level", "macro"),
)
BASE_OUTCOME_COLUMNS = ("ip", "cpi", "unrate", "gs2", "baa10y")
DEFAULT_OUTCOME_LABELS = [spec.label for spec in OUTCOME_SPECS if spec.column in BASE_OUTCOME_COLUMNS]


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def available_outcome_specs(panel: pd.DataFrame) -> list[OutcomeSpec]:
    columns = {str(c).lower() for c in panel.columns}
    return [spec for spec in OUTCOME_SPECS if spec.column in columns]


def outcome_columns_for_panel(panel: pd.DataFrame) -> list[str]:
    return [spec.column for spec in available_outcome_specs(panel)]


def outcome_labels_for_panel(panel: pd.DataFrame) -> list[str]:
    return [spec.label for spec in available_outcome_specs(panel)]


def outcome_signature_for_panel(panel: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple((spec.column, spec.label) for spec in available_outcome_specs(panel))


def outcome_group_indices(labels: list[str]) -> tuple[list[int], list[int]]:
    groups = {spec.label: spec.group for spec in OUTCOME_SPECS}
    macro = [i for i, label in enumerate(labels) if groups.get(label, "macro") == "macro"]
    financial = [i for i, label in enumerate(labels) if groups.get(label, "macro") == "financial"]
    return macro, financial


def build_outcome_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Return transformed outcome variables in horizon-stack order."""
    specs = available_outcome_specs(panel)
    if not specs:
        raise ValueError("No recognized outcome columns found in panel.")
    data: dict[str, pd.Series] = {}
    for spec in specs:
        values = pd.to_numeric(panel[spec.column], errors="coerce")
        if spec.transform == "log100":
            values = values.where(values > 0)
            data[spec.label] = 100.0 * np.log(values)
        elif spec.transform == "level":
            data[spec.label] = values
        else:
            raise ValueError(f"Unknown outcome transform: {spec.transform}")
    return pd.DataFrame(data, index=panel.index)


def _col_to_idx(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")
    idx = 0
    for char in match.group(1):
        idx = idx * 26 + ord(char) - ord("A") + 1
    return idx - 1


def _read_xlsx_xml_table(path: Path) -> pd.DataFrame:
    """Read a simple XLSX worksheet without relying on workbook metadata."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                texts = [
                    t.text or ""
                    for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                ]
                shared_strings.append("".join(texts))
        sheet_name = next(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
        root = ET.fromstring(zf.read(sheet_name))
        rows: list[list[object]] = []
        max_col = 0
        for row in root.findall(".//m:sheetData/m:row", ns):
            vals: dict[int, object] = {}
            for cell in row.findall("m:c", ns):
                idx = _col_to_idx(cell.attrib.get("r", ""))
                value_node = cell.find("m:v", ns)
                value: object = None if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared_strings[int(float(str(value)))]
                vals[idx] = value
                max_col = max(max_col, idx)
            if vals:
                rows.append([vals.get(i) for i in range(max_col + 1)])
    if not rows:
        return pd.DataFrame()
    header = [str(x).strip() for x in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)


def read_cpi10_monthly(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path)
    except Exception:
        raw = _read_xlsx_xml_table(path)
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    required = {"YEAR", "QUARTER", "CPI10"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}")
    q = raw[["YEAR", "QUARTER", "CPI10"]].copy()
    q["YEAR"] = pd.to_numeric(q["YEAR"], errors="coerce")
    q["QUARTER"] = pd.to_numeric(q["QUARTER"], errors="coerce")
    q["cpi10"] = pd.to_numeric(q["CPI10"].replace("#N/A", np.nan), errors="coerce")
    q = q.dropna(subset=["YEAR", "QUARTER", "cpi10"])
    rows = []
    for row in q.itertuples(index=False):
        month = int((int(row.QUARTER) - 1) * 3 + 1)
        start = pd.Timestamp(year=int(row.YEAR), month=month, day=1)
        for offset in range(3):
            rows.append({"date": start + pd.DateOffset(months=offset), "cpi10": float(row.cpi10)})
    return pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def read_monthly_csv_series(path: Path, value_col: str, output_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"{path.name} must include {date_col!r} and {value_col!r}.")
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: output_col})
    out["date"] = _month_start(out["date"])
    out[output_col] = pd.to_numeric(out[output_col], errors="coerce")
    return out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def load_extra_outcome_series(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    data_dir = data_dir.resolve() if data_dir is not None else None
    downloads = Path.home() / "Downloads"
    paths = {
        "cpi10": _first_existing(
            [
                Path(os.environ["OVK_CPI10_PATH"]) if os.environ.get("OVK_CPI10_PATH") else None,
                data_dir / "Median_CPI10_Level.xlsx" if data_dir is not None else None,
                downloads / "Median_CPI10_Level.xlsx" if data_dir is None else None,
            ]
        ),
        "expinf5yr": _first_existing(
            [
                Path(os.environ["OVK_EXPINF5YR_PATH"]) if os.environ.get("OVK_EXPINF5YR_PATH") else None,
                data_dir / "EXPINF5YR.csv" if data_dir is not None else None,
                downloads / "EXPINF5YR.csv" if data_dir is None else None,
            ]
        ),
        "mich": _first_existing(
            [
                Path(os.environ["OVK_MICH_PATH"]) if os.environ.get("OVK_MICH_PATH") else None,
                data_dir / "MICH.csv" if data_dir is not None else None,
                downloads / "MICH.csv" if data_dir is None else None,
            ]
        ),
    }
    out: dict[str, pd.DataFrame] = {}
    if paths["cpi10"] is not None:
        out["cpi10"] = read_cpi10_monthly(paths["cpi10"])
    if paths["expinf5yr"] is not None:
        out["expinf5yr"] = read_monthly_csv_series(paths["expinf5yr"], "EXPINF5YR", "expinf5yr")
    if paths["mich"] is not None:
        out["mich"] = read_monthly_csv_series(paths["mich"], "MICH", "mich")
    return out


def merge_extra_outcome_data(fred: pd.DataFrame, data_dir: Path | None = None, balanced: bool = True) -> pd.DataFrame:
    merged = fred.copy()
    merged.columns = [str(c).lower() if str(c).lower() in {spec.column for spec in OUTCOME_SPECS} else c for c in merged.columns]
    merged["date"] = _month_start(merged["date"])
    for extra in load_extra_outcome_series(data_dir).values():
        merged = merged.merge(extra, on="date", how="left")
    merged = merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if balanced:
        merged = merged.dropna(subset=outcome_columns_for_panel(merged)).reset_index(drop=True)
    return merged
