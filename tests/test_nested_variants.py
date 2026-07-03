"""Targeted tests for nested mean/covariance outcome variants."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OVK_NESTED_OUTDIR", str(ROOT / "tmp" / "test_nested_import"))
os.environ.setdefault("OVK_NESTED_SCORE_BOOTSTRAP_DRAWS", "5")
sys.path.insert(0, str(ROOT / "code"))

import ovk_nested_workflow as nested  # noqa: E402


def synthetic_panel(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(20260604)
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


def test_nested_outcome_variants_select_five_and_eight_outcomes() -> None:
    panel = synthetic_panel()
    specs = nested.nested_variant_specs(panel)
    assert [spec.key for spec in specs] == ["base5_headline", "all8_expectation_overlap"]

    base = nested.build_lp_scores(panel, H=3, L=2, outcome_columns=tuple(nested.BASE_OUTCOME_COLUMNS))
    all8 = nested.build_lp_scores(panel, H=3, L=2, outcome_columns=tuple(nested.ALL_OUTCOME_COLUMNS))

    assert base["outcome_labels"] == ["IP", "CPI", "Unemployment", "2Y yield", "BAA-10Y spread"]
    assert all8["outcome_labels"] == [
        "IP",
        "CPI",
        "Median CPI10",
        "Unemployment",
        "2Y yield",
        "BAA-10Y spread",
        "5Y expected inflation",
        "Michigan inflation expectations",
    ]
    assert base["Q"].shape[1] == (3 + 1) * 5
    assert all8["Q"].shape[1] == (3 + 1) * 8
    assert all8["Q"].shape[0] == base["Q"].shape[0]
