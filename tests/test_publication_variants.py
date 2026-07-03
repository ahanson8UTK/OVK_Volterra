"""Targeted tests for publication-grade variant plumbing."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OVK_PUBLICATION_ROOT", str(ROOT / "tmp" / "test_publication_import"))
os.environ.setdefault("OVK_PUBLICATION_WORKERS", "1")
os.environ.setdefault("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "2")
os.environ.setdefault("OVK_PUBLICATION_STATE_DRAWS", "2")
sys.path.insert(0, str(ROOT / "code"))

import run_publication_grade_ovk as ovk  # noqa: E402


def synthetic_panel(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(20260603)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    panel = pd.DataFrame(
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
            "MP_median_fallback": rng.normal(0, 1, n),
            "CBI_median_fallback": rng.normal(0, 1, n),
        }
    )
    return panel


def test_outcome_selection_and_optional_cbi() -> None:
    panel = synthetic_panel()
    base = ovk.build_lp_scores(
        panel,
        "MP_median_fallback",
        "CBI_median_fallback",
        H=3,
        L=2,
        outcome_columns=tuple(ovk.BASE_OUTCOME_COLUMNS),
    )
    no_cbi = ovk.build_lp_scores(
        panel,
        "MP_median_fallback",
        None,
        H=3,
        L=2,
        outcome_columns=tuple(ovk.BASE_OUTCOME_COLUMNS),
    )
    all8 = ovk.build_lp_scores(
        panel,
        "MP_median_fallback",
        "CBI_median_fallback",
        H=3,
        L=2,
        outcome_columns=tuple(ovk.ALL_OUTCOME_COLUMNS),
    )
    assert base["outcome_labels"] == ["IP", "CPI", "Unemployment", "2Y yield", "BAA-10Y spread"]
    assert len(all8["outcome_labels"]) == 8
    assert no_cbi["Q_scores"].shape[0] == base["Q_scores"].shape[0]
    assert no_cbi["X_design"].shape[1] < base["X_design"].shape[1]


def test_lp_scores_respect_missing_calendar_months() -> None:
    panel = synthetic_panel(20)
    panel = panel[~panel["date"].eq(pd.Timestamp("2001-01-01"))].reset_index(drop=True)
    scores = ovk.build_lp_scores(
        panel,
        "MP_median_fallback",
        "CBI_median_fallback",
        H=3,
        L=2,
        outcome_columns=tuple(ovk.BASE_OUTCOME_COLUMNS),
    )
    dates = set(pd.to_datetime(scores["dates"]))
    assert pd.Timestamp("2000-10-01") not in dates
    coverage = ovk.build_publication_sample_coverage(panel, scores)
    items = dict(zip(coverage["item"], coverage["value"].astype(str)))
    assert items["Headline outcome incomplete months"] == "2001-01-01 (1 month)"
    assert items["State index attached to"] == "base month t (the plotted date)"


def test_standardization_equalizes_outcome_block_traces() -> None:
    rng = np.random.default_rng(11)
    labels = ["a", "b", "c"]
    q = rng.normal(size=(80, (ovk.H + 1) * len(labels)))
    q[:, 0::3] *= 4.0
    weights, _, _ = ovk.standardization_weights(q, labels)
    q_std = q * weights[None, :]
    traces = ovk.outcome_block_traces(q_std, labels)
    assert np.allclose(traces, traces.mean(), rtol=1e-10, atol=1e-10)
    assert q_std.shape == q.shape


def test_placebo_shocks_are_deterministic_and_shifted() -> None:
    panel = synthetic_panel(36)
    a = ovk.add_placebo_shocks(panel, seed=123, shift_months=7)
    b = ovk.add_placebo_shocks(panel, seed=123, shift_months=7)
    assert np.allclose(a["MP_placebo_permuted"], b["MP_placebo_permuted"])
    assert np.allclose(a["MP_placebo_shift84"], np.roll(panel["MP_median_fallback"].to_numpy(float), 7))


def test_spline_smoothing_preserves_shape_and_reduces_roughness() -> None:
    rng = np.random.default_rng(7)
    labels = ["x", "y"]
    h = np.arange(ovk.H + 1)
    smooth = np.sin(h / 4.0)
    q = np.tile(np.column_stack([smooth, smooth]).reshape(1, -1), (40, 1))
    q += rng.normal(0, 0.35, q.shape)
    q_sm = ovk.smooth_score_surfaces(q, labels, smoothing=10.0)
    assert q_sm.shape == q.shape
    assert ovk.score_roughness(q_sm, labels) < ovk.score_roughness(q, labels)


def test_episode_spike_uncertainty() -> None:
    dates = pd.date_range("1994-01-01", periods=24, freq="MS")
    tau = np.ones(len(dates))
    tau[2] = 5.0
    boot = np.tile(tau, (5, 1))
    res = SimpleNamespace(dates=pd.Series(dates), tau=tau)
    out = ovk.episode_spike_uncertainty(res, boot)
    row = out[out["episode"].eq("1994 tightening")].iloc[0]
    assert row["probability_any_month_in_top10"] == 1.0
    assert row["median_episode_max_tau"] == 5.0


def test_sf_fed_parser_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "sf_fixture.csv"
    pd.DataFrame(
        {
            "meeting_date": ["2020-03-15", "2020-03-23", "2020-04-29"],
            "mp_surprise": [1.0, -0.25, 0.5],
            "information_component": [9.0, 9.0, 9.0],
        }
    ).to_csv(fixture, index=False)
    parsed = ovk.load_sf_fed_surprises(fixture)
    assert list(parsed.columns)[:2] == ["date", "SF_raw_surprise"]
    assert parsed.loc[parsed["date"].eq(pd.Timestamp("2020-03-01")), "SF_raw_surprise"].iloc[0] == 0.75
