from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import plot_yosida_stress_selected_dates as yosida  # noqa: E402


def _synthetic_state() -> pd.DataFrame:
    dates = pd.date_range("1997-01-01", "2020-03-01", freq="MS")
    tau = np.full(len(dates), 0.75, dtype=float)
    month = pd.Series(dates).dt.strftime("%Y-%m")
    overrides = {
        "1999-11": 1.8,
        "2001-07": 1.1,
        "2007-10": 2.1,
        "2015-02": 1.6,
        "2019-07": 2.4,
        "2020-03": 1.9,
    }
    for key, value in overrides.items():
        tau[month.eq(key).to_numpy()] = value
    return pd.DataFrame({"date": dates, "tau_soft": tau})


def test_select_yosida_months_uses_requested_dates_and_peak_windows() -> None:
    selected = yosida.select_yosida_months(_synthetic_state())
    assert [month.date_month for month in selected] == [
        "1999-11",
        "2001-07",
        "2007-10",
        "2015-02",
        "2019-07",
        "2020-03",
    ]
    assert selected[0].role == "late_90s_tau_soft_peak"
    assert selected[2].role == "gfc_tau_soft_peak"
    assert selected[3].role == "twenty_fifteen_tau_soft_peak"


def test_yosida_stress_data_matches_scalar_identity_case() -> None:
    state = _synthetic_state()
    tau = state["tau_soft"].to_numpy(float)
    A_stack = tau[:, None, None] * np.eye(3)[None, :, :]
    C_hat = np.diag([2.0, 1.0, 0.5])
    lambda_grid = np.array([0.5, 1.0, 2.0], dtype=float)

    curves, selected = yosida.build_yosida_stress_data(A_stack, C_hat, 0.1, state, lambda_grid)

    assert len(selected) == 6
    assert len(curves) == 18
    assert curves["q_shape"].between(0.0, 1.0).all()
    first = selected[0]
    first_rows = curves[curves["date_month"].eq(first.date_month)].sort_values("lambda")
    expected = 1.0 / (1.0 + lambda_grid)
    assert np.allclose(first_rows["tau_q"].to_numpy(float), first.tau_soft)
    assert np.allclose(first_rows["q_shape"].to_numpy(float), expected)
    assert np.allclose(first_rows["q_shape"].to_numpy(float), first_rows["neutral"].to_numpy(float))


def test_yosida_all_month_stress_data_includes_every_month() -> None:
    state = _synthetic_state()
    tau = state["tau_soft"].to_numpy(float)
    A_stack = tau[:, None, None] * np.eye(3)[None, :, :]
    C_hat = np.diag([2.0, 1.0, 0.5])
    lambda_grid = np.array([0.5, 1.0, 2.0], dtype=float)

    curves = yosida.build_yosida_all_month_stress_data(A_stack, C_hat, 0.1, state, lambda_grid)

    assert len(curves) == len(state) * len(lambda_grid)
    assert curves["date_month"].nunique() == len(state)
    assert curves["q_shape"].between(0.0, 1.0).all()
    tau_q = curves.groupby("date_month", sort=False)["tau_q"].first().to_numpy(float)
    assert np.allclose(tau_q, tau)
