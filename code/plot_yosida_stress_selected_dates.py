#!/usr/bin/env python3
"""Plot full-coordinate Yosida stress curves for selected Section 3.1 months."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ritz_certification as rc  # noqa: E402


DEFAULT_PUBLICATION_ROOT = ROOT / "results" / "publication_grade_ovk"
FIXED_MONTHS = ("2001-07", "2019-07", "2020-03")
PEAK_WINDOWS = (
    ("late_90s_tau_soft_peak", "Late-90s tau_soft peak", "1997-01", "1999-12"),
    ("gfc_tau_soft_peak", "2007-2008 tau_soft peak", "2007-01", "2008-12"),
    ("twenty_fifteen_tau_soft_peak", "2015 tau_soft peak", "2015-01", "2015-12"),
)
DEFAULT_LAMBDA_GRID = np.logspace(-2.0, 2.0, 200)


@dataclass(frozen=True)
class SelectedMonth:
    date_month: str
    role: str
    label: str
    tau_soft: float
    row_index: int


def publication_paths(publication_root: Path) -> tuple[Path, Path, Path, Path]:
    tables = publication_root / "outputs" / "tables"
    charts = publication_root / "outputs" / "charts"
    components = tables / "publication_grade_full_coordinate_covariance_components.npz"
    state_path = tables / "publication_grade_headline_state_path.csv"
    return tables, charts, components, state_path


def _with_month_column(state_frame: pd.DataFrame) -> pd.DataFrame:
    state = state_frame.copy()
    if "date_month" not in state.columns:
        state["date_month"] = pd.to_datetime(state["date"]).dt.strftime("%Y-%m")
    return state


def select_yosida_months(state_frame: pd.DataFrame) -> list[SelectedMonth]:
    """Select fixed months and peak months requested for the stress plot."""
    state = _with_month_column(state_frame)
    selected: list[SelectedMonth] = []
    seen: set[str] = set()

    for month in FIXED_MONTHS:
        matches = state.index[state["date_month"].eq(month)].to_numpy()
        if len(matches) == 0:
            raise ValueError(f"Requested month {month} is not present in the state path.")
        idx = int(matches[0])
        selected.append(
            SelectedMonth(
                date_month=month,
                role="requested_month",
                label=month,
                tau_soft=float(state.loc[idx, "tau_soft"]),
                row_index=idx,
            )
        )
        seen.add(month)

    dates = pd.to_datetime(state["date"])
    for role, label, start, end in PEAK_WINDOWS:
        mask = (dates >= pd.Timestamp(f"{start}-01")) & (dates <= pd.Timestamp(f"{end}-01"))
        window = state.loc[mask]
        if window.empty:
            raise ValueError(f"No state rows found for peak window {start} to {end}.")
        idx = int(window["tau_soft"].astype(float).idxmax())
        month = str(state.loc[idx, "date_month"])
        if month in seen:
            continue
        selected.append(
            SelectedMonth(
                date_month=month,
                role=role,
                label=f"{month} ({label})",
                tau_soft=float(state.loc[idx, "tau_soft"]),
                row_index=idx,
            )
        )
        seen.add(month)

    return sorted(selected, key=lambda month: month.date_month)


def yosida_stress_curve(A: np.ndarray, Q: np.ndarray, lambda_grid: np.ndarray) -> np.ndarray:
    """Compute tr[Q A(lambda I + A)^(-1)] over a positive lambda grid."""
    grid = np.asarray(lambda_grid, dtype=float)
    if np.any(grid <= 0.0):
        raise ValueError("lambda_grid must be strictly positive.")
    vals, vecs = np.linalg.eigh(rc.sym(A))
    tol = max(1.0e-10, 1.0e-8 * max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0))
    if vals.size and float(vals.min()) < -tol:
        raise ValueError(f"A has a material negative eigenvalue {float(vals.min()):.6g}.")
    vals = np.maximum(vals, 0.0)
    q_diag = np.einsum("ij,ji->i", vecs.T @ rc.sym(Q), vecs, optimize=True)
    curves = np.array([np.sum((vals / (float(lam) + vals)) * q_diag) for lam in grid], dtype=float)
    return np.clip(curves, 0.0, 1.0)


def matched_probe_mean(A: np.ndarray, Q: np.ndarray) -> float:
    """Return tr(Q A), the matched-probe scale removed from the shape plot."""
    tau_q = float(np.einsum("ij,ji->", rc.sym(Q), rc.sym(A), optimize=True))
    if not np.isfinite(tau_q) or tau_q <= 0.0:
        raise ValueError(f"Matched probe mean must be positive and finite; got {tau_q}.")
    return tau_q


def yosida_shape_stress_curve(A: np.ndarray, Q: np.ndarray, lambda_grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute Yosida stress after normalizing A so tr(Q A)=1."""
    tau_q = matched_probe_mean(A, Q)
    return yosida_stress_curve(np.asarray(A, dtype=float) / tau_q, Q, lambda_grid), tau_q


def build_yosida_stress_data(
    A_stack: np.ndarray,
    C_hat: np.ndarray,
    rho: float,
    state_frame: pd.DataFrame,
    lambda_grid: np.ndarray,
) -> tuple[pd.DataFrame, list[SelectedMonth]]:
    state = _with_month_column(state_frame)
    if A_stack.shape[0] != len(state):
        raise ValueError(f"A_stack has {A_stack.shape[0]} states but state path has {len(state)} rows.")
    selected = select_yosida_months(state)
    Q_soft = rc.build_probe_soft(C_hat, rho)

    rows: list[dict[str, object]] = []
    neutral = 1.0 / (1.0 + np.asarray(lambda_grid, dtype=float))
    for month in selected:
        curve, tau_q = yosida_shape_stress_curve(A_stack[month.row_index], Q_soft, lambda_grid)
        for lam, q_value, neutral_value in zip(lambda_grid, curve, neutral):
            rows.append(
                {
                    "date_month": month.date_month,
                    "selection_role": month.role,
                    "selection_label": month.label,
                    "tau_soft": month.tau_soft,
                    "tau_q": float(tau_q),
                    "lambda": float(lam),
                    "q_shape": float(q_value),
                    "neutral": float(neutral_value),
                }
            )
    return pd.DataFrame(rows), selected


def build_yosida_all_month_stress_data(
    A_stack: np.ndarray,
    C_hat: np.ndarray,
    rho: float,
    state_frame: pd.DataFrame,
    lambda_grid: np.ndarray,
) -> pd.DataFrame:
    """Compute full-coordinate soft-probe Yosida stress curves for every state month."""
    state = _with_month_column(state_frame)
    if A_stack.shape[0] != len(state):
        raise ValueError(f"A_stack has {A_stack.shape[0]} states but state path has {len(state)} rows.")
    Q_soft = rc.build_probe_soft(C_hat, rho)

    rows: list[dict[str, object]] = []
    neutral = 1.0 / (1.0 + np.asarray(lambda_grid, dtype=float))
    date_months = state["date_month"].astype(str).to_numpy()
    tau_values = state["tau_soft"].astype(float).to_numpy()
    for row_index, (date_month, tau_soft) in enumerate(zip(date_months, tau_values)):
        curve, tau_q = yosida_shape_stress_curve(A_stack[row_index], Q_soft, lambda_grid)
        for lam, q_value, neutral_value in zip(lambda_grid, curve, neutral):
            rows.append(
                {
                    "date_month": date_month,
                    "tau_soft": float(tau_soft),
                    "tau_q": float(tau_q),
                    "lambda": float(lam),
                    "q_shape": float(q_value),
                    "neutral": float(neutral_value),
                }
            )
    return pd.DataFrame(rows)


def plot_yosida_stress(curves: pd.DataFrame, selected: list[SelectedMonth], output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.6),
        gridspec_kw={"height_ratios": [2.25, 1.0]},
        constrained_layout=True,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
    for color, month in zip(colors, selected):
        sub = curves[curves["date_month"].eq(month.date_month)]
        axes[0].plot(sub["lambda"], sub["q_shape"], color=color, linewidth=1.8, label=month.label)
    neutral = curves[["lambda", "neutral"]].drop_duplicates()
    axes[0].plot(neutral["lambda"], neutral["neutral"], color="0.55", linewidth=1.1, linestyle=":", label="neutral")
    axes[0].axvline(1.0, color="0.25", linewidth=0.9, alpha=0.65)
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Yosida shape stress")
    axes[0].set_title("A. Probe-mean-normalized soft-probe Yosida shape stress")
    axes[0].legend(fontsize=7.2, ncol=2)

    labels = [month.date_month for month in selected]
    tau_values = [month.tau_soft for month in selected]
    axes[1].bar(labels, tau_values, color=colors, width=0.72)
    axes[1].axhline(1.0, color="0.25", linewidth=0.9, alpha=0.65)
    axes[1].set_ylabel("tau_soft")
    axes[1].set_title("B. Selected-month multiplicative scale")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(True, alpha=0.22)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_yosida_stress_all_months(curves: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    month_order = curves["date_month"].drop_duplicates().tolist()
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(month_order), 1)))
    for color, month in zip(colors, month_order):
        sub = curves[curves["date_month"].eq(month)]
        ax.plot(sub["lambda"], sub["q_shape"], color=color, linewidth=0.55, alpha=0.28)

    neutral = curves[["lambda", "neutral"]].drop_duplicates()
    ax.plot(neutral["lambda"], neutral["neutral"], color="0.05", linewidth=1.35, linestyle=":")
    ax.axvline(1.0, color="0.25", linewidth=0.9, alpha=0.65)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("lambda")
    ax.set_ylabel("Yosida shape stress")
    ax.set_title("Probe-mean-normalized soft-probe Yosida shape stress, all sample months")
    ax.grid(True, alpha=0.22)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_yosida_stress_facets(curves: pd.DataFrame, selected: list[SelectedMonth], output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
    neutral = curves[["lambda", "neutral"]].drop_duplicates()
    role_labels = {
        "requested_month": "requested",
        "late_90s_tau_soft_peak": "late-90s peak",
        "gfc_tau_soft_peak": "2007-2008 peak",
        "twenty_fifteen_tau_soft_peak": "2015 peak",
    }

    for ax, color, month in zip(axes.ravel(), colors, selected):
        sub = curves[curves["date_month"].eq(month.date_month)]
        ax.plot(sub["lambda"], sub["q_shape"], color=color, linewidth=2.0)
        ax.plot(neutral["lambda"], neutral["neutral"], color="0.6", linewidth=1.0, linestyle=":")
        ax.axvline(1.0, color="0.25", linewidth=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{month.date_month}\ntau_soft={month.tau_soft:.3f}, {role_labels.get(month.role, month.role)}", fontsize=9)
        ax.grid(True, alpha=0.22)

    for ax in axes[:, 0]:
        ax.set_ylabel("Yosida shape stress")
    for ax in axes[-1, :]:
        ax.set_xlabel("lambda")
    fig.suptitle("Probe-mean-normalized soft-probe Yosida shape stress by selected month", fontsize=13)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    publication_root = Path(os.environ.get("OVK_PUBLICATION_ROOT", str(DEFAULT_PUBLICATION_ROOT)))
    tables, charts, components_path, state_path = publication_paths(publication_root)
    if not components_path.exists():
        raise FileNotFoundError(f"Missing full-coordinate component bundle: {components_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing headline state path: {state_path}")

    state = pd.read_csv(state_path)
    with np.load(components_path) as bundle:
        A_stack = np.asarray(bundle["A_hat"], dtype=float)
        C_hat = np.asarray(bundle["C_hat"], dtype=float)
        rho = float(np.asarray(bundle["rho"]).reshape(-1)[0])
        d_rho = float(np.asarray(bundle["d_rho"]).reshape(-1)[0]) if "d_rho" in bundle.files else float("nan")

    curves, selected = build_yosida_stress_data(A_stack, C_hat, rho, state, DEFAULT_LAMBDA_GRID)
    all_month_curves = build_yosida_all_month_stress_data(A_stack, C_hat, rho, state, DEFAULT_LAMBDA_GRID)

    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "yosida_stress_selected_dates.csv"
    metadata_path = tables / "yosida_stress_selected_dates_metadata.json"
    png_path = charts / "yosida_stress_selected_dates.png"
    pdf_path = charts / "yosida_stress_selected_dates.pdf"
    facet_png_path = charts / "yosida_stress_selected_dates_facets.png"
    facet_pdf_path = charts / "yosida_stress_selected_dates_facets.pdf"
    all_month_png_path = charts / "yosida_stress_all_months.png"
    all_month_pdf_path = charts / "yosida_stress_all_months.pdf"

    curves.to_csv(csv_path, index=False)
    metadata = {
        "definition": "q_shape(lambda; s) = tr[Q_soft A_shape_s (lambda I + A_shape_s)^(-1)], A_shape_s = A_s / tau_q_s, tau_q_s = tr(Q_soft A_s)",
        "component_path": str(components_path),
        "state_path": str(state_path),
        "rho": rho,
        "d_rho": d_rho,
        "lambda_grid_min": float(DEFAULT_LAMBDA_GRID.min()),
        "lambda_grid_max": float(DEFAULT_LAMBDA_GRID.max()),
        "lambda_grid_count": int(DEFAULT_LAMBDA_GRID.size),
        "selected_months": [asdict(month) for month in selected],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_yosida_stress(curves, selected, png_path, pdf_path)
    plot_yosida_stress_facets(curves, selected, facet_png_path, facet_pdf_path)
    plot_yosida_stress_all_months(all_month_curves, all_month_png_path, all_month_pdf_path)

    print("Wrote", png_path)
    print("Wrote", pdf_path)
    print("Wrote", facet_png_path)
    print("Wrote", facet_pdf_path)
    print("Wrote", all_month_png_path)
    print("Wrote", all_month_pdf_path)
    print("Wrote", csv_path)
    for month in selected:
        print(f"{month.date_month}: tau_soft={month.tau_soft:.6g}, role={month.role}")


if __name__ == "__main__":
    main()
