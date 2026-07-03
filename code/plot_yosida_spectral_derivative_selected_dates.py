#!/usr/bin/env python3
"""One-off spectral concentration plot for selected Yosida stress months."""
from __future__ import annotations

import os
import sys
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

import plot_yosida_stress_selected_dates as yosida  # noqa: E402
import ritz_certification as rc  # noqa: E402


DEFAULT_X_GRID = np.logspace(-10.0, 2.0, 700)


def q_weighted_spectrum(A: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(rc.sym(A))
    tol = max(1.0e-10, 1.0e-8 * max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0))
    if vals.size and float(vals.min()) < -tol:
        raise ValueError(f"A has a material negative eigenvalue {float(vals.min()):.6g}.")
    vals = np.maximum(vals, 0.0)
    weights = np.einsum("ij,ji->i", vecs.T @ rc.sym(Q), vecs, optimize=True)
    return vals, weights


def spectral_concentration_curve(vals: np.ndarray, weights: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_grid, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("x_grid must be strictly positive.")
    lam = np.asarray(vals, dtype=float)[None, :]
    w = np.asarray(weights, dtype=float)[None, :]
    xx = x[:, None]
    F = np.sum((lam / (xx + lam)) * w, axis=1)
    log_derivative = np.sum(((xx * lam) / ((xx + lam) ** 2)) * w, axis=1)
    return np.clip(F, 0.0, 1.0), np.maximum(log_derivative, 0.0)


def build_selected_spectral_concentration_data(
    A_stack: np.ndarray,
    C_hat: np.ndarray,
    rho: float,
    state_frame: pd.DataFrame,
    x_grid: np.ndarray,
) -> tuple[pd.DataFrame, list[yosida.SelectedMonth]]:
    state = yosida._with_month_column(state_frame)
    selected = yosida.select_yosida_months(state)
    Q_soft = rc.build_probe_soft(C_hat, rho)

    rows: list[dict[str, object]] = []
    neutral_F = 1.0 / (1.0 + np.asarray(x_grid, dtype=float))
    neutral_log_derivative = np.asarray(x_grid, dtype=float) / ((1.0 + np.asarray(x_grid, dtype=float)) ** 2)
    for month in selected:
        tau_q = yosida.matched_probe_mean(A_stack[month.row_index], Q_soft)
        A_shape = np.asarray(A_stack[month.row_index], dtype=float) / tau_q
        vals, weights = q_weighted_spectrum(A_shape, Q_soft)
        F, log_derivative = spectral_concentration_curve(vals, weights, x_grid)
        peak_idx = int(np.argmax(log_derivative))
        for x_value, F_value, d_value, n_F, n_d in zip(
            x_grid,
            F,
            log_derivative,
            neutral_F,
            neutral_log_derivative,
        ):
            rows.append(
                {
                    "date_month": month.date_month,
                    "selection_role": month.role,
                    "selection_label": month.label,
                    "tau_soft": month.tau_soft,
                    "tau_q": float(tau_q),
                    "x": float(x_value),
                    "F_shape": float(F_value),
                    "minus_dF_dlogx": float(d_value),
                    "neutral_F": float(n_F),
                    "neutral_minus_dF_dlogx": float(n_d),
                    "peak_x": float(x_grid[peak_idx]),
                    "peak_minus_dF_dlogx": float(log_derivative[peak_idx]),
                }
            )
    return pd.DataFrame(rows), selected


def plot_selected_spectral_concentration(curves: pd.DataFrame, selected: list[yosida.SelectedMonth], output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.1),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
    neutral = curves[["x", "neutral_F", "neutral_minus_dF_dlogx"]].drop_duplicates()

    for color, month in zip(colors, selected):
        sub = curves[curves["date_month"].eq(month.date_month)]
        axes[0].plot(sub["x"], sub["F_shape"], color=color, linewidth=1.6, label=month.label)
        axes[1].plot(sub["x"], sub["minus_dF_dlogx"], color=color, linewidth=1.8, label=month.label)
        peak = sub.iloc[int(np.argmax(sub["minus_dF_dlogx"].to_numpy(float)))]
        axes[1].scatter([peak["x"]], [peak["minus_dF_dlogx"]], color=color, s=18, zorder=5)

    axes[0].plot(neutral["x"], neutral["neutral_F"], color="0.55", linewidth=1.1, linestyle=":", label="neutral scalar")
    axes[1].plot(neutral["x"], neutral["neutral_minus_dF_dlogx"], color="0.55", linewidth=1.1, linestyle=":", label="neutral scalar")
    for ax in axes:
        ax.axvline(1.0, color="0.25", linewidth=0.9, alpha=0.65)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("F_Q(x)")
    axes[0].set_title("A. Probe-mean-normalized Yosida drop")
    axes[0].legend(fontsize=7.2, ncol=2)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("-dF_Q / d log x")
    axes[1].set_title("B. Q-weighted spectral concentration")
    fig.suptitle("Selected-month soft-probe spectral concentration", fontsize=13)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    publication_root = Path(os.environ.get("OVK_PUBLICATION_ROOT", str(yosida.DEFAULT_PUBLICATION_ROOT)))
    tables, charts, components_path, state_path = yosida.publication_paths(publication_root)
    if not components_path.exists():
        raise FileNotFoundError(f"Missing full-coordinate component bundle: {components_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing headline state path: {state_path}")

    state = pd.read_csv(state_path)
    with np.load(components_path) as bundle:
        A_stack = np.asarray(bundle["A_hat"], dtype=float)
        C_hat = np.asarray(bundle["C_hat"], dtype=float)
        rho = float(np.asarray(bundle["rho"]).reshape(-1)[0])

    curves, selected = build_selected_spectral_concentration_data(A_stack, C_hat, rho, state, DEFAULT_X_GRID)
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "yosida_spectral_concentration_selected_dates.csv"
    png_path = charts / "yosida_spectral_concentration_selected_dates.png"
    pdf_path = charts / "yosida_spectral_concentration_selected_dates.pdf"
    curves.to_csv(csv_path, index=False)
    plot_selected_spectral_concentration(curves, selected, png_path, pdf_path)

    print("Wrote", png_path)
    print("Wrote", pdf_path)
    print("Wrote", csv_path)
    for month in selected:
        sub = curves[curves["date_month"].eq(month.date_month)]
        peak = sub.iloc[int(np.argmax(sub["minus_dF_dlogx"].to_numpy(float)))]
        print(
            f"{month.date_month}: tau_q={float(peak['tau_q']):.6g}, "
            f"peak_x={float(peak['peak_x']):.6g}, "
            f"peak_height={float(peak['peak_minus_dF_dlogx']):.6g}"
        )


if __name__ == "__main__":
    main()
