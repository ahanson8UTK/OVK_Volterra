#!/usr/bin/env python3
"""Run monetary-policy LP OVK targets with HAC and Volterra memory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

os.environ.setdefault("OVK_PUBLICATION_ROOT", str(ROOT / "tmp" / "memory_targets_publication_import"))
os.environ.setdefault("OVK_REPORTS_DIR", str(ROOT / "tmp" / "memory_targets_publication_import" / "reports"))
os.environ.setdefault("OVK_DISABLE_CACHE", "1")
os.environ.setdefault("OVK_PUBLICATION_BOOTSTRAP_DRAWS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_publication_grade_ovk as pub
from hilbert_volterra import (
    HilbertVolterraKernelConfig,
    make_hilbert_volterra_target,
)
from time_series_targets import (
    TargetResult,
    check_psd_symmetric,
    effective_support,
    make_diagonal_old_target,
    make_hac_filtered_target,
    relative_geometry_from_target,
)


TARGET_LABELS = {
    "diagonal_old": "Old diagonal",
    "hac_filtered": "HAC filtered",
    "hilbert_volterra": "Hilbert-Volterra",
}
TARGET_DIRS = {
    "diagonal_old": "diagonal_old",
    "hac_filtered": "hac_filtered_L{L}",
    "hilbert_volterra": "hilbert_volterra_L{L}_gamma{gamma}_memory_{memory}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["diagonal_old", "hac_filtered", "hilbert_volterra"],
        choices=["diagonal_old", "hac_filtered", "hilbert_volterra"],
    )
    parser.add_argument("--hac-lags", type=int, default=12)
    parser.add_argument("--memory-half-lives", nargs="+", type=float, default=[3.0, 12.0, 36.0])
    parser.add_argument("--memory-weights", default="equal")
    parser.add_argument("--signature-gamma", type=float, default=0.05)
    parser.add_argument("--base-inner", choices=["reference_soft", "euclidean"], default="reference_soft")
    parser.add_argument("--time-bandwidth", type=float, default=0.08, help="eta for the old graph-resolvent time kernel")
    parser.add_argument("--feature-bandwidth", default="median")
    parser.add_argument("--strict-past", dest="strict_past", action="store_true", default=True)
    parser.add_argument("--include-current", dest="strict_past", action="store_false")
    parser.add_argument("--kernel-normalize", dest="kernel_normalize", action="store_true", default=True)
    parser.add_argument("--no-kernel-normalize", dest="kernel_normalize", action="store_false")
    parser.add_argument("--volterra-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--volterra-level", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--volterra-pca-dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--volterra-half-lives", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-draws", type=int, default=0)
    parser.add_argument("--bootstrap-block-len", type=int, default=18)
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=0,
        help="Parallel workers for bootstrap draws. Use 0 for auto, 1 for serial.",
    )
    parser.add_argument(
        "--bootstrap-chunk-size",
        type=int,
        default=0,
        help="Bootstrap draws per worker task. Use 0 for automatic chunking.",
    )
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "monetary_lp_memory_targets")
    parser.add_argument("--panel", type=Path, default=ROOT / "data_processed" / "processed_panel_three_shock_definitions.csv")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic scores even when empirical data are present")
    parser.add_argument("--add-rotation-diagnostics", action="store_true")
    parser.add_argument("--rotation-reference", choices=["pooled", "diagonal", "hac", "hilbert_volterra"], default="pooled")
    parser.add_argument("--rotation-lambda-min", type=float, default=1e-2)
    parser.add_argument("--rotation-lambda-max", type=float, default=1e2)
    parser.add_argument("--rotation-lambda-count", type=int, default=41)
    parser.add_argument("--min-rotation-anisotropy", type=float, default=0.05)
    args = parser.parse_args()
    legacy = []
    for name in ["volterra_rank", "volterra_level", "volterra_pca_dim", "volterra_half_lives"]:
        if getattr(args, name) is not None:
            legacy.append(name.replace("_", "-"))
    if legacy:
        warnings.warn(
            "Finite Volterra rank/level features are legacy. The main Hilbert-Volterra target "
            "uses a kernelized infinite-level Fock Gram matrix and does not construct Phi_t. "
            f"Ignored legacy options: {', '.join(legacy)}",
            DeprecationWarning,
            stacklevel=2,
        )
    return args


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def coordinate_map(labels: list[str], H: int) -> list[dict[str, Any]]:
    rows = []
    pvars = len(labels)
    for h in range(H + 1):
        for j, label in enumerate(labels):
            rows.append({"coordinate": int(h * pvars + j), "horizon_months": int(h), "outcome": label})
    return rows


def synthetic_scores(seed: int, n: int = 120, pvars: int = 5, H: int = 24) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = pvars * (H + 1)
    factors = rng.normal(size=(n, 6))
    loadings = rng.normal(scale=0.45, size=(6, p))
    seasonal = np.sin(np.linspace(0.0, 8.0 * np.pi, n))[:, None] * rng.normal(scale=0.30, size=(1, p))
    psi = factors @ loadings + seasonal + 0.10 * rng.normal(size=(n, p))
    dates = pd.Series(pd.date_range("1992-11-01", periods=n, freq="MS"))
    labels = ["IP", "CPI", "Unemployment", "2Y yield", "BAA-10Y spread"]
    return {
        "psi": psi,
        "dates": dates,
        "labels": labels,
        "outcome_columns": list(pub.BASE_OUTCOME_COLUMNS),
        "H": H,
        "L": 12,
        "pvars": pvars,
        "source": "synthetic smoke scores",
        "used_synthetic_scores": True,
        "panel_date_range": "synthetic",
        "missing_empirical_data": [],
        "score_metadata": {},
    }


def load_empirical_scores(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        return synthetic_scores(args.seed)
    missing = []
    if not args.panel.exists():
        missing.append(str(args.panel))
    if missing:
        data = synthetic_scores(args.seed)
        data["missing_empirical_data"] = missing
        return data
    panel = pd.read_csv(args.panel, parse_dates=["date"])
    scores = pub.build_lp_scores(
        panel,
        "MP_median_fallback",
        "CBI_median_fallback",
        H=24,
        L=12,
        outcome_columns=pub.BASE_OUTCOME_COLUMNS,
    )
    dates = pd.to_datetime(scores["dates"]).reset_index(drop=True)
    panel_dates = pd.to_datetime(panel["date"], errors="coerce")
    return {
        "psi": np.asarray(scores["Q_scores"], dtype=float),
        "dates": dates,
        "labels": list(scores["outcome_labels"]),
        "outcome_columns": list(scores["outcome_columns"]),
        "H": int(scores["H"]),
        "L": int(scores["L"]),
        "pvars": int(scores["pvars"]),
        "source": str(args.panel),
        "used_synthetic_scores": False,
        "panel_date_range": f"{panel_dates.min().strftime('%Y-%m-%d')} to {panel_dates.max().strftime('%Y-%m-%d')}",
        "missing_empirical_data": [],
        "score_metadata": {
            "shock_col": str(scores.get("shock_col", "")),
            "cbi_col": str(scores.get("cbi_col", "")),
            "sigma_m2": float(scores.get("sigma_m2", np.nan)),
            "valid_idx_min": int(np.min(scores["valid_idx"])),
            "valid_idx_max": int(np.max(scores["valid_idx"])),
        },
    }


def target_directory(base: Path, target: str, args: argparse.Namespace) -> Path:
    memory_tag = "_".join(f"{h:g}".replace(".", "p") for h in getattr(args, "memory_half_lives", []))
    gamma_tag = f"{int(round(float(getattr(args, 'signature_gamma', 0.0)) * 100)):03d}"
    name = TARGET_DIRS[target].format(L=args.hac_lags, gamma=gamma_tag, memory=memory_tag)
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_target(target: str, data: dict[str, Any], args: argparse.Namespace) -> TargetResult:
    psi = data["psi"]
    dates = data["dates"]
    if target == "diagonal_old":
        return make_diagonal_old_target(
            psi,
            dates,
            time_bandwidth=args.time_bandwidth,
            kernel="graph_resolvent",
            reference="empirical",
        )
    if target == "hac_filtered":
        return make_hac_filtered_target(
            psi,
            dates,
            L=args.hac_lags,
            time_bandwidth=args.time_bandwidth,
            kernel="graph_resolvent",
            reference="empirical",
        )
    if target == "hilbert_volterra":
        feature_bandwidth: str | float
        try:
            feature_bandwidth = float(args.feature_bandwidth)
        except ValueError:
            feature_bandwidth = str(args.feature_bandwidth)
        memory_weights: str | tuple[float, ...]
        if str(args.memory_weights).lower() == "equal":
            memory_weights = "equal"
        else:
            memory_weights = tuple(float(x) for x in str(args.memory_weights).split(","))
        config = HilbertVolterraKernelConfig(
            memory_half_lives=tuple(float(x) for x in args.memory_half_lives),
            memory_weights=memory_weights,
            gamma=float(args.signature_gamma),
            base_inner=str(args.base_inner),
            strict_past=bool(args.strict_past),
            normalize_kernel=bool(args.kernel_normalize),
            feature_bandwidth=feature_bandwidth,
        )
        return make_hilbert_volterra_target(
            psi,
            dates,
            hac_lags=args.hac_lags,
            config=config,
            time_bandwidth=args.time_bandwidth,
            reference="grid_induced",
        )
    raise ValueError(f"unknown target: {target}")


def cell_probes(K: np.ndarray, C: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pub.full_coordinate_cell_probes(K, C, tau)


def result_for_old_helpers(
    target: str,
    target_result: TargetResult,
    geometry: dict[str, Any],
    labels: list[str],
    kernel_eta: float,
) -> Any:
    cell_amp, cell_shape, low = cell_probes(
        geometry["K_by_state"],
        geometry["C_ref"],
        geometry["tau_soft"],
    )
    return pub.FullCoordinateResult(
        variant=target,
        label=TARGET_LABELS[target],
        dates=target_result.state_dates,
        outcome_labels=labels,
        chi=target_result.scores,
        C_hat=geometry["C_ref"],
        D_rho=geometry["D_rho"],
        D_invsqrt=geometry["D_invsqrt"],
        rho=float(geometry["rho"]),
        d_rho=float(geometry["d_rho"]),
        temporal_weights=target_result.weights,
        K_hat=geometry["K_by_state"],
        A_hat=geometry["A_by_state"],
        tau_soft=geometry["tau_soft"],
        cell_amp=cell_amp,
        cell_shape=cell_shape,
        low_variance_cell_mask=low,
        backend=target,
        kernel_eta=float(kernel_eta),
    )


def plot_tau(df: pd.DataFrame, path: Path, title: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.plot(pd.to_datetime(df["date"]), df["tau_soft"], label=label, color="black", linewidth=1.4)
    ax.axhline(1.0, linewidth=0.9, color="tab:red", linestyle="--", label="reference = 1")
    ax.set_title(title)
    ax.set_ylabel("tau_soft")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _short_episode_label(episode: Any) -> str:
    label = str(episode).replace("_", " ").strip().lower()
    replacements = {
        "reference neutral full coordinate": "Neutral",
        "reference neutral": "Neutral",
        "max tau soft": "Peak tau",
        "max full coordinate shape dispersion": "Peak shape",
        "max shape dispersion": "Peak shape",
        "march 2020": "Mar 2020",
    }
    return replacements.get(label, label.title())


def plot_selected_stress(result: Any, episodes: list[dict[str, Any]], H: int, labels: list[str], path: Path, _title: str) -> None:
    pvars = len(labels)
    n = max(1, len(episodes))
    if n <= 3:
        ncols = n
    elif n <= 4:
        ncols = 2
    else:
        ncols = min(3, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))
    selected = np.vstack([result.cell_shape[ep["idx"]].reshape(H + 1, pvars) for ep in episodes])
    vmax = float(np.nanquantile(np.abs(selected), 0.98)) if selected.size else 1.0
    vmax = max(vmax, 0.25)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.9 * ncols + 0.7, 2.95 * nrows + 0.25),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    yticks = [h for h in [0, 3, 6, 12, 18, H] if h <= H]
    for i, (ax, ep) in enumerate(zip(axes.ravel(), episodes)):
        mat = result.cell_shape[ep["idx"]].reshape(H + 1, pvars)
        image = ax.imshow(mat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        date_label = pd.to_datetime(ep["date"]).strftime("%Y-%m")
        ax.set_title(f"{date_label}\n{_short_episode_label(ep['episode'])}", fontsize=9, pad=4)
        row = i // ncols
        col = i % ncols
        show_x = row == nrows - 1
        show_y = col == 0
        ax.set_xticks(np.arange(pvars))
        if show_x:
            ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(yticks)
        if show_y:
            ax.set_ylabel("Horizon", fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes.ravel()[len(episodes) :]:
        ax.axis("off")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel()[: len(episodes)].tolist(), shrink=0.9)
        colorbar.set_label("log relative cell stress", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def metadata_for_json(value: Any) -> Any:
    """Convert metadata to JSON-safe objects without inlining large arrays."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"kappa_raw", "kappa_norm", "fock_distance"}:
                out[key] = f"saved separately as {key}.npy"
            else:
                out[key] = metadata_for_json(item)
        return out
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [metadata_for_json(x) for x in value]
    return value


def resolve_bootstrap_workers(args: argparse.Namespace, draws: int) -> int:
    """Resolve the number of bootstrap worker processes."""
    if draws <= 1:
        return 1
    requested = int(getattr(args, "bootstrap_workers", 0))
    if requested == 0:
        auto = max(1, min((os.cpu_count() or 2) - 1, 8))
        return max(1, min(draws, auto))
    return max(1, min(draws, requested))


def make_bootstrap_draw_indices(n: int, block_len: int, draws: int, seed: int) -> list[tuple[int, np.ndarray]]:
    """Precompute moving-block bootstrap indices so worker count cannot change resamples."""
    rng = np.random.default_rng(int(seed) + 17)
    return [(b, pub.circular_block_indices(int(n), int(block_len), rng)) for b in range(int(draws))]


def bootstrap_draw_plan_digest(draw_items: list[tuple[int, np.ndarray]]) -> str:
    """Stable digest for a shared bootstrap draw plan."""
    h = hashlib.sha256()
    for draw_id, indices in draw_items:
        h.update(np.asarray([int(draw_id)], dtype=np.int64).tobytes())
        h.update(np.asarray(indices, dtype=np.int64).tobytes())
    return h.hexdigest()


def save_bootstrap_draw_plan(output_dir: Path, args: argparse.Namespace, data: dict[str, Any], draw_items: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
    """Save the paired bootstrap draw plan shared by all target models."""
    draws = int(args.bootstrap_draws)
    if draws <= 0:
        return {
            "draws": 0,
            "paired_across_targets": False,
            "draw_plan_digest": None,
            "draw_indices_file": None,
        }
    if len(draw_items) != draws:
        raise ValueError(f"bootstrap draw plan has {len(draw_items)} draws, expected {draws}")
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_ids = np.asarray([draw_id for draw_id, _ in draw_items], dtype=np.int64)
    indices = np.vstack([np.asarray(ix, dtype=np.int64) for _, ix in draw_items])
    digest = bootstrap_draw_plan_digest(draw_items)
    np.save(output_dir / "bootstrap_draw_indices.npy", indices)
    metadata = {
        "draws": draws,
        "draw_ids": draw_ids.tolist(),
        "n_observations": int(np.asarray(data["psi"]).shape[0]),
        "block_length": int(args.bootstrap_block_len),
        "seed": int(args.seed),
        "index_shape": list(indices.shape),
        "paired_across_targets": True,
        "draw_plan_digest": digest,
        "draw_indices_file": "bootstrap_draw_indices.npy",
        "note": "Rows are bootstrap draw ids and columns are source-row indices. The same row is used for every target model.",
    }
    (output_dir / "bootstrap_draw_plan_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def chunk_bootstrap_draws(draw_items: list[tuple[int, np.ndarray]], workers: int, chunk_size: int = 0) -> list[list[tuple[int, np.ndarray]]]:
    """Split bootstrap draw items into load-balanced chunks."""
    if not draw_items:
        return []
    if chunk_size <= 0:
        chunk_size = max(1, int(np.ceil(len(draw_items) / max(1, workers * 4))))
    return [draw_items[i : i + chunk_size] for i in range(0, len(draw_items), chunk_size)]


def _bootstrap_chunk_rows(target: str, data: dict[str, Any], args: argparse.Namespace, draw_items: list[tuple[int, np.ndarray]]) -> list[dict[str, Any]]:
    psi = np.asarray(data["psi"], dtype=float)
    dates = pd.to_datetime(pd.Series(data["dates"]), errors="coerce").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for b, ix in draw_items:
        boot_data = dict(data)
        boot_data["psi"] = psi[np.asarray(ix, dtype=int)]
        boot_data["dates"] = dates.reset_index(drop=True)
        target_result = build_target(target, boot_data, args)
        geom = relative_geometry_from_target(target_result.K_by_state, target_result.C_ref)
        for i, date in enumerate(target_result.state_dates):
            rows.append({"draw": int(b), "date": pd.to_datetime(date).strftime("%Y-%m-%d"), "tau_soft": float(geom["tau_soft"][i])})
    return rows


def _bootstrap_chunk_worker(payload: tuple[str, dict[str, Any], argparse.Namespace, list[tuple[int, np.ndarray]]]) -> list[dict[str, Any]]:
    target, data, args, draw_items = payload
    return _bootstrap_chunk_rows(target, data, args, draw_items)


def bootstrap_tau(
    target: str,
    data: dict[str, Any],
    args: argparse.Namespace,
    builder: Callable[[str, dict[str, Any], argparse.Namespace], TargetResult] | None = None,
    draw_items: list[tuple[int, np.ndarray]] | None = None,
) -> pd.DataFrame:
    draws = int(args.bootstrap_draws)
    if draws <= 0:
        return pd.DataFrame()
    if draw_items is None:
        draw_items = make_bootstrap_draw_indices(
            n=len(np.asarray(data["psi"])),
            block_len=int(args.bootstrap_block_len),
            draws=draws,
            seed=int(args.seed),
        )
    elif len(draw_items) != draws:
        raise ValueError(f"bootstrap draw plan has {len(draw_items)} draws, expected {draws}")
    workers = resolve_bootstrap_workers(args, draws)
    rows: list[dict[str, Any]] = []
    if workers <= 1 or (builder is not None and builder is not build_target):
        builder_fn = build_target if builder is None else builder
        psi = np.asarray(data["psi"], dtype=float)
        dates = pd.to_datetime(pd.Series(data["dates"]), errors="coerce").reset_index(drop=True)
        for b, ix in draw_items:
            boot_data = dict(data)
            boot_data["psi"] = psi[np.asarray(ix, dtype=int)]
            boot_data["dates"] = dates.reset_index(drop=True)
            target_result = builder_fn(target, boot_data, args)
            geom = relative_geometry_from_target(target_result.K_by_state, target_result.C_ref)
            for i, date in enumerate(target_result.state_dates):
                rows.append({"draw": int(b), "date": pd.to_datetime(date).strftime("%Y-%m-%d"), "tau_soft": float(geom["tau_soft"][i])})
    else:
        chunks = chunk_bootstrap_draws(draw_items, workers, int(getattr(args, "bootstrap_chunk_size", 0)))
        payloads = [(target, data, args, chunk) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_bootstrap_chunk_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                rows.extend(future.result())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["draw", "date"]).reset_index(drop=True)


def save_target_outputs(
    target: str,
    target_result: TargetResult,
    data: dict[str, Any],
    args: argparse.Namespace,
    bootstrap_draw_items: list[tuple[int, np.ndarray]] | None = None,
    bootstrap_plan_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = target_directory(args.output_dir, target, args)
    geometry = relative_geometry_from_target(target_result.K_by_state, target_result.C_ref)
    result = result_for_old_helpers(target, target_result, geometry, data["labels"], args.time_bandwidth)
    H = int(data["H"])
    pvars = int(data["pvars"])
    labels = list(data["labels"])
    coord_rows = coordinate_map(labels, H)
    shape = pub.full_coordinate_shape_metrics(result, H, pvars, labels)
    block_df = pub.full_coordinate_block_shape_paths(result, H, pvars, labels)
    shape_rms = shape["shape_rms"]
    dates = pd.to_datetime(result.dates).reset_index(drop=True)
    march_mask = dates.dt.strftime("%Y-%m") == "2020-03"
    march_idx = int(np.where(march_mask.to_numpy())[0][0]) if march_mask.any() else None
    episodes = pub.selected_full_coordinate_episodes(result.dates, result.tau_soft, shape_rms, march_idx)

    ess_abs = 1.0 / np.maximum(np.sum(target_result.weights**2, axis=1), 1e-300)
    ess_norm = effective_support(target_result.weights)
    tau_df = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "tau_soft": result.tau_soft,
            "log_tau_soft": np.log(np.maximum(result.tau_soft, 1e-12)),
            "shape_rms_log_relative": shape_rms,
            "weight_ess": ess_abs,
            "weight_ess_normalized": ess_norm,
        }
    )
    tau_df.to_csv(out_dir / "tau_soft.csv", index=False)
    block_df.to_csv(out_dir / "block_probes.csv", index=False)
    pd.DataFrame(episodes).drop(columns=["idx"]).to_csv(out_dir / "selected_month_metadata.csv", index=False)
    np.save(out_dir / "K_by_state.npy", np.asarray(result.K_hat, dtype=float))
    np.save(out_dir / "C_ref.npy", np.asarray(result.C_hat, dtype=float))
    pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")}).to_csv(out_dir / "state_dates.csv", index=False)
    pd.DataFrame(coord_rows).to_csv(out_dir / "coordinate_map.csv", index=False)

    title = {
        "diagonal_old": "Old diagonal response-score OVK amplification",
        "hac_filtered": f"HAC-aware filtered-score OVK amplification, Bartlett L={args.hac_lags}",
        "hilbert_volterra": (
            "Kernelized Hilbert/Fock Volterra HAC OVK amplification, "
            f"L={args.hac_lags}, gamma={args.signature_gamma:g}"
        ),
    }[target]
    plot_tau(tau_df, out_dir / "tau_soft.png", title, TARGET_LABELS[target])
    plot_selected_stress(result, episodes, H, labels, out_dir / "selected_month_stress.png", title)

    target_meta = metadata_for_json(target_result.metadata)
    if target == "hilbert_volterra":
        np.save(out_dir / "kappa_raw.npy", np.asarray(target_result.metadata["kappa_raw"], dtype=float))
        np.save(out_dir / "kappa_norm.npy", np.asarray(target_result.metadata["kappa_norm"], dtype=float))
        np.save(out_dir / "fock_distance.npy", np.asarray(target_result.metadata["fock_distance"], dtype=float))
        np.save(out_dir / "weights.npy", np.asarray(target_result.weights, dtype=float))

    boot_df = bootstrap_tau(target, data, args, build_target, draw_items=bootstrap_draw_items)
    if len(boot_df):
        boot_df.to_csv(out_dir / "tau_soft_bootstrap_draws.csv", index=False)

    psd_diag = check_psd_symmetric(result.K_hat)
    C_psd_diag = check_psd_symmetric(result.C_hat)
    q = np.full(len(result.tau_soft), 1.0 / max(len(result.tau_soft), 1))
    induced_C = np.einsum("s,sij->ij", q, result.K_hat, optimize=True)
    normalization = {
        "uniform_state_average_tau_soft": float(np.dot(q, result.tau_soft)),
        "uniform_state_average_tau_soft_error": float(abs(np.dot(q, result.tau_soft) - 1.0)),
        "mean_K_matches_C_ref_fro_relative": float(
            np.linalg.norm(induced_C - result.C_hat, ord="fro") / max(np.linalg.norm(result.C_hat, ord="fro"), 1e-12)
        ),
    }
    metadata = {
        "target_type": target,
        "score_source": data["source"],
        "used_synthetic_scores": bool(data.get("used_synthetic_scores", False)),
        "missing_empirical_data": data["missing_empirical_data"],
        "score_matrix_shape": list(np.asarray(data["psi"]).shape),
        "filtered_score_shape": list(target_result.scores.shape),
        "date_range": f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}",
        "panel_date_range": data["panel_date_range"],
        "p": int(np.asarray(data["psi"]).shape[1]),
        "coordinate_map": coord_rows,
        "outcomes": labels,
        "outcome_columns": data["outcome_columns"],
        "horizons": [0, H],
        "L": int(args.hac_lags if target != "diagonal_old" else data["L"]),
        "HAC_lag_L": int(args.hac_lags if target != "diagonal_old" else data["L"]),
        "time_bandwidth": float(args.time_bandwidth),
        "feature_bandwidth": target_meta.get("hilbert_volterra_weights", {}).get("feature_bandwidth"),
        "memory_half_lives": target_meta.get("hilbert_volterra_kernel", {}).get("memory_half_lives"),
        "memory_weights": target_meta.get("hilbert_volterra_kernel", {}).get("memory_weights"),
        "signature_gamma": target_meta.get("hilbert_volterra_kernel", {}).get("signature_gamma"),
        "base_inner_product_method": target_meta.get("base_score_gram", {}).get("base_inner"),
        "C_base_trace": target_meta.get("base_score_gram", {}).get("C_base_trace"),
        "score_gram_rho": target_meta.get("base_score_gram", {}).get("rho"),
        "score_gram_d_rho": target_meta.get("base_score_gram", {}).get("d_rho"),
        "strict_past": target_meta.get("hilbert_volterra_kernel", {}).get("strict_past"),
        "kernel_normalize": target_meta.get("hilbert_volterra_kernel", {}).get("normalize_kernel"),
        "raw_kernel_min": target_meta.get("hilbert_volterra_kernel", {}).get("raw_kernel_min"),
        "raw_kernel_max": target_meta.get("hilbert_volterra_kernel", {}).get("raw_kernel_max"),
        "raw_kernel_diag_range": [
            target_meta.get("hilbert_volterra_kernel", {}).get("raw_kernel_diag_min"),
            target_meta.get("hilbert_volterra_kernel", {}).get("raw_kernel_diag_max"),
        ],
        "normalized_kernel_min": target_meta.get("hilbert_volterra_kernel", {}).get("normalized_kernel_min"),
        "normalized_kernel_max": target_meta.get("hilbert_volterra_kernel", {}).get("normalized_kernel_max"),
        "normalized_kernel_diag_range": [
            target_meta.get("hilbert_volterra_kernel", {}).get("normalized_kernel_diag_min"),
            target_meta.get("hilbert_volterra_kernel", {}).get("normalized_kernel_diag_max"),
        ],
        "Fock_distance_min_median_max": [
            target_meta.get("hilbert_volterra_kernel", {}).get("distance_min"),
            target_meta.get("hilbert_volterra_kernel", {}).get("distance_median_positive"),
            target_meta.get("hilbert_volterra_kernel", {}).get("distance_max"),
        ],
        "rho": float(result.rho),
        "d_rho": float(result.d_rho),
        "reference_rule": target_result.metadata.get("reference_rule"),
        "min_ess": float(np.min(ess_abs)),
        "median_ess": float(np.median(ess_abs)),
        "max_ess": float(np.max(ess_abs)),
        "min_normalized_ess": float(np.min(ess_norm)),
        "median_normalized_ess": float(np.median(ess_norm)),
        "max_normalized_ess": float(np.max(ess_norm)),
        "PSD_diagnostics": {"K_by_state": psd_diag, "C_ref": C_psd_diag},
        "kernel_PSD_diagnostics": {
            "kappa_raw": target_meta.get("hilbert_volterra_kernel", {}).get("raw_kernel_psd"),
            "kappa_norm": target_meta.get("hilbert_volterra_kernel", {}).get("normalized_kernel_psd"),
        },
        "normalization_diagnostics": normalization,
        "package_versions": package_versions(),
        "target_metadata": target_meta,
        "score_metadata": data["score_metadata"],
        "bootstrap": {
            "draws": int(args.bootstrap_draws),
            "block_length": int(args.bootstrap_block_len),
            "workers_requested": int(getattr(args, "bootstrap_workers", 0)),
            "workers_resolved": int(resolve_bootstrap_workers(args, int(args.bootstrap_draws))),
            "chunk_size_requested": int(getattr(args, "bootstrap_chunk_size", 0)),
            "parallel_backend": "process_pool" if resolve_bootstrap_workers(args, int(args.bootstrap_draws)) > 1 else "serial",
            "indices_precomputed": bool(int(args.bootstrap_draws) > 0),
            "paired_across_targets": bool((bootstrap_plan_metadata or {}).get("paired_across_targets", False)),
            "draw_plan_digest": (bootstrap_plan_metadata or {}).get("draw_plan_digest"),
            "draw_indices_file": (bootstrap_plan_metadata or {}).get("draw_indices_file"),
        },
        "note": "This target is a diagnostic moment field, not a time-varying causal effect.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if target == "hilbert_volterra":
        (out_dir / "hilbert_volterra_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "target": target,
        "label": TARGET_LABELS[target],
        "dir": out_dir,
        "tau_df": tau_df,
        "block_df": block_df,
        "metadata": metadata,
        "result": result,
    }


def save_comparison(outputs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    comp = args.output_dir / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    merged: pd.DataFrame | None = None
    top_rows = []
    for item in outputs:
        target = item["target"]
        tau = item["tau_df"][["date", "tau_soft", "shape_rms_log_relative", "weight_ess"]].copy()
        tau = tau.rename(
            columns={
                "tau_soft": f"tau_soft_{target}",
                "shape_rms_log_relative": f"shape_rms_{target}",
                "weight_ess": f"weight_ess_{target}",
            }
        )
        merged = tau if merged is None else merged.merge(tau, on="date", how="outer")
        top = item["tau_df"].sort_values("tau_soft", ascending=False).head(10).reset_index(drop=True)
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            top_rows.append(
                {
                    "target": target,
                    "rank": rank,
                    "date": row.date,
                    "tau_soft": float(row.tau_soft),
                    "shape_rms_log_relative": float(row.shape_rms_log_relative),
                    "weight_ess": float(row.weight_ess),
                }
            )
    assert merged is not None
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(comp / "tau_soft_comparison.csv", index=False)
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(comp / "top_months.csv", index=False)

    tau_cols = [c for c in merged.columns if c.startswith("tau_soft_")]
    corr = merged[tau_cols].corr(min_periods=12)
    corr.to_csv(comp / "target_correlations.csv")

    old_top = set(top_df.loc[top_df["target"].eq("diagonal_old"), "date"])
    new_rows = []
    for target in [item["target"] for item in outputs if item["target"] != "diagonal_old"]:
        sub = top_df[top_df["target"].eq(target)]
        for row in sub.itertuples(index=False):
            if row.date not in old_top:
                new_rows.append(row._asdict())
    pd.DataFrame(new_rows).to_csv(comp / "newly_highlighted_months.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    for item in outputs:
        col = f"tau_soft_{item['target']}"
        if col in merged:
            ax.plot(pd.to_datetime(merged["date"]), merged[col], label=item["label"], linewidth=1.25)
    ax.axhline(1.0, linewidth=0.9, color="black", linestyle="--", label="reference = 1")
    ax.set_title("Old vs HAC-filtered vs Hilbert-Volterra OVK amplification")
    ax.set_ylabel("tau_soft")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(comp / "tau_soft_comparison.png", dpi=180)
    plt.close(fig)

    summary = build_summary(outputs, top_df, corr, pd.DataFrame(new_rows), args)
    (comp / "summary.md").write_text(summary, encoding="utf-8")


def build_summary(
    outputs: list[dict[str, Any]],
    top_df: pd.DataFrame,
    corr: pd.DataFrame,
    new_df: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    by_target = {item["target"]: item for item in outputs}
    source = next(iter(outputs))["metadata"]["score_source"] if outputs else "not available"
    missing = next(iter(outputs))["metadata"].get("missing_empirical_data", []) if outputs else []

    def top_lines(target: str) -> str:
        sub = top_df[top_df["target"].eq(target)].head(5)
        if sub.empty:
            return "- not run"
        return "\n".join(f"- {row.date}: tau_soft={row.tau_soft:.3f}" for row in sub.itertuples(index=False))

    corr_md = markdown_table(corr.round(3), index=True) if not corr.empty else "not available"
    new_md = (
        markdown_table(new_df[["target", "rank", "date", "tau_soft"]].head(20), index=False)
        if not new_df.empty
        else "None in the top-10 lists."
    )
    diag_lines = []
    for target, item in by_target.items():
        meta = item["metadata"]
        norm = meta["normalization_diagnostics"]
        psd = meta["PSD_diagnostics"]["K_by_state"]
        extra = ""
        if target == "hilbert_volterra":
            kpsd = meta.get("kernel_PSD_diagnostics", {}).get("kappa_norm") or {}
            extra = f", kappa_norm min eig {kpsd.get('min_eigenvalue', float('nan')):.3e}"
        diag_lines.append(
            f"- {target}: min eig {psd['min_eigenvalue']:.3e}, "
            f"mean tau error {norm['uniform_state_average_tau_soft_error']:.3e}, "
            f"ESS median {meta['median_ess']:.1f}{extra}"
        )
    caveat = "Empirical monetary-policy data were available." if not missing else f"Empirical data missing; smoke scores used. Missing files: {missing}."
    return f"""# Monetary LP Memory-Target OVK Summary

{caveat}

Score source: `{source}`.

## 1. What changed from the old target

- Old: `K_old(s) = sum_t w(s,t) psi_t psi_t'`.
- HAC: `K_HAC(s) = sum_t w_time(s,t) Z_t Z_t'`, with Bartlett-filtered scores and `L={args.hac_lags}`.
- Hilbert-Volterra: `K_HV(s) = sum_t w_HV(s,t) Z_t Z_t'`, where `w_HV` uses a normalized infinite-level Fock Gram matrix.
- The old finite `Phi_t` prototype is not used by the main empirical target: no rank `r`, no level `M`, and no PCA feature truncation.

## 2. Mathematical target

`w_HV(s,t)` is proportional to the old calendar-time kernel times `exp(-d_Fock(s,t)^2/(2 h_Fock^2))`. The distance is computed from the normalized Fock Gram matrix, not from explicit tensor features.

## 3. Why the target is Hilbert-space consistent

The score vector `psi_t` lives in the original coefficient/influence space `H` represented here by `R^125`. The nonlinear history object lives in the weighted Fock space `F_beta(H)`, but only its Gram matrix is required. The final `K_HV(s)` remains a positive `p x p` operator on `H`, so the old ridge-soft relative-moment machinery applies unchanged.

## 4. Why HAC_filtered is HAC-aware

With `Z_t = (1/sqrt(L+1)) sum_ell psi_{{t-ell}}`, expanding `Z_t Z_t'` adds all cross-period products `psi_{{t-ell}} psi_{{t-m}}'`. Grouping by lag gives Bartlett weights `1 - h/(L+1)` without forming a giant lag-stack covariance.

## 5. What nonlinear Volterra/Fock state similarity adds

The Hilbert-Volterra target borrows from months with similar ordered score-history geometry. The recursion includes all finite-sample tensor orders; `gamma={args.signature_gamma:g}` controls high-order weighting, not truncation. Memory half-lives are `{[float(x) for x in args.memory_half_lives]}` with `{args.memory_weights}` weights.

## 6. Empirical comparison

Top old diagonal months:
{top_lines('diagonal_old')}

Top HAC-filtered months:
{top_lines('hac_filtered')}

Top Hilbert-Volterra months:
{top_lines('hilbert_volterra')}

Tau-path correlations:

{corr_md}

Months newly highlighted by HAC or Hilbert-Volterra top-10 lists:

{new_md}

Block-probe CSVs in each target directory report outcome-group and horizon-bucket differences.

## 7. Diagnostics

{chr(10).join(diag_lines)}

## 8. Caveats

These are diagnostic moment fields, not new time-varying causal effects. The HAC target is a filtered long-run exposure target, not a conventional Newey-West standard error. Hilbert-Volterra similarity depends on memory kernel, gamma, base inner product, and smoothing bandwidth; conventional HAC inference remains separate unless that inferential covariance target is explicitly defined.
"""


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a small markdown table without requiring pandas[tabulate]."""
    table = df.copy()
    if index:
        table = table.reset_index().rename(columns={"index": ""})
    columns = [str(c) for c in table.columns]
    rows = [columns, ["---"] * len(columns)]
    for values in table.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_empirical_scores(args)
    bootstrap_draw_items: list[tuple[int, np.ndarray]] | None = None
    if int(args.bootstrap_draws) > 0:
        bootstrap_draw_items = make_bootstrap_draw_indices(
            n=len(np.asarray(data["psi"])),
            block_len=int(args.bootstrap_block_len),
            draws=int(args.bootstrap_draws),
            seed=int(args.seed),
        )
    bootstrap_plan_metadata = save_bootstrap_draw_plan(args.output_dir, args, data, bootstrap_draw_items or [])
    outputs = []
    for target in args.targets:
        target_result = build_target(target, data, args)
        outputs.append(
            save_target_outputs(
                target,
                target_result,
                data,
                args,
                bootstrap_draw_items=bootstrap_draw_items,
                bootstrap_plan_metadata=bootstrap_plan_metadata,
            )
        )
    save_comparison(outputs, args)
    if args.add_rotation_diagnostics:
        from route_rotation import RouteRotationConfig, run_rotation_diagnostics

        routes = tuple(target_directory(args.output_dir, target, args).name for target in ["diagonal_old", "hac_filtered", "hilbert_volterra"])
        run_rotation_diagnostics(
            RouteRotationConfig(
                targets_dir=args.output_dir,
                comparison_dir=args.output_dir / "comparison",
                routes=routes,
                rotation_reference=args.rotation_reference,
                lambda_min=float(args.rotation_lambda_min),
                lambda_max=float(args.rotation_lambda_max),
                lambda_count=int(args.rotation_lambda_count),
                min_anisotropy=float(args.min_rotation_anisotropy),
            )
        )
    print(f"Wrote outputs to {args.output_dir}")
    if data["missing_empirical_data"]:
        print(f"Empirical data missing; smoke scores used: {data['missing_empirical_data']}")
    elif data.get("used_synthetic_scores", False):
        print(f"Synthetic smoke score matrix: {data['psi'].shape[0]} x {data['psi'].shape[1]}")
    else:
        print(f"Empirical score matrix: {data['psi'].shape[0]} x {data['psi'].shape[1]}")


if __name__ == "__main__":
    main()
