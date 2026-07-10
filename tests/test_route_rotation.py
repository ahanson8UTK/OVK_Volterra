from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from route_rotation import (  # noqa: E402
    PAIR_DEFS,
    PROBE_DISPLAY,
    RouteRotationConfig,
    build_trace_class_probes,
    compute_commutator_rotation,
    run_rotation_diagnostics,
    yosida_alignment_for_pair,
)


def _coordinate_map(H: int = 24) -> pd.DataFrame:
    labels = ["IP", "CPI", "Unemployment", "2Y yield", "BAA-10Y spread"]
    rows = []
    for h in range(H + 1):
        for j, label in enumerate(labels):
            rows.append({"coordinate": h * len(labels) + j, "horizon_months": h, "outcome": label})
    return pd.DataFrame(rows)


def _rot(p: int, angle: float) -> np.ndarray:
    R = np.eye(p)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    R[:2, :2] = [[c, -s], [s, c]]
    return R


def test_probe_construction_returns_trace_class_psd_probes() -> None:
    p = 125
    vals = np.linspace(0.4, 2.0, p)
    S_star = np.diag(vals)
    probes = build_trace_class_probes(S_star, _coordinate_map(H=24))
    assert set(probes) == set(PROBE_DISPLAY)
    for probe in probes.values():
        assert probe.available
        assert np.allclose(probe.Q, probe.Q.T)
        eigvals = np.linalg.eigvalsh(probe.Q)
        assert eigvals.min() >= -1e-10
        assert abs(float(np.trace(probe.Q)) - 1.0) < 1e-10
        assert probe.diagnostics["rank_estimate"] > 0

    unavailable = build_trace_class_probes(np.eye(10), _coordinate_map(H=1))
    assert not unavailable["long_horizons"].available
    assert "reason" in unavailable["long_horizons"].diagnostics


def test_yosida_alignment_self_and_symmetry() -> None:
    p = 6
    A = np.diag(np.linspace(0.5, 2.0, p))
    B = _rot(p, 0.35) @ np.diag(np.linspace(0.7, 1.8, p)) @ _rot(p, 0.35).T
    Q = np.eye(p) / p
    lambdas = np.logspace(-2, 1, 9)
    self_omega, _, self_valid = yosida_alignment_for_pair(A, A, Q, lambdas)
    assert np.all(self_valid)
    assert np.allclose(self_omega, 1.0, atol=1e-10)

    lr, _, valid_lr = yosida_alignment_for_pair(A, B, Q, lambdas)
    rl, _, valid_rl = yosida_alignment_for_pair(B, A, Q, lambdas)
    assert np.array_equal(valid_lr, valid_rl)
    assert np.allclose(lr[valid_lr], rl[valid_rl], atol=1e-10)


def test_scale_invariance_after_tau_normalization() -> None:
    p = 5
    dates = pd.Series(pd.date_range("2000-01-01", periods=1, freq="MS"))
    A = np.diag([0.5, 0.8, 1.0, 1.4, 1.8])
    Abar_1 = A / (np.trace(A) / p)
    Abar_2 = (7.0 * A) / (np.trace(7.0 * A) / p)
    assert np.allclose(Abar_1, Abar_2)

    Q = np.eye(p) / p
    omega, _, valid = yosida_alignment_for_pair(Abar_1, Abar_2, Q, np.logspace(-2, 2, 11))
    assert np.all(valid)
    assert np.allclose(omega, 1.0, atol=1e-10)

    shapes = {"D": Abar_1[None, :, :], "H": Abar_2[None, :, :], "V": Abar_1[None, :, :]}
    by_date, _, _, _ = compute_commutator_rotation(shapes, dates, np.eye(p), min_anisotropy=0.01)
    dh = by_date[by_date["pair_key"].eq("D_H")].iloc[0]
    assert dh.valid
    assert abs(float(dh.commutator_index)) < 1e-10


def test_same_eigenbasis_has_zero_commutator_but_yosida_can_differ() -> None:
    p = 5
    dates = pd.Series(pd.date_range("2000-01-01", periods=1, freq="MS"))
    A = np.diag([0.5, 0.8, 1.0, 1.5, 2.2])
    B = np.diag([1.8, 1.4, 1.0, 0.7, 0.4])
    A = A / (np.trace(A) / p)
    B = B / (np.trace(B) / p)
    shapes = {"D": A[None, :, :], "H": B[None, :, :], "V": A[None, :, :]}
    by_date, _, _, _ = compute_commutator_rotation(shapes, dates, np.eye(p), min_anisotropy=0.01)
    dh = by_date[by_date["pair_key"].eq("D_H")].iloc[0]
    assert abs(float(dh.commutator_hs_norm)) < 1e-12
    assert abs(float(dh.commutator_index)) < 1e-12

    omega, _, valid = yosida_alignment_for_pair(A, B, np.eye(p) / p, np.logspace(-2, 2, 21))
    assert np.all(valid)
    assert float(np.nanmean(omega)) < 0.999


def test_same_spectrum_rotation_detects_direction_change() -> None:
    p = 6
    A = np.diag([0.4, 0.8, 1.0, 1.2, 1.5, 2.0])
    A = A / (np.trace(A) / p)
    R = _rot(p, 0.55)
    B = R @ A @ R.T
    lambdas = np.logspace(-2, 2, 21)
    eig_stress_A = [np.trace(A @ np.linalg.solve(lam * np.eye(p) + A, np.eye(p))) for lam in lambdas]
    eig_stress_B = [np.trace(B @ np.linalg.solve(lam * np.eye(p) + B, np.eye(p))) for lam in lambdas]
    assert np.allclose(eig_stress_A, eig_stress_B, atol=1e-10)

    Q = np.eye(p) / p
    omega, _, valid = yosida_alignment_for_pair(A, B, Q, lambdas)
    assert np.all(valid)
    assert float(np.nanmean(omega)) < 0.999

    dates = pd.Series(pd.date_range("2000-01-01", periods=1, freq="MS"))
    shapes = {"D": A[None, :, :], "H": B[None, :, :], "V": A[None, :, :]}
    by_date, _, _, _ = compute_commutator_rotation(shapes, dates, np.eye(p), min_anisotropy=0.01)
    dh = by_date[by_date["pair_key"].eq("D_H")].iloc[0]
    assert float(dh.commutator_index) > 0.01


def test_commutator_antisymmetry_and_norm() -> None:
    p = 5
    A = np.diag(np.linspace(0.5, 2.0, p))
    B = _rot(p, 0.4) @ np.diag(np.linspace(2.0, 0.5, p)) @ _rot(p, 0.4).T
    AB = A @ B - B @ A
    BA = B @ A - A @ B
    assert np.allclose(AB, -BA)
    assert abs(float(np.linalg.norm(AB, ord="fro") - np.linalg.norm(BA, ord="fro"))) < 1e-12


def _write_route(folder: Path, key: str, dates: pd.Series, coordinate_map: pd.DataFrame, angle: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    p = len(coordinate_map)
    R = _rot(p, angle)
    mats = []
    tau = []
    for i in range(len(dates)):
        diag = 0.7 + 0.5 * np.sin(np.linspace(0, np.pi, p) + 0.2 * i) ** 2
        diag[:3] += 0.1 * i
        A = R @ np.diag(diag) @ R.T
        mats.append(A)
        tau.append(float(np.trace(A) / p))
    K = np.stack(mats)
    np.save(folder / "K_by_state.npy", K)
    np.save(folder / "C_ref.npy", np.eye(p))
    pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")}).to_csv(folder / "state_dates.csv", index=False)
    coordinate_map.to_csv(folder / "coordinate_map.csv", index=False)
    pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "tau_soft": tau}).to_csv(folder / "tau_soft.csv", index=False)
    (folder / "metadata.json").write_text('{"rho": 1e-6}', encoding="utf-8")


def test_rotation_diagnostics_output_smoke(tmp_path: Path) -> None:
    targets = tmp_path / "targets"
    comparison = targets / "comparison"
    dates = pd.Series(pd.date_range("2010-01-01", periods=7, freq="MS"))
    cmap = _coordinate_map(H=13)
    route_specs = [
        ("diagonal_old", "D", 0.0),
        ("hac_filtered_L12", "H", 0.2),
        ("hilbert_volterra_L12_gamma005_memory_3_12_36", "V", 0.45),
    ]
    for folder, key, angle in route_specs:
        _write_route(targets / folder, key, dates, cmap, angle)
    metadata = run_rotation_diagnostics(
        RouteRotationConfig(
            targets_dir=targets,
            comparison_dir=comparison,
            lambda_count=5,
            routes=tuple(folder for folder, _, _ in route_specs),
            min_anisotropy=0.01,
        )
    )
    assert metadata["whether_K_by_state_had_to_be_recomputed"] is False
    required = [
        "probed_yosida_alignment_long.csv",
        "probed_yosida_alignment_by_date.csv",
        "probed_yosida_alignment_summary.csv",
        "probed_yosida_rotation_compact_table.csv",
        "probed_yosida_rotation_compact_table_p90.csv",
        "probed_yosida_rotation_full_soft_timeseries.png",
        "probed_yosida_rotation_compact_heatmap.png",
        "probed_yosida_rotation_p90_heatmap.png",
        "probed_yosida_alignment_selected_dates.png",
        "probed_yosida_alignment_selected_dates.csv",
        "commutator_rotation_by_date.csv",
        "commutator_rotation_summary.csv",
        "commutator_rotation_top_dates.csv",
        "commutator_rotation_timeseries.png",
        "route_anisotropy_paths.png",
        "commutator_vs_anisotropy.png",
        "commutator_rotation_summary.png",
        "rotation_diagnostics_metadata.json",
        "summary.md",
    ]
    for name in required:
        assert (comparison / name).exists(), name
    compact = pd.read_csv(comparison / "probed_yosida_rotation_compact_table.csv")
    assert compact.shape[0] == len(PAIR_DEFS)

