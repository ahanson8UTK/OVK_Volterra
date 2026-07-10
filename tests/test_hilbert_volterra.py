from __future__ import annotations

import argparse
import itertools
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SCRIPTS = ROOT / "scripts"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_monetary_lp_memory_targets as runner  # noqa: E402
from hilbert_volterra import (  # noqa: E402
    HilbertVolterraKernelConfig,
    check_kernel_psd,
    check_moment_psd,
    compute_base_score_gram,
    compute_hilbert_volterra_gram,
    make_hilbert_volterra_target,
    make_hilbert_volterra_weights,
)
from time_series_targets import make_bartlett_filtered_scores, relative_geometry_from_target  # noqa: E402


def _scores(seed: int = 77, n: int = 36, p: int = 7) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 3))
    load = rng.normal(size=(3, p))
    psi = factors @ load + 0.15 * rng.normal(size=(n, p))
    dates = pd.Series(pd.date_range("2001-01-01", periods=n, freq="MS"))
    return psi, dates


def _brute_force_kappa(c: np.ndarray, a: float, gamma: float) -> np.ndarray:
    T = c.shape[0]
    out = np.ones((T, T), dtype=float)

    def subsequences(endpoint: int, m: int) -> list[tuple[int, ...]]:
        return list(itertools.combinations(range(endpoint), m))

    def weight(endpoint: int, seq: tuple[int, ...]) -> float:
        if not seq:
            return 1.0
        nodes = (*seq, endpoint)
        val = 1.0
        for left, right in zip(nodes[:-1], nodes[1:]):
            val *= a ** (right - left)
        return val

    for t in range(T):
        for u in range(T):
            total = 1.0
            for m in range(1, min(t, u) + 1):
                level = 0.0
                for I in subsequences(t, m):
                    wI = weight(t, I)
                    for J in subsequences(u, m):
                        prod = 1.0
                        for i, j in zip(I, J):
                            prod *= c[i, j]
                        level += wI * weight(u, J) * prod
                total += (gamma**m) * level
            out[t, u] = total
    return out


def test_base_score_gram_euclidean_and_reference_soft_are_psd() -> None:
    psi, _ = _scores()
    for method in ["euclidean", "reference_soft"]:
        c, meta = compute_base_score_gram(psi, method=method)
        diag = np.diag(c)
        assert np.isfinite(c).all()
        assert np.all(diag > 0.0)
        assert check_kernel_psd(c)["ok"], meta


def test_infinite_level_recursion_matches_bruteforce_toy() -> None:
    psi, dates = _scores(n=5, p=2)
    c, _ = compute_base_score_gram(psi, method="euclidean")
    half_life = 2.0
    a = np.exp(-np.log(2.0) / half_life)
    gamma = 0.03
    cfg = HilbertVolterraKernelConfig(
        memory_half_lives=(half_life,),
        memory_weights="equal",
        gamma=gamma,
        base_inner="euclidean",
    )
    raw, _, _, _ = compute_hilbert_volterra_gram(c, dates, cfg)
    brute = _brute_force_kappa(c, a=a, gamma=gamma)
    assert np.allclose(raw, brute, atol=1e-10, rtol=1e-10)


def test_kernel_psd_normalization_and_distance_identity() -> None:
    psi, dates = _scores(n=28, p=5)
    c, _ = compute_base_score_gram(psi, method="reference_soft")
    cfg = HilbertVolterraKernelConfig(memory_half_lives=(3.0, 12.0), gamma=0.04)
    raw, norm, distance, _ = compute_hilbert_volterra_gram(c, dates, cfg)
    assert check_kernel_psd(raw)["ok"]
    assert check_kernel_psd(norm)["ok"]
    assert np.allclose(np.diag(norm), 1.0, atol=1e-10)
    assert np.allclose(np.diag(distance), 0.0, atol=1e-10)
    expected = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * norm))
    assert np.allclose(distance, expected, atol=1e-10)
    assert np.isfinite(distance).all()
    assert np.min(distance) >= -1e-12


def test_hilbert_volterra_weights_are_normalized_and_ess_behaves() -> None:
    psi, dates = _scores(n=32, p=5)
    c, _ = compute_base_score_gram(psi, method="reference_soft")
    cfg = HilbertVolterraKernelConfig(memory_half_lives=(3.0, 12.0), gamma=0.04)
    _, norm, distance, _ = compute_hilbert_volterra_gram(c, dates, cfg)
    W_small, meta_small = make_hilbert_volterra_weights(dates, norm, distance, time_bandwidth=6.0, feature_bandwidth=0.25, base_time_kernel="gaussian")
    W_large, meta_large = make_hilbert_volterra_weights(dates, norm, distance, time_bandwidth=6.0, feature_bandwidth=2.0, base_time_kernel="gaussian")
    assert np.all(W_small >= 0.0)
    assert np.all(W_large >= 0.0)
    assert np.allclose(W_small.sum(axis=1), 1.0)
    assert np.allclose(W_large.sum(axis=1), 1.0)
    assert meta_small["ESS_min"] > 0.0
    assert meta_large["ESS_median"] >= 0.75 * meta_small["ESS_median"]


def test_hilbert_volterra_moment_psd_and_reference_identity() -> None:
    psi, dates = _scores(n=38, p=9)
    target = make_hilbert_volterra_target(
        psi,
        dates,
        hac_lags=4,
        config=HilbertVolterraKernelConfig(memory_half_lives=(3.0, 12.0), gamma=0.04),
        time_bandwidth=5.0,
        reference="grid_induced",
    )
    diag = check_moment_psd(target.K_by_state, target.C_ref)
    assert diag["K_by_state"]["ok"]
    assert diag["C_ref"]["ok"]
    q = np.full(target.K_by_state.shape[0], 1.0 / target.K_by_state.shape[0])
    induced = np.einsum("s,sij->ij", q, target.K_by_state, optimize=True)
    assert np.allclose(induced, target.C_ref, atol=1e-10)
    geom = relative_geometry_from_target(target.K_by_state, target.C_ref)
    assert abs(float(q @ geom["tau_soft"]) - 1.0) < 1e-10


def test_bartlett_equivalence_still_holds_for_filtered_scores() -> None:
    rng = np.random.default_rng(13)
    psi = rng.normal(size=(18, 4))
    L = 4
    Z, valid, _ = make_bartlett_filtered_scores(psi, L=L)
    left = (Z.T @ Z) / len(Z)
    right = np.zeros((psi.shape[1], psi.shape[1]))
    for t in valid:
        for ell in range(L + 1):
            for m in range(L + 1):
                right += np.outer(psi[t - ell], psi[t - m]) / (L + 1)
    right /= len(valid)
    assert np.allclose(left, right, atol=1e-12)
    weights = np.asarray([(L + 1 - h) / (L + 1) for h in range(L + 1)])
    assert np.allclose(weights, [1.0, 0.8, 0.6, 0.4, 0.2])


def test_main_runner_does_not_call_legacy_finite_volterra_features() -> None:
    source = inspect.getsource(runner)
    banned = [
        "make_projected_score_path",
        "make_volterra_signature_features",
        "make_volterra_nonlinear_target",
        "pca_dim=",
    ]
    for token in banned:
        assert token not in source
    assert "make_hilbert_volterra_target" in source


def test_bootstrap_indices_are_deterministic_and_chunked() -> None:
    draws1 = runner.make_bootstrap_draw_indices(n=25, block_len=5, draws=7, seed=123)
    draws2 = runner.make_bootstrap_draw_indices(n=25, block_len=5, draws=7, seed=123)
    assert [b for b, _ in draws1] == list(range(7))
    for (b1, ix1), (b2, ix2) in zip(draws1, draws2):
        assert b1 == b2
        assert np.array_equal(ix1, ix2)
    chunks = runner.chunk_bootstrap_draws(draws1, workers=3, chunk_size=2)
    flattened = [b for chunk in chunks for b, _ in chunk]
    assert flattened == list(range(7))
    assert all(len(chunk) <= 2 for chunk in chunks)


def test_serial_bootstrap_tau_is_deterministic_and_sorted() -> None:
    psi, dates = _scores(n=24, p=5)
    args = argparse.Namespace(
        bootstrap_draws=3,
        bootstrap_block_len=4,
        bootstrap_workers=1,
        bootstrap_chunk_size=0,
        seed=321,
        time_bandwidth=3.0,
    )
    data = {"psi": psi, "dates": dates}
    left = runner.bootstrap_tau("diagonal_old", data, args)
    right = runner.bootstrap_tau("diagonal_old", data, args)
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 3 * len(dates)
    assert left[["draw", "date"]].equals(left.sort_values(["draw", "date"])[["draw", "date"]].reset_index(drop=True))


def test_shared_bootstrap_draw_plan_is_saved_and_controls_targets(tmp_path: Path) -> None:
    psi, dates = _scores(n=24, p=5)
    args = argparse.Namespace(
        bootstrap_draws=3,
        bootstrap_block_len=4,
        bootstrap_workers=1,
        bootstrap_chunk_size=0,
        seed=321,
        time_bandwidth=3.0,
    )
    data = {"psi": psi, "dates": dates}
    draw_items = runner.make_bootstrap_draw_indices(n=len(psi), block_len=4, draws=3, seed=321)
    metadata = runner.save_bootstrap_draw_plan(tmp_path, args, data, draw_items)
    saved = np.load(tmp_path / "bootstrap_draw_indices.npy")
    assert saved.shape == (3, len(psi))
    assert metadata["paired_across_targets"] is True
    assert metadata["draw_plan_digest"] == runner.bootstrap_draw_plan_digest(draw_items)
    assert (tmp_path / "bootstrap_draw_plan_metadata.json").exists()

    args_other_seed = argparse.Namespace(**{**vars(args), "seed": 999})
    left = runner.bootstrap_tau("diagonal_old", data, args, draw_items=draw_items)
    right = runner.bootstrap_tau("diagonal_old", data, args_other_seed, draw_items=draw_items)
    pd.testing.assert_frame_equal(left, right)
