"""Comparison diagnostics for publication-grade OVK versus SEC robustness."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def principal_angles(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """Return principal angles in degrees between two column spaces."""
    q = min(Va.shape[1], Vb.shape[1])
    s = np.linalg.svd(np.asarray(Va).T @ np.asarray(Vb), compute_uv=False)
    s = np.clip(s[:q], -1.0, 1.0)
    return np.degrees(np.arccos(s))


def basis_match(Vbase: np.ndarray, Vother: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match basis vectors by absolute correlation."""
    C = np.abs(np.asarray(Vbase).T @ np.asarray(Vother))
    row_ind, col_ind = linear_sum_assignment(-C)
    order = np.zeros(Vbase.shape[1], dtype=int)
    corr = np.zeros(Vbase.shape[1], dtype=float)
    for r, c in zip(row_ind, col_ind):
        order[r] = c
        corr[r] = C[r, c]
    return order, corr


def _common_path(
    dates: pd.Series | np.ndarray,
    sec_values: np.ndarray,
    baseline: pd.DataFrame,
    baseline_col: str,
) -> pd.DataFrame:
    left = pd.DataFrame({"date": pd.to_datetime(dates), "sec": np.asarray(sec_values, dtype=float)})
    right = baseline[["date", baseline_col]].copy()
    right["date"] = pd.to_datetime(right["date"])
    right = right.rename(columns={baseline_col: "baseline"})
    return left.merge(right, on="date", how="inner").sort_values("date").reset_index(drop=True)


def crisis_window_mask(dates: pd.Series) -> np.ndarray:
    """Return mask for 1998-1999, 2007-2008, and 2020 crisis windows."""
    d = pd.to_datetime(dates)
    return (
        ((d.dt.year >= 1998) & (d.dt.year <= 1999))
        | ((d.dt.year >= 2007) & (d.dt.year <= 2008))
        | (d.dt.year == 2020)
    ).to_numpy()


def tau_path_comparison(
    variant: str,
    label: str,
    dates: pd.Series,
    tau_sec: np.ndarray,
    baseline_path: pd.DataFrame,
) -> dict[str, object]:
    """Compare baseline publication-grade tau_t with SEC tau_t."""
    common = _common_path(dates, tau_sec, baseline_path, "tau")
    if len(common) < 3:
        raise ValueError(f"Too few common dates for tau comparison: {variant}")
    corr = float(np.corrcoef(common["baseline"], common["sec"])[0, 1])
    rmse = float(np.sqrt(np.mean((common["baseline"] - common["sec"]) ** 2)))
    rank_corr = float(spearmanr(common["baseline"], common["sec"]).correlation)
    mask = crisis_window_mask(common["date"])
    crisis_corr = float(np.corrcoef(common.loc[mask, "baseline"], common.loc[mask, "sec"])[0, 1]) if mask.sum() >= 3 else np.nan
    march = common[pd.to_datetime(common["date"]).dt.strftime("%Y-%m").eq("2020-03")]
    return {
        "variant": variant,
        "label": label,
        "n_common": int(len(common)),
        "tau_corr": corr,
        "tau_rmse": rmse,
        "tau_spearman": rank_corr,
        "tau_crisis_window_corr": crisis_corr,
        "baseline_tau_mean": float(common["baseline"].mean()),
        "sec_tau_mean": float(common["sec"].mean()),
        "baseline_tau_max": float(common["baseline"].max()),
        "sec_tau_max": float(common["sec"].max()),
        "baseline_tau_max_month": pd.to_datetime(common.loc[common["baseline"].idxmax(), "date"]).strftime("%Y-%m"),
        "sec_tau_max_month": pd.to_datetime(common.loc[common["sec"].idxmax(), "date"]).strftime("%Y-%m"),
        "march_2020_baseline_tau": float(march["baseline"].iloc[0]) if len(march) else np.nan,
        "march_2020_sec_tau": float(march["sec"].iloc[0]) if len(march) else np.nan,
    }


def top_amplification_months(
    variant: str,
    label: str,
    dates: pd.Series,
    tau_sec: np.ndarray,
    A_sec: np.ndarray,
    baseline_path: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return top-month table and overlap diagnostics."""
    sec_df = pd.DataFrame({"date": pd.to_datetime(dates), "tau": tau_sec})
    for j in range(A_sec.shape[1]):
        sec_df[f"A{j+1}{j+1}"] = A_sec[:, j, j]
    base = baseline_path.copy()
    base["date"] = pd.to_datetime(base["date"])
    base_top = base.sort_values("tau", ascending=False).head(top_n).copy()
    sec_top = sec_df.sort_values("tau", ascending=False).head(top_n).copy()
    base_set = set(base_top["date"].dt.strftime("%Y-%m"))
    sec_set = set(sec_top["date"].dt.strftime("%Y-%m"))
    rows = []
    for source, frame in [("publication_grade", base_top), ("SEC", sec_top)]:
        for rank, (_, row) in enumerate(frame.iterrows(), start=1):
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "source": source,
                    "rank": rank,
                    "date": pd.to_datetime(row["date"]).strftime("%Y-%m"),
                    "tau": float(row["tau"]),
                    "in_other_top10": pd.to_datetime(row["date"]).strftime("%Y-%m") in (sec_set if source == "publication_grade" else base_set),
                    **{
                        f"A{j+1}{j+1}": float(row.get(f"A{j+1}{j+1}", np.nan))
                        for j in range(A_sec.shape[1])
                    },
                }
            )
    march_mask_sec = sec_df["date"].dt.strftime("%Y-%m").eq("2020-03").to_numpy()
    march_mask_base = base["date"].dt.strftime("%Y-%m").eq("2020-03").to_numpy()
    sec_order = np.argsort(sec_df["tau"].to_numpy())[::-1]
    base_order = np.argsort(base["tau"].to_numpy())[::-1]
    sec_march_idx = int(np.where(march_mask_sec)[0][0]) if march_mask_sec.any() else None
    base_march_idx = int(np.where(march_mask_base)[0][0]) if march_mask_base.any() else None
    diag = {
        "variant": variant,
        "label": label,
        "top10_overlap_count": int(len(base_set & sec_set)),
        "top10_overlap_months": ", ".join(sorted(base_set & sec_set)),
        "march_2020_sec_tau": float(sec_df.loc[sec_march_idx, "tau"]) if sec_march_idx is not None else np.nan,
        "march_2020_sec_rank": int(np.where(sec_order == sec_march_idx)[0][0] + 1) if sec_march_idx is not None else np.nan,
        "march_2020_baseline_tau": float(base.loc[base_march_idx, "tau"]) if base_march_idx is not None else np.nan,
        "march_2020_baseline_rank": int(np.where(base_order == base_march_idx)[0][0] + 1) if base_march_idx is not None else np.nan,
    }
    return pd.DataFrame(rows), diag


def basis_diag_correlations(
    variant: str,
    label: str,
    dates: pd.Series,
    A_sec: np.ndarray,
    baseline_path: pd.DataFrame,
) -> pd.DataFrame:
    """Compare baseline and SEC diagonal A_jj paths in the same basis."""
    rows = []
    for j in range(A_sec.shape[1]):
        col = f"A{j+1}{j+1}"
        if col not in baseline_path.columns:
            continue
        common = _common_path(dates, A_sec[:, j, j], baseline_path, col)
        rows.append(
            {
                "variant": variant,
                "label": label,
                "basis": j + 1,
                "A_diag_path_corr": float(np.corrcoef(common["baseline"], common["sec"])[0, 1]),
                "A_diag_rmse": float(np.sqrt(np.mean((common["baseline"] - common["sec"]) ** 2))),
                "baseline_sd": float(common["baseline"].std(ddof=0)),
                "sec_sd": float(common["sec"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def state_manifold_extreme_states(
    variant: str,
    label: str,
    dates: pd.Series,
    embedding: np.ndarray,
    states: np.ndarray,
    tau_sec: np.ndarray,
    top_n: int = 12,
) -> pd.DataFrame:
    """Identify high-amplification states and nearest named crisis months."""
    d = pd.to_datetime(dates)
    emb = np.asarray(embedding, dtype=float)
    S = np.asarray(states, dtype=float)
    rows = []
    top_idx = np.argsort(tau_sec)[::-1][:top_n]
    named_months = ["1998-10", "1999-01", "2007-09", "2008-10", "2020-03"]
    for rank, idx in enumerate(top_idx, start=1):
        rows.append(
            {
                "variant": variant,
                "label": label,
                "record_type": "top_sec_tau",
                "rank": rank,
                "date": d.iloc[idx].strftime("%Y-%m"),
                "tau_sec": float(tau_sec[idx]),
                "phi1": float(emb[idx, 0]),
                "phi2": float(emb[idx, 1]) if emb.shape[1] > 1 else np.nan,
                "nearest_top_sec_tau_month": "",
                "state_distance_to_nearest_top": np.nan,
            }
        )
    for month in named_months:
        mask = d.dt.strftime("%Y-%m").eq(month).to_numpy()
        if not mask.any():
            continue
        idx = int(np.where(mask)[0][0])
        distances = np.linalg.norm(S[top_idx] - S[idx], axis=1)
        nearest = int(top_idx[int(np.argmin(distances))])
        rows.append(
            {
                "variant": variant,
                "label": label,
                "record_type": "named_crisis_state",
                "rank": np.nan,
                "date": month,
                "tau_sec": float(tau_sec[idx]),
                "phi1": float(emb[idx, 0]),
                "phi2": float(emb[idx, 1]) if emb.shape[1] > 1 else np.nan,
                "nearest_top_sec_tau_month": d.iloc[nearest].strftime("%Y-%m"),
                "state_distance_to_nearest_top": float(np.min(distances)),
            }
        )
    return pd.DataFrame(rows)
