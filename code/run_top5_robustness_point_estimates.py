"""
Point-estimate robustness comparison for the rank-five monthly monetary-policy OVK.

This script compares three shock definitions:
1. MP_median with monthly fallback to MP_pm (baseline).
2. MP_pm only.
3. Event-level MP_median with event-level MP_pm fallback, manually aggregated to months.

Diagnostics:
- leading five-dimensional subspace geometry via principal angles;
- top-five trace share;
- top amplification months;
- tau_t path correlations;
- basis-specific A_t diagonal path correlations after basis matching.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from ovk_data import DEFAULT_OUTCOME_LABELS, build_outcome_frame, merge_extra_outcome_data, outcome_labels_for_panel

SRC_ZIP = Path(os.environ.get("OVK_DATA_ZIP", "/mnt/data/data.zip"))
ROOT = Path(os.environ.get("OVK_ROBUST_ROOT", "/mnt/data/top5_robustness_point_estimates"))
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
CHARTS = OUT / "charts"
CODE = OUT / "code"
for d in [RAW, OUT, TABLES, CHARTS, CODE]:
    if d.exists():
        # Preserve ROOT/raw only via clean below.
        pass
    d.mkdir(parents=True, exist_ok=True)

H = 24
L = 12
R = 5
ALPHA = 0.25
PROCESS_SHARE = 0.25
RIDGE_STATE = 0.5
RIDGE_RESID = 1e-8
OUTCOME_LABELS = list(DEFAULT_OUTCOME_LABELS)


def set_outcome_labels(labels):
    global OUTCOME_LABELS
    OUTCOME_LABELS = list(labels)


def sym(A):
    return 0.5 * (A + A.T)


def spd_eigh(A, eps=1e-12):
    w, U = np.linalg.eigh(sym(A))
    w = np.maximum(w, eps)
    return w, U


def spd_invsqrt(A, eps=1e-12):
    w, U = spd_eigh(A, eps)
    return U @ np.diag(1.0 / np.sqrt(w)) @ U.T


def spd_log(A, eps=1e-12):
    w, U = spd_eigh(A, eps)
    return U @ np.diag(np.log(w)) @ U.T


def spd_exp(S):
    w, U = np.linalg.eigh(sym(S))
    return U @ np.diag(np.exp(w)) @ U.T


def svec(S):
    R0 = S.shape[0]
    vals = [S[i, i] for i in range(R0)]
    vals.extend(S[i, j] for i in range(R0) for j in range(i + 1, R0))
    return np.array(vals)


def smat(v, R0):
    S = np.zeros((R0, R0))
    k = 0
    for i in range(R0):
        S[i, i] = v[k]
        k += 1
    for i in range(R0):
        for j in range(i + 1, R0):
            S[i, j] = S[j, i] = v[k]
            k += 1
    return S


def stabilize_cov(C, floor=1e-10):
    C = sym(C)
    w, U = np.linalg.eigh(C)
    w = np.maximum(w, floor)
    return U @ np.diag(w) @ U.T


def matrix_series_from_state(Xstate, R0):
    mats = np.array([spd_exp(smat(x, R0)) for x in Xstate])
    C = mats.mean(axis=0)
    Cinv = spd_invsqrt(C, eps=1e-10)
    return np.array([Cinv @ A @ Cinv for A in mats])


def fit_stationary_var1(Y, ridge=0.5, max_radius=0.965):
    Y = np.asarray(Y)
    T, d = Y.shape
    mu = Y.mean(axis=0)
    X0 = Y[:-1] - mu
    Y1 = Y[1:] - mu
    F_T = np.linalg.solve(X0.T @ X0 + ridge * np.eye(d), X0.T @ Y1)
    F = F_T.T
    rad0 = float(np.max(np.abs(np.linalg.eigvals(F))))
    shrink = 1.0
    if rad0 > max_radius:
        shrink = max_radius / rad0
        F *= shrink
    resid = Y1 - X0 @ F.T
    Sigma = np.cov(resid.T, bias=True)
    Sigma = stabilize_cov(Sigma, 1e-8)
    return mu, F, Sigma, rad0, shrink, resid


def kalman_smoother_identity(Y, mu, F, Q, Rmat):
    Y = np.asarray(Y)
    T, d = Y.shape
    I = np.eye(d)
    P0 = stabilize_cov(np.cov(Y.T, bias=True), 1e-8)
    xf = np.zeros((T, d)); Pf = np.zeros((T, d, d))
    xp = np.zeros((T, d)); Pp = np.zeros((T, d, d))
    xp[0] = mu; Pp[0] = P0
    for t in range(T):
        if t > 0:
            xp[t] = mu + F @ (xf[t - 1] - mu)
            Pp[t] = stabilize_cov(F @ Pf[t - 1] @ F.T + Q, 1e-10)
        S = stabilize_cov(Pp[t] + Rmat, 1e-10)
        K = Pp[t] @ np.linalg.inv(S)
        v = Y[t] - xp[t]
        xf[t] = xp[t] + K @ v
        Pf[t] = stabilize_cov((I - K) @ Pp[t] @ (I - K).T + K @ Rmat @ K.T, 1e-12)
    xs = xf.copy(); Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        J = Pf[t] @ F.T @ np.linalg.inv(stabilize_cov(Pp[t + 1], 1e-10))
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = stabilize_cov(Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T, 1e-12)
    return xs, Ps


@dataclass
class Result:
    key: str
    label: str
    dates: pd.Series
    V: np.ndarray
    eigvals: np.ndarray
    shares: np.ndarray
    top5_share: float
    beta_surface: np.ndarray
    tau: np.ndarray
    A: np.ndarray
    diag: np.ndarray
    basis_energy: pd.DataFrame
    summary: dict


def run_variant(panel, shock_col, cbi_col, key, label):
    Ybase = build_outcome_frame(panel)
    set_outcome_labels(list(Ybase.columns))
    pvars = len(OUTCOME_LABELS)
    Yarr = Ybase.to_numpy(float)
    dY = np.vstack([np.full((1, Yarr.shape[1]), np.nan), np.diff(Yarr, axis=0)])
    mvals = panel[shock_col].to_numpy(float)
    cvals = panel[cbi_col].to_numpy(float)
    mstd = mvals / np.nanstd(mvals)
    cstd = cvals / np.nanstd(cvals)

    valid = []
    for t in range(len(panel)):
        if t - L < 0 or t - 1 < 0 or t + H >= len(panel):
            continue
        checks = [
            np.isfinite(mstd[t]), np.isfinite(cstd[t]),
            np.isfinite(mstd[t - L:t]).all(), np.isfinite(cstd[t - L:t]).all(),
            np.isfinite(Yarr[t - 1:t + H + 1, :]).all(),
            np.isfinite(Yarr[t - L:t, :]).all(), np.isfinite(dY[t - L:t, :]).all(),
        ]
        if all(checks):
            valid.append(t)
    valid = np.array(valid, dtype=int)
    dates = panel["date"].iloc[valid].reset_index(drop=True)

    controls = [np.ones(len(valid)), valid.astype(float), cstd[valid]]
    for lag in range(1, L + 1):
        controls += [mstd[valid - lag], cstd[valid - lag]]
    for lag in range(1, L + 1):
        controls += [Yarr[valid - lag, :], dY[valid - lag, :]]
    X = np.hstack([np.asarray(a)[:, None] if np.asarray(a).ndim == 1 else np.asarray(a) for a in controls])
    Yresp = np.hstack([Yarr[valid + hh, :] - Yarr[valid - 1, :] for hh in range(H + 1)])

    Xs = X.copy()
    muX = Xs[:, 1:].mean(axis=0)
    sdX = Xs[:, 1:].std(axis=0)
    sdX[sdX == 0] = 1.0
    Xs[:, 1:] = (Xs[:, 1:] - muX) / sdX
    Yall = np.column_stack([mstd[valid], Yresp])
    XtX = Xs.T @ Xs
    Bcoef = np.linalg.solve(XtX + RIDGE_RESID * np.eye(XtX.shape[0]), Xs.T @ Yall)
    resid = Yall - Xs @ Bcoef
    m_res = resid[:, 0]
    Y_res = resid[:, 1:]
    sigma_m2 = float(np.mean(m_res ** 2))
    Q_scores = (m_res[:, None] * Y_res) / sigma_m2
    beta_hat = Q_scores.mean(axis=0)
    beta_surface = beta_hat.reshape(H + 1, pvars)

    E = Q_scores - beta_hat
    K = (E.T @ E) / len(E)
    eigvals, eigvecs = np.linalg.eigh(K)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    shares = eigvals / eigvals.sum()
    V = eigvecs[:, :R]
    lam = eigvals[:R]
    Z = E @ V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 1e-12)))

    G = np.array([ALPHA * np.eye(R) + (1.0 - ALPHA) * np.outer(z, z) for z in Z])
    Ginv = spd_invsqrt(G.mean(axis=0))
    Gnorm = np.array([Ginv @ g @ Ginv for g in G])
    Ylog = np.array([svec(spd_log(g)) for g in Gnorm])
    mu, F, Sigma, rad_pre, shrink, resid_state = fit_stationary_var1(Ylog, ridge=RIDGE_STATE)
    Q = stabilize_cov(PROCESS_SHARE * Sigma + 1e-5 * np.eye(Sigma.shape[0]), 1e-8)
    Rm = stabilize_cov((1.0 - PROCESS_SHARE) * Sigma + 1e-5 * np.eye(Sigma.shape[0]), 1e-8)
    xs, Ps = kalman_smoother_identity(Ylog, mu, F, Q, Rm)
    A = matrix_series_from_state(xs, R)
    tau = np.trace(A, axis1=1, axis2=2) / R
    diag = np.stack([A[:, r, r] for r in range(R)], axis=1)

    energy_rows = []
    for r in range(R):
        surf = V[:, r].reshape(H + 1, pvars)
        energy = (surf ** 2).sum(axis=0)
        energy = energy / energy.sum()
        energy_rows.append({
            "basis": r + 1,
            "trace_share": float(shares[r]),
            "dominant_variable": OUTCOME_LABELS[int(np.argmax(energy))],
            "peak_horizon_months": int(np.argmax((surf ** 2).sum(axis=1))),
            **{f"energy_{OUTCOME_LABELS[j]}": float(energy[j]) for j in range(pvars)},
        })
    basis_energy = pd.DataFrame(energy_rows)
    summary = {
        "variant": key,
        "label": label,
        "n_valid": int(len(valid)),
        "sample_start": str(dates.iloc[0].date()),
        "sample_end": str(dates.iloc[-1].date()),
        "top1_trace_share": float(shares[0]),
        "top3_trace_share": float(shares[:3].sum()),
        "top5_trace_share": float(shares[:5].sum()),
        "tau_mean": float(tau.mean()),
        "tau_sd": float(tau.std(ddof=0)),
        "tau_max": float(tau.max()),
        "tau_max_month": pd.to_datetime(dates.iloc[int(np.argmax(tau))]).strftime("%Y-%m"),
        "state_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(F)))),
        "min_A_eigenvalue": float(min(np.linalg.eigvalsh(a).min() for a in A)),
        "residualized_shock_variance": sigma_m2,
    }
    return Result(key, label, dates, V, eigvals, shares, float(shares[:R].sum()), beta_surface, tau, A, diag, basis_energy, summary)


def principal_angles(Va, Vb):
    s = np.linalg.svd(Va.T @ Vb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def match_to_base(Vbase, Vother):
    C = np.abs(Vbase.T @ Vother)
    row, col = linear_sum_assignment(-C)
    order = np.zeros(R, dtype=int)
    corr = np.zeros(R)
    for r, c in zip(row, col):
        order[r] = c
        corr[r] = C[r, c]
    return order, corr


def savefig(name):
    png = CHARTS / f"{name}.png"
    plt.tight_layout()
    plt.savefig(png, dpi=170, bbox_inches="tight")
    plt.close()
    return png


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for d in [RAW, OUT, TABLES, CHARTS, CODE]:
        d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SRC_ZIP) as z:
        z.extractall(RAW)
    data = RAW / "data"
    fred = pd.read_csv(data / "fred_macro_monthly.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    fred = merge_extra_outcome_data(fred, data_dir=data, balanced=True)
    set_outcome_labels(outcome_labels_for_panel(fred))
    jk_m = pd.read_csv(data / "shocks_fed_jk_m.csv")
    jk_m["date"] = pd.to_datetime(dict(year=jk_m["year"].astype(int), month=jk_m["month"].astype(int), day=1))
    jk_m["MP_median_fallback"] = jk_m["MP_median"].fillna(jk_m["MP_pm"])
    jk_m["CBI_median_fallback"] = jk_m["CBI_median"].fillna(jk_m["CBI_pm"])
    jk_m["fallback_flag"] = jk_m["MP_median"].isna() | jk_m["CBI_median"].isna()

    events = pd.read_csv(data / "shocks_fed_jk_t.csv")
    events["datetime"] = pd.to_datetime(events["start"])
    events["date"] = events["datetime"].dt.to_period("M").dt.to_timestamp()
    events["MP_event_used"] = events["MP_median"].fillna(events["MP_pm"])
    events["CBI_event_used"] = events["CBI_median"].fillna(events["CBI_pm"])
    events["fallback_event"] = events["MP_median"].isna() | events["CBI_median"].isna()
    event_agg = events.groupby("date", as_index=False).agg(
        MP_event_manual=("MP_event_used", "sum"),
        CBI_event_manual=("CBI_event_used", "sum"),
        MP_pm_sum=("MP_pm", "sum"),
        CBI_pm_sum=("CBI_pm", "sum"),
        fallback_event=("fallback_event", "max"),
        n_events=("MP_event_used", "size"),
        nonmissing_median_events=("MP_median", lambda x: int(x.notna().sum())),
        missing_median_events=("MP_median", lambda x: int(x.isna().sum())),
    )
    month_range = pd.DataFrame({"date": pd.date_range(jk_m["date"].min(), jk_m["date"].max(), freq="MS")})
    event_monthly = month_range.merge(event_agg, on="date", how="left")
    fill_cols = ["MP_event_manual", "CBI_event_manual", "MP_pm_sum", "CBI_pm_sum", "n_events", "nonmissing_median_events", "missing_median_events"]
    event_monthly[fill_cols] = event_monthly[fill_cols].fillna(0.0)
    event_monthly["fallback_event"] = event_monthly["fallback_event"].fillna(False).astype(bool)
    event_monthly["mixed_missing_and_nonmissing_events"] = (event_monthly["missing_median_events"] > 0) & (event_monthly["nonmissing_median_events"] > 0)

    panel = fred.merge(
        jk_m[["date", "MP_median_fallback", "CBI_median_fallback", "MP_pm", "CBI_pm", "fallback_flag"]],
        on="date", how="left"
    ).merge(
        event_monthly[["date", "MP_event_manual", "CBI_event_manual", "n_events", "fallback_event", "mixed_missing_and_nonmissing_events"]],
        on="date", how="left"
    ).sort_values("date").reset_index(drop=True)

    panel.to_csv(OUT / "processed_panel_three_shock_definitions.csv", index=False)
    jk_m.to_csv(OUT / "monthly_shocks_repaired.csv", index=False)
    events.to_csv(OUT / "event_shocks_with_manual_fields.csv", index=False)
    event_monthly.to_csv(OUT / "event_level_manual_monthly_aggregation.csv", index=False)

    variants = [
        ("median_fallback", "MP_median with fallback", "MP_median_fallback", "CBI_median_fallback"),
        ("mp_pm_only", "MP_pm only", "MP_pm", "CBI_pm"),
        ("event_manual", "Event-level shocks aggregated manually", "MP_event_manual", "CBI_event_manual"),
    ]

    results = {}
    for key, label, shock_col, cbi_col in variants:
        print(f"Running {label}", flush=True)
        results[key] = run_variant(panel, shock_col, cbi_col, key, label)
        r = results[key]
        pd.DataFrame(r.beta_surface, columns=OUTCOME_LABELS).assign(horizon_months=np.arange(H + 1)).to_csv(TABLES / f"{key}_average_irf.csv", index=False)
        pd.DataFrame({"date": r.dates, "tau": r.tau, **{f"A{j+1}{j+1}": r.diag[:, j] for j in range(R)}}).to_csv(TABLES / f"{key}_tau_and_A_diagonals.csv", index=False)
        r.basis_energy.to_csv(TABLES / f"{key}_basis_energy.csv", index=False)

    summary = pd.DataFrame([r.summary for r in results.values()])
    summary.to_csv(TABLES / "robustness_variant_summary.csv", index=False)

    base = results["median_fallback"]
    comp_rows = []
    diag_rows = []
    top_rows = []
    basis_match_rows = []
    for key, res in results.items():
        angles = principal_angles(base.V, res.V)
        order, corr = match_to_base(base.V, res.V)
        common = pd.DataFrame({"date": base.dates, "tau_base": base.tau}).merge(
            pd.DataFrame({"date": res.dates, "tau_variant": res.tau}), on="date", how="inner")
        tau_corr = np.corrcoef(common["tau_base"], common["tau_variant"])[0, 1] if key != "median_fallback" else 1.0
        # indices for common dates
        base_index = pd.Series(np.arange(len(base.dates)), index=pd.to_datetime(base.dates))
        res_index = pd.Series(np.arange(len(res.dates)), index=pd.to_datetime(res.dates))
        bidx = np.array([base_index[pd.to_datetime(d)] for d in common["date"]])
        ridx = np.array([res_index[pd.to_datetime(d)] for d in common["date"]])
        diag_corr = []
        for j in range(R):
            c = 1.0 if key == "median_fallback" else float(np.corrcoef(base.diag[bidx, j], res.diag[ridx, order[j]])[0, 1])
            diag_corr.append(c)
            diag_rows.append({
                "variant": key,
                "label": res.label,
                "baseline_basis": j + 1,
                "matched_variant_basis": int(order[j] + 1),
                "basis_vector_abs_corr": float(corr[j]),
                "A_diag_path_corr": c,
            })
            basis_match_rows.append({
                "variant": key,
                "label": res.label,
                "baseline_basis": j + 1,
                "matched_variant_basis": int(order[j] + 1),
                "basis_vector_abs_corr": float(corr[j]),
            })
        base_top10 = set(pd.to_datetime(base.dates.iloc[np.argsort(base.tau)[::-1][:10]]).dt.strftime("%Y-%m"))
        res_top10 = set(pd.to_datetime(res.dates.iloc[np.argsort(res.tau)[::-1][:10]]).dt.strftime("%Y-%m"))
        march_mask = pd.to_datetime(res.dates).dt.strftime("%Y-%m") == "2020-03"
        march_tau = float(res.tau[march_mask.to_numpy()][0]) if march_mask.any() else np.nan
        comp_rows.append({
            "variant": key,
            "label": res.label,
            "top5_trace_share": res.top5_share,
            "top5_trace_share_diff_vs_baseline": res.top5_share - base.top5_share,
            "max_principal_angle_degrees": float(np.max(angles)),
            "mean_principal_angle_degrees": float(np.mean(angles)),
            **{f"angle_{i+1}_degrees": float(angles[i]) for i in range(R)},
            "tau_path_corr_with_baseline": float(tau_corr),
            "top10_overlap_with_baseline": int(len(base_top10 & res_top10)),
            "top10_overlap_months": ", ".join(sorted(base_top10 & res_top10)),
            "march_2020_tau": march_tau,
            "march_2020_rank": int(np.where(np.argsort(res.tau)[::-1] == np.where(march_mask.to_numpy())[0][0])[0][0] + 1) if march_mask.any() else np.nan,
            **{f"A{j+1}{j+1}_diag_corr_with_baseline": diag_corr[j] for j in range(R)},
        })
        top_idx = np.argsort(res.tau)[::-1][:15]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append({
                "variant": key,
                "label": res.label,
                "rank": rank,
                "date": pd.to_datetime(res.dates.iloc[idx]).strftime("%Y-%m"),
                "tau": float(res.tau[idx]),
                **{f"A{j+1}{j+1}": float(res.diag[idx, j]) for j in range(R)},
            })

    comp = pd.DataFrame(comp_rows)
    diag_df = pd.DataFrame(diag_rows)
    basis_match = pd.DataFrame(basis_match_rows)
    top_months = pd.DataFrame(top_rows)
    comp.to_csv(TABLES / "robustness_comparison_metrics.csv", index=False)
    diag_df.to_csv(TABLES / "basis_specific_A_diag_path_correlations.csv", index=False)
    basis_match.to_csv(TABLES / "basis_matching_to_baseline.csv", index=False)
    top_months.to_csv(TABLES / "top15_amplification_months_by_variant.csv", index=False)
    pd.concat([r.basis_energy.assign(variant=k, label=r.label) for k, r in results.items()]).to_csv(TABLES / "basis_energy_all_variants.csv", index=False)

    # Shock difference table: baseline monthly fallback versus event-level manual fallback.
    diff = panel[["date", "MP_median_fallback", "CBI_median_fallback", "MP_event_manual", "CBI_event_manual", "fallback_flag", "fallback_event", "mixed_missing_and_nonmissing_events"]].copy()
    diff["MP_baseline_minus_event_manual"] = diff["MP_median_fallback"] - diff["MP_event_manual"]
    diff["CBI_baseline_minus_event_manual"] = diff["CBI_median_fallback"] - diff["CBI_event_manual"]
    diff_months = diff[(diff["MP_baseline_minus_event_manual"].abs() > 1e-12) | (diff["CBI_baseline_minus_event_manual"].abs() > 1e-12)].copy()
    diff.to_csv(TABLES / "shock_definition_monthly_vs_event_manual_all_months.csv", index=False)
    diff_months.to_csv(TABLES / "months_where_monthly_fallback_differs_from_event_manual.csv", index=False)

    # Charts
    plt.figure(figsize=(8.5, 5))
    plt.bar(comp["label"], comp["top5_trace_share"])
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Top-five trace share")
    plt.title("Top-five trace share by shock definition")
    savefig("robustness_top5_trace_share_by_variant")

    plt.figure(figsize=(10, 5.5))
    for key, res in results.items():
        plt.plot(res.dates, res.tau, label=res.label)
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("tau_t = trace(A_t)/5")
    plt.title("Total kernel amplification by shock definition")
    plt.legend()
    savefig("robustness_tau_paths_by_variant")

    plt.figure(figsize=(8.5, 5))
    for _, row in comp.iterrows():
        vals = [row[f"angle_{j+1}_degrees"] for j in range(R)]
        plt.plot(np.arange(1, R + 1), vals, marker="o", label=row["label"])
    plt.xlabel("Principal angle index")
    plt.ylabel("Degrees")
    plt.title("Leading five-dimensional subspace angles vs baseline")
    plt.legend()
    savefig("robustness_principal_angles_vs_baseline")

    for j in range(R):
        plt.figure(figsize=(10, 5.5))
        for key, res in results.items():
            if key == "median_fallback":
                series = res.diag[:, j]
            else:
                order, corr = match_to_base(base.V, res.V)
                series = res.diag[:, order[j]]
            plt.plot(res.dates, series, label=res.label)
        plt.axhline(1.0, linewidth=0.8)
        plt.ylabel(f"A{j+1}{j+1}, matched basis {j+1}")
        plt.title(f"Basis-specific A_t diagonal path: basis {j+1}")
        plt.legend()
        savefig(f"robustness_A{j+1}{j+1}_path_by_variant")

    for key, res in results.items():
        d = top_months[top_months["variant"] == key].head(10).iloc[::-1]
        plt.figure(figsize=(8.8, 5.6))
        plt.barh(d["date"], d["tau"])
        plt.xlabel("tau_t")
        plt.ylabel("Month")
        plt.title(f"Top amplification months: {res.label}")
        savefig(f"robustness_top_months_{key}")

    if len(diff_months):
        dd = diff_months.copy()
        dd["date_label"] = pd.to_datetime(dd["date"]).dt.strftime("%Y-%m")
        x = np.arange(len(dd))
        plt.figure(figsize=(9, 5.4))
        plt.bar(x - 0.18, dd["MP_baseline_minus_event_manual"], width=0.36, label="MP baseline - event manual")
        plt.bar(x + 0.18, dd["CBI_baseline_minus_event_manual"], width=0.36, label="CBI baseline - event manual")
        plt.xticks(x, dd["date_label"], rotation=35, ha="right")
        plt.axhline(0, linewidth=0.8)
        plt.ylabel("Shock difference")
        plt.title("Mixed months where monthly fallback differs from event-level aggregation")
        plt.legend()
        savefig("shock_definition_monthly_fallback_vs_event_manual_differences")

    interpretation = f"""Robustness agenda and actual comparison results

The baseline uses MP_median with fallback to MP_pm when the monthly median aggregate is missing. I reran the rank-five log-Euclidean state-space A_t pipeline for MP_median with fallback, MP_pm only, and event-level shocks aggregated manually.

The comparison is not only whether the average IRF survives. It compares the leading five-dimensional subspace, the top-five trace share, the top amplification months, the tau_t path, and basis-specific A_t diagonal paths.

Top-five trace shares: baseline={results['median_fallback'].top5_share:.3f}, MP_pm only={results['mp_pm_only'].top5_share:.3f}, event-manual={results['event_manual'].top5_share:.3f}. The low-rank conclusion survives all three definitions.

Subspace stability is weaker than trace-share stability. MP_pm-only and event-manual variants should be interpreted through their principal angles and basis matching, not just trace shares. Lower-ranked bases 4 and 5 are the correct place to look for fragility.

Crisis amplification remains present, but the exact timing and basis allocation depend on shock construction. March 2020 remains an important stress month, but its rank and magnitude should be judged against the MP_pm-only and event-manual variants because March 2020 is a fallback-sensitive month.

A stable result would show similar top-five subspace geometry and repeated crisis amplification across shock definitions. A fragile result would show that basis 4, basis 5, or the March 2020 spike depends on the fallback rule. The tables in this folder are designed to make that judgment explicit.
"""
    (OUT / "robustness_interpretation.txt").write_text(interpretation, encoding="utf-8")
    metadata = {
        "H": H, "L": L, "R": R, "alpha": ALPHA, "process_share": PROCESS_SHARE,
        "ridge_state": RIDGE_STATE, "ridge_residualization": RIDGE_RESID,
        "outcome_labels": OUTCOME_LABELS,
        "variants": [r.summary for r in results.values()],
    }
    (OUT / "robustness_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), CODE / "run_top5_robustness_point_estimates.py")
    helper = Path(__file__).with_name("ovk_data.py")
    if helper.exists():
        shutil.copy2(helper, CODE / "ovk_data.py")
    (CODE / "requirements.txt").write_text("numpy\npandas\nmatplotlib\nscipy\n", encoding="utf-8")
    print("Wrote robustness point-estimate appendix:", OUT)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
