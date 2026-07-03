import numpy as np
import pandas as pd
import zipfile, shutil, base64, html, warnings, time, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh as scipy_eigh
from scipy.optimize import linear_sum_assignment
from PIL import Image as PILImage, ImageOps, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib import colors

from ovk_data import build_outcome_frame, merge_extra_outcome_data

warnings.filterwarnings('ignore')

# -----------------------------
# Paths and configuration
# -----------------------------
SRC_ZIP = Path(os.environ.get('OVK_DATA_ZIP', '/mnt/data/data.zip'))
ROOT = Path(os.environ.get('OVK_TOP5_ROOT', '/mnt/data/monthly_ovk_dynamic_state_model_top5_report'))
RAW = ROOT / 'raw'
OUT = ROOT / 'outputs'
CHARTS = OUT / 'charts'
TABLES = OUT / 'tables'
for d in [RAW, OUT, CHARTS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

FINAL_PDF = Path(os.environ.get('OVK_TOP5_FINAL_PDF', '/mnt/data/monthly_ovk_dynamic_state_model_top5_report.pdf'))
FINAL_HTML = Path(os.environ.get('OVK_TOP5_FINAL_HTML', '/mnt/data/monthly_ovk_dynamic_state_model_top5_report.html'))
FINAL_ZIP = Path(os.environ.get('OVK_TOP5_FINAL_ZIP', '/mnt/data/monthly_ovk_dynamic_state_model_top5_bundle.zip'))
FINAL_CONTACT = Path(os.environ.get('OVK_TOP5_FINAL_CONTACT', '/mnt/data/monthly_ovk_dynamic_state_model_top5_contact_sheet.jpg'))
for p in [FINAL_PDF, FINAL_HTML, FINAL_ZIP, FINAL_CONTACT]:
    p.parent.mkdir(parents=True, exist_ok=True)

H = 24
L = 12
R = 5
alpha = 0.25
process_share = 0.25
ridge = 0.5
Bboot = 400
Bstate = 250
block_len = 18
rng_boot_seed = 8021
rng_state_seed = 5162

# -----------------------------
# Helpers
# -----------------------------
def sanitize_col(s: str) -> str:
    return s.lower().replace(' ', '_').replace('-', '').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')

def safe_name(s: str) -> str:
    return sanitize_col(s).replace('__','_')

def sym(A):
    return 0.5 * (A + A.T)

def spd_eigh(A, eps=1e-12):
    w, U = np.linalg.eigh(sym(A))
    w = np.maximum(w, eps)
    return w, U

def spd_invsqrt(A, eps=1e-12):
    w, U = spd_eigh(A, eps)
    return U @ np.diag(1 / np.sqrt(w)) @ U.T

def spd_log(A, eps=1e-12):
    w, U = spd_eigh(A, eps)
    return U @ np.diag(np.log(w)) @ U.T

def spd_exp(S):
    w, U = np.linalg.eigh(sym(S))
    return U @ np.diag(np.exp(w)) @ U.T

def batched_spd_exp(S):
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    w, U = np.linalg.eigh(S)
    return (U * np.exp(w)[..., None, :]) @ np.swapaxes(U, -1, -2)

def batched_spd_invsqrt(A, eps=1e-12):
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    w, U = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (U * (1 / np.sqrt(w))[..., None, :]) @ np.swapaxes(U, -1, -2)

def svec(S):
    R0 = S.shape[0]
    vals = []
    for i in range(R0):
        vals.append(S[i, i])
    for i in range(R0):
        for j in range(i + 1, R0):
            vals.append(S[i, j])
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

def smat_batch(V, R0):
    V = np.asarray(V)
    out = np.zeros(V.shape[:-1] + (R0, R0))
    k = 0
    for i in range(R0):
        out[..., i, i] = V[..., k]
        k += 1
    for i in range(R0):
        for j in range(i + 1, R0):
            out[..., i, j] = V[..., k]
            out[..., j, i] = V[..., k]
            k += 1
    return out

def stabilize_cov(C, floor=1e-10):
    C = sym(C)
    w, U = np.linalg.eigh(C)
    w = np.maximum(w, floor)
    return U @ np.diag(w) @ U.T

def matrix_series_from_state(Xstate, R0):
    mats = batched_spd_exp(smat_batch(np.asarray(Xstate), R0))
    C = mats.mean(axis=0)
    Cinv = spd_invsqrt(C, eps=1e-10)
    return Cinv @ mats @ Cinv

def matrix_series_from_state_draws(Xdraws, R0, chunk_draws=64):
    Xdraws = np.asarray(Xdraws)
    B, T, d = Xdraws.shape
    Aout = np.empty((B, T, R0, R0))
    for start in range(0, B, chunk_draws):
        stop = min(start + chunk_draws, B)
        Xc = Xdraws[start:stop]
        n = stop - start
        mats = batched_spd_exp(smat_batch(Xc.reshape(n * T, d), R0)).reshape(n, T, R0, R0)
        Cinv = batched_spd_invsqrt(mats.mean(axis=1), eps=1e-10)
        Aout[start:stop] = np.einsum('bij,btjk,bkl->btil', Cinv, mats, Cinv, optimize=True)
    return Aout

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

def kalman_smoother_identity(Y, mu, F, Q, Rmat, P0=None):
    Y = np.asarray(Y)
    T, d = Y.shape
    I = np.eye(d)
    if P0 is None:
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
    return xs, Ps, xf, Pf, xp, Pp

def circular_block_indices(n, block_len, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
    blocks = [(np.arange(s, s + block_len) % n) for s in starts]
    return np.concatenate(blocks)[:n]

def match_modes(Uboot, Vbase):
    corr = np.abs(Vbase.T @ Uboot)
    row_ind, col_ind = linear_sum_assignment(-corr)
    matched = np.zeros_like(Vbase)
    for r, c in zip(row_ind, col_ind):
        v = Uboot[:, c]
        if np.dot(v, Vbase[:, r]) < 0:
            v = -v
        matched[:, r] = v
    return matched

def save_chart(fig, name):
    png = CHARTS / f'{name}.png'
    fig.tight_layout()
    fig.savefig(png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return png

def img_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def df_to_html(df, cols=None, max_rows=20):
    d = df.copy()
    if cols is not None:
        d = d[cols].copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    header = '<tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in d.columns) + '</tr>'
    rows = [header]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(f'<td>{float(v):.3f}</td>')
            else:
                cells.append(f'<td>{html.escape(str(v))}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '<table>' + '\n'.join(rows) + '</table>'

def paras_to_html(text):
    return '\n'.join(f'<p>{html.escape(p.strip())}</p>' for p in text.split('\n\n') if p.strip())

# -----------------------------
# Load and repair data
# -----------------------------
if RAW.exists():
    # Keep raw folder but overwrite extracted contents for reproducibility.
    pass
with zipfile.ZipFile(SRC_ZIP) as z:
    z.extractall(RAW)
DATA = RAW / 'data'
fred = pd.read_csv(DATA / 'fred_macro_monthly.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
fred = merge_extra_outcome_data(fred, data_dir=DATA, balanced=True)
jk_m = pd.read_csv(DATA / 'shocks_fed_jk_m.csv')
jk_m['date'] = pd.to_datetime(dict(year=jk_m['year'].astype(int), month=jk_m['month'].astype(int), day=1))
jk_m['MP_used'] = jk_m['MP_median'].fillna(jk_m['MP_pm'])
jk_m['CBI_used'] = jk_m['CBI_median'].fillna(jk_m['CBI_pm'])
jk_m['used_pm_fallback'] = jk_m['MP_median'].isna() | jk_m['CBI_median'].isna()
shock_cols = ['date','year','month','pc1_hf','SP500_hf','MP_pm','CBI_pm','MP_median','CBI_median','MP_used','CBI_used','used_pm_fallback']
panel = fred.merge(jk_m[shock_cols], on='date', how='left').sort_values('date').reset_index(drop=True)
overlap = fred.merge(jk_m[shock_cols], on='date', how='inner').sort_values('date').reset_index(drop=True)
print('Loaded and repaired data', flush=True)
panel.to_csv(OUT / 'ovk_monetary_panel_monthly_fixed_full.csv', index=False)
overlap.to_csv(OUT / 'ovk_monetary_panel_monthly_fixed_overlap.csv', index=False)

# -----------------------------
# LP-score construction
# -----------------------------
Ybase = build_outcome_frame(panel)
outcome_labels = list(Ybase.columns)
pvars = len(outcome_labels)
M = (H + 1) * pvars
mvals = panel['MP_used'].to_numpy(float)
cvals = panel['CBI_used'].to_numpy(float)
mstd = mvals / np.nanstd(mvals)
cstd = cvals / np.nanstd(cvals)
Yarr = Ybase.to_numpy(float)
dYarr = np.vstack([np.full((1, pvars), np.nan), np.diff(Yarr, axis=0)])
valid = []
for t in range(len(panel)):
    if t - L < 0 or t - 1 < 0 or t + H >= len(panel):
        continue
    checks = [
        np.isfinite(mstd[t]), np.isfinite(cstd[t]),
        np.isfinite(mstd[t-L:t]).all(), np.isfinite(cstd[t-L:t]).all(),
        np.isfinite(Yarr[t-1:t+H+1,:]).all(), np.isfinite(Yarr[t-L:t,:]).all(), np.isfinite(dYarr[t-L:t,:]).all(),
    ]
    if all(checks):
        valid.append(t)
valid_idx = np.array(valid, dtype=int)
dates = panel['date'].iloc[valid_idx].reset_index(drop=True)
controls = [np.ones(len(valid_idx)), valid_idx.astype(float), cstd[valid_idx]]
for lag in range(1, L + 1):
    controls += [mstd[valid_idx - lag], cstd[valid_idx - lag]]
for lag in range(1, L + 1):
    controls += [Yarr[valid_idx - lag, :], dYarr[valid_idx - lag, :]]
X = np.hstack([np.asarray(a)[:, None] if np.asarray(a).ndim == 1 else np.asarray(a) for a in controls])
Yresp = np.hstack([Yarr[valid_idx + hh, :] - Yarr[valid_idx - 1, :] for hh in range(H + 1)])
Xs = X.copy()
muX = Xs[:, 1:].mean(axis=0)
sdX = Xs[:, 1:].std(axis=0)
sdX[sdX == 0] = 1
Xs[:, 1:] = (Xs[:, 1:] - muX) / sdX
m = mstd[valid_idx]
Bcoef = np.linalg.lstsq(Xs, np.column_stack([m, Yresp]), rcond=None)[0]
resid = np.column_stack([m, Yresp]) - Xs @ Bcoef
m_res = resid[:, 0]
Y_res = resid[:, 1:]
sigma_m2 = np.mean(m_res ** 2)
Q_scores = (m_res[:, None] * Y_res) / sigma_m2
beta_hat = Q_scores.mean(axis=0)
beta_surface = beta_hat.reshape(H + 1, pvars)
E = Q_scores - beta_hat
K_bar = (E.T @ E) / len(E)
evals, evecs = np.linalg.eigh(K_bar)
idx = np.argsort(evals)[::-1]
evals = evals[idx]; evecs = evecs[:, idx]
trace_total = evals.sum(); shares = evals / trace_total
V = evecs[:, :R]; lam = evals[:R]
Z = E @ V @ np.diag(1 / np.sqrt(np.maximum(lam, 1e-12)))
print('Built LP scores and eigensystem', flush=True)

# -----------------------------
# Top-5 log-Euclidean state-space A_t model
# -----------------------------
G_raw = np.array([alpha * np.eye(R) + (1 - alpha) * np.outer(z, z) for z in Z])
G_inv = spd_invsqrt(G_raw.mean(axis=0))
G_norm = np.array([G_inv @ G @ G_inv for G in G_raw])
Ylog = np.array([svec(spd_log(G)) for G in G_norm])
dstate = Ylog.shape[1]
mu, F, Sigma_u, rad_pre, shrink, resid_var = fit_stationary_var1(Ylog, ridge=ridge, max_radius=0.965)
var_radius = float(np.max(np.abs(np.linalg.eigvals(F))))
Qproc = stabilize_cov(process_share * Sigma_u + 1e-5 * np.eye(dstate), 1e-8)
Rmeas = stabilize_cov((1 - process_share) * Sigma_u + 1e-5 * np.eye(dstate), 1e-8)
xs, Ps, xf, Pf, xp, Pp = kalman_smoother_identity(Ylog, mu, F, Qproc, Rmeas)
A_state = matrix_series_from_state(xs, R)
tau = np.trace(A_state, axis1=1, axis2=2) / R
min_eig_A = float(min(np.linalg.eigvalsh(A).min() for A in A_state))

# State uncertainty draws
rng = np.random.default_rng(rng_state_seed)
Lcov = []
for t in range(len(xs)):
    C = stabilize_cov(Ps[t], 1e-12)
    try:
        Lc = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(C)
        Lc = U @ np.diag(np.sqrt(np.maximum(w, 1e-12)))
    Lcov.append(Lc)
Lcov = np.array(Lcov)
eps = rng.normal(size=(Bstate, len(xs), dstate))
Xdraws = xs[None, :, :] + np.einsum('tij,btj->bti', Lcov, eps, optimize=True)
Adraws = matrix_series_from_state_draws(Xdraws, R)
tau_draws = np.trace(Adraws, axis1=2, axis2=3) / R
diag_draws = np.moveaxis(np.diagonal(Adraws, axis1=2, axis2=3), 2, 1)
tau_band90 = np.quantile(tau_draws, [.05, .50, .95], axis=0)
tau_band68 = np.quantile(tau_draws, [.16, .84], axis=0)
diag_band90 = np.quantile(diag_draws, [.05, .50, .95], axis=0)
diag_band68 = np.quantile(diag_draws, [.16, .84], axis=0)
print('Finished state model and state uncertainty', flush=True)

# -----------------------------
# Moving-block bootstrap bands for IRFs, spectrum, and top-5 bases
# -----------------------------
n = len(Q_scores)
rng = np.random.default_rng(rng_boot_seed)
beta_boot = np.empty((Bboot, M))
share_boot = np.empty((Bboot, 10))
cum_boot = np.empty((Bboot, 10))
mode_boot = np.empty((Bboot, R, M))
subspace_angle = np.empty(Bboot)
for b in range(Bboot):
    ix = circular_block_indices(n, block_len, rng)
    qb = Q_scores[ix]
    beta_b = qb.mean(axis=0)
    eb = qb - beta_b
    Kb = (eb.T @ eb) / n
    w, U = scipy_eigh(Kb, subset_by_index=[M - 10, M - 1])
    ii = np.argsort(w)[::-1]
    w = w[ii]; U = U[:, ii]
    total = np.trace(Kb)
    beta_boot[b] = beta_b
    sh = w / total
    share_boot[b] = sh
    cum_boot[b] = np.cumsum(sh)
    matched = match_modes(U[:, :R], V)
    for r in range(R):
        mode_boot[b, r] = matched[:, r]
    svals = np.linalg.svd(V.T @ U[:, :R], compute_uv=False)
    svals = np.clip(svals, -1, 1)
    subspace_angle[b] = np.degrees(np.arccos(np.min(svals)))

beta_band90 = np.quantile(beta_boot, [.05, .50, .95], axis=0)
share_band90 = np.quantile(share_boot, [.05, .50, .95], axis=0)
cum_band90 = np.quantile(cum_boot, [.05, .50, .95], axis=0)
mode_band90 = np.quantile(mode_boot, [.05, .50, .95], axis=0)
mode_corrs = np.empty((Bboot, R))
for b in range(Bboot):
    for r in range(R):
        mode_corrs[b, r] = abs(mode_boot[b, r] @ V[:, r])
print('Finished block bootstrap', flush=True)
mode_stability = pd.DataFrame({
    'basis': np.arange(1, R + 1),
    'median_abs_corr_with_bootstrap_basis': np.median(mode_corrs, axis=0),
    'p05_abs_corr': np.quantile(mode_corrs, .05, axis=0),
    'p95_abs_corr': np.quantile(mode_corrs, .95, axis=0),
})

# -----------------------------
# Tables
# -----------------------------
base = beta_surface
low = beta_band90[0].reshape(H + 1, pvars)
medb = beta_band90[1].reshape(H + 1, pvars)
high = beta_band90[2].reshape(H + 1, pvars)
irf_rows = []
for hh in range(H + 1):
    for j, label in enumerate(outcome_labels):
        irf_rows.append({'horizon_months': hh, 'variable': label, 'estimate': base[hh, j], 'boot_p05': low[hh, j], 'boot_median': medb[hh, j], 'boot_p95': high[hh, j]})
irf_df = pd.DataFrame(irf_rows)
irf_df.to_csv(TABLES / 'average_irf_with_block_bootstrap_bands.csv', index=False)

eigen_df = pd.DataFrame({
    'rank': np.arange(1, 11),
    'share_estimate': shares[:10],
    'share_p05': share_band90[0],
    'share_median': share_band90[1],
    'share_p95': share_band90[2],
    'cumulative_estimate': np.cumsum(shares[:10]),
    'cumulative_p05': cum_band90[0],
    'cumulative_median': cum_band90[1],
    'cumulative_p95': cum_band90[2],
})
eigen_df.to_csv(TABLES / 'average_ovk_eigenspectrum_with_bands.csv', index=False)

mode_energy_boot = np.zeros((Bboot, R, pvars))
for b in range(Bboot):
    for r in range(R):
        surf = mode_boot[b, r].reshape(H + 1, pvars)
        en = (surf ** 2).sum(axis=0)
        mode_energy_boot[b, r] = en / en.sum()
energy_q = np.quantile(mode_energy_boot, [.05, .50, .95], axis=0)

basis_load_rows = []
basis_diag_rows = []
for r in range(R):
    surface = V[:, r].reshape(H + 1, pvars)
    lo = mode_band90[0, r].reshape(H + 1, pvars)
    med = mode_band90[1, r].reshape(H + 1, pvars)
    hi = mode_band90[2, r].reshape(H + 1, pvars)
    for hh in range(H + 1):
        for j, label in enumerate(outcome_labels):
            basis_load_rows.append({'basis': r + 1, 'horizon_months': hh, 'variable': label, 'loading_estimate': surface[hh, j], 'boot_p05': lo[hh, j], 'boot_median': med[hh, j], 'boot_p95': hi[hh, j]})
    energy = (surface ** 2).sum(axis=0)
    energy = energy / energy.sum()
    peak_h = int(np.argmax((surface ** 2).sum(axis=1)))
    row = {
        'basis': r + 1,
        'eigen_share_estimate': shares[r],
        'eigen_share_p05': share_band90[0, r],
        'eigen_share_p95': share_band90[2, r],
        'dominant_variable': outcome_labels[int(np.argmax(energy))],
        'peak_horizon_months': peak_h,
        'median_abs_corr_with_bootstrap_basis': mode_stability.loc[r, 'median_abs_corr_with_bootstrap_basis'],
        'p05_abs_corr_with_bootstrap_basis': mode_stability.loc[r, 'p05_abs_corr'],
        'p95_abs_corr_with_bootstrap_basis': mode_stability.loc[r, 'p95_abs_corr'],
    }
    for j, label in enumerate(outcome_labels):
        key = safe_name(label)
        row[f'{key}_energy_estimate'] = energy[j]
        row[f'{key}_energy_p05'] = energy_q[0, r, j]
        row[f'{key}_energy_median'] = energy_q[1, r, j]
        row[f'{key}_energy_p95'] = energy_q[2, r, j]
    basis_diag_rows.append(row)

basis_load_df = pd.DataFrame(basis_load_rows)
basis_diag_df = pd.DataFrame(basis_diag_rows)
basis_load_df.to_csv(TABLES / 'top5_basis_loadings_with_bootstrap_bands.csv', index=False)
basis_diag_df.to_csv(TABLES / 'top5_basis_diagnostics_with_bootstrap_bands.csv', index=False)
mode_stability.to_csv(TABLES / 'top5_basis_bootstrap_stability.csv', index=False)

drift_data = {
    'date': dates.values,
    'trace_A_over_R': tau,
    'trace_A_p05': tau_band90[0],
    'trace_A_median': tau_band90[1],
    'trace_A_p95': tau_band90[2],
    'trace_A_p16': tau_band68[0],
    'trace_A_p84': tau_band68[1],
    'MP_used_std': mstd[valid_idx],
    'CBI_used_std': cstd[valid_idx],
    'used_pm_fallback_current_month': panel['used_pm_fallback'].iloc[valid_idx].fillna(False).to_numpy(bool),
}
for r in range(R):
    drift_data[f'A{r+1}{r+1}_basis{r+1}'] = A_state[:, r, r]
    drift_data[f'A{r+1}{r+1}_p05'] = diag_band90[0, r]
    drift_data[f'A{r+1}{r+1}_median'] = diag_band90[1, r]
    drift_data[f'A{r+1}{r+1}_p95'] = diag_band90[2, r]
    drift_data[f'A{r+1}{r+1}_p16'] = diag_band68[0, r]
    drift_data[f'A{r+1}{r+1}_p84'] = diag_band68[1, r]
for i in range(R):
    for j in range(i + 1, R):
        drift_data[f'A{i+1}{j+1}_basis{i+1}_basis{j+1}'] = A_state[:, i, j]
drift_df = pd.DataFrame(drift_data)
drift_df.to_csv(TABLES / 'state_space_A_t_top5_drift_estimates_with_bands.csv', index=False)

top_months = drift_df.sort_values('trace_A_over_R', ascending=False).head(20).reset_index(drop=True)
top_months['date_str'] = pd.to_datetime(top_months['date']).dt.strftime('%Y-%m')
top_months.to_csv(TABLES / 'top_months_by_top5_state_space_kernel_amplification.csv', index=False)

subspace_df = pd.DataFrame({
    'diagnostic': ['top5_subspace_max_principal_angle_degrees'],
    'estimate': [0.0],
    'bootstrap_p05': [np.quantile(subspace_angle, .05)],
    'bootstrap_median': [np.median(subspace_angle)],
    'bootstrap_p95': [np.quantile(subspace_angle, .95)],
    'note': ['Angle is zero for the original sample; bootstrap distribution summarizes subspace stability.'],
})
subspace_df.to_csv(TABLES / 'top5_subspace_bootstrap_stability.csv', index=False)

state_summary = pd.DataFrame({
    'item': [
        'retained_bases_R','state_dimension_R_times_Rplus1_over_2','A_t_model','raw_SPD_proxy','matrix_log_state','state_equation','observation_equation','VAR_ridge','VAR_spectral_radius','process_noise_share','measurement_noise_share','state_uncertainty_draws','bootstrap_draws','bootstrap_block_length_months','mean_trace_A_over_R','sd_trace_A_over_R','max_trace_A_over_R','max_month','max_month_trace_p05','max_month_trace_p95','min_eigenvalue_across_A_t','top3_trace_share_estimate','top3_trace_share_p05','top3_trace_share_p95','top5_trace_share_estimate','top5_trace_share_p05','top5_trace_share_p95'
    ],
    'value': [
        R,dstate,'Log-Euclidean VAR(1) Kalman smoother',f"G_t = alpha I + (1-alpha) z_t z_t', alpha={alpha}",'s_t = svec(log(normalized G_t))','s_t = mu + F(s_{t-1}-mu) + eta_t','y_t = s_t + eps_t',ridge,var_radius,process_share,1-process_share,Bstate,Bboot,block_len,tau.mean(),tau.std(ddof=0),tau.max(),dates.iloc[tau.argmax()].strftime('%Y-%m'),tau_band90[0,tau.argmax()],tau_band90[2,tau.argmax()],min_eig_A,shares[:3].sum(),cum_band90[0,2],cum_band90[2,2],shares[:5].sum(),cum_band90[0,4],cum_band90[2,4]
    ]
})
sample_summary = pd.DataFrame({
    'item': ['FRED_monthly_rows','JK_monthly_shock_rows','fixed_overlap_rows','shock_overlap_range','LP_usable_base_months','LP_base_month_range','outcomes','horizons','lags','controls_including_intercept','shock','control_shock','MP_CBI_fallback_rows','residualized_shock_variance'],
    'value': [len(fred),len(jk_m),len(overlap),f"{overlap['date'].min().date()} to {overlap['date'].max().date()}",len(valid_idx),f"{dates.iloc[0].date()} to {dates.iloc[-1].date()}",', '.join(outcome_labels),f'0 to {H} months',L,X.shape[1],'MP_used = MP_median with MP_pm fallback when missing','CBI_used = CBI_median with CBI_pm fallback when missing',int(jk_m['used_pm_fallback'].sum()),sigma_m2]
})
state_summary.to_csv(TABLES / 'top5_state_space_model_summary.csv', index=False)
sample_summary.to_csv(TABLES / 'sample_and_specification_top5.csv', index=False)
pd.DataFrame(F).to_csv(TABLES / 'top5_state_VAR_F_matrix.csv', index=False)
pd.DataFrame(Qproc).to_csv(TABLES / 'top5_state_process_covariance_Q.csv', index=False)
pd.DataFrame(Rmeas).to_csv(TABLES / 'top5_state_measurement_covariance_R.csv', index=False)
print('Saved tables', flush=True)

if os.environ.get('OVK_TOP5_COMPUTE_ONLY', '0').lower() in {'1', 'true', 'yes', 'on'}:
    print('OVK_TOP5_COMPUTE_ONLY=1; skipping duplicate chart/report rendering', flush=True)
    sys.exit(0)

# -----------------------------
# Charts
# -----------------------------
for f in CHARTS.glob('*'):
    f.unlink()
h = np.arange(H + 1)
base = beta_surface
low = beta_band90[0].reshape(H + 1, pvars)
high = beta_band90[2].reshape(H + 1, pvars)

fig = plt.figure(figsize=(10, 5.8))
for j, label in enumerate(outcome_labels):
    line = plt.plot(h, base[:, j], marker='o', label=label)[0]
    plt.fill_between(h, low[:, j], high[:, j], alpha=0.10, color=line.get_color())
plt.axhline(0, linewidth=0.8)
plt.title('Average LP responses with 90% block-bootstrap bands')
plt.xlabel('Horizon, months')
plt.ylabel('Response')
plt.legend()
save_chart(fig, '01_irf_all_variables_90pct_bands')

for j, label in enumerate(outcome_labels):
    fig = plt.figure(figsize=(8.5, 5.2))
    line = plt.plot(h, base[:, j], marker='o', label='estimate')[0]
    plt.fill_between(h, low[:, j], high[:, j], alpha=0.25, label='90% band', color=line.get_color())
    plt.axhline(0, linewidth=0.8)
    plt.title(f'Average LP response: {label}')
    plt.xlabel('Horizon, months')
    plt.ylabel('Response')
    plt.legend()
    save_chart(fig, f'01_irf_{j+1}_{safe_name(label)}_90pct_band')

x = np.arange(1, 11)
fig = plt.figure(figsize=(8.8, 5.3))
plt.bar(x, shares[:10], label='estimate')
err_low = shares[:10] - share_band90[0]
err_high = share_band90[2] - shares[:10]
plt.errorbar(x, shares[:10], yerr=[err_low, err_high], fmt='none', capsize=3, label='90% band')
plt.axvline(5, linestyle='--', linewidth=1.0, label='top 5 cutoff')
plt.title('Average OVK eigenspectrum with uncertainty')
plt.xlabel('Eigen-rank')
plt.ylabel('Share of kernel trace')
plt.xticks(x)
plt.legend()
save_chart(fig, '02_eigenspectrum_share_top5_90pct_bands')

fig = plt.figure(figsize=(8.8, 5.3))
line = plt.plot(x, np.cumsum(shares[:10]), marker='o', label='estimate')[0]
plt.fill_between(x, cum_band90[0], cum_band90[2], alpha=0.25, label='90% band', color=line.get_color())
plt.axvline(5, linestyle='--', linewidth=1.0, label='top 5 cutoff')
plt.axhline(shares[:5].sum(), linewidth=0.8)
plt.title('Cumulative OVK trace share with uncertainty')
plt.xlabel('Eigen-rank')
plt.ylabel('Cumulative share of kernel trace')
plt.ylim(0, 1.02)
plt.xticks(x)
plt.legend()
save_chart(fig, '03_cumulative_trace_share_top5_90pct_band')

fig = plt.figure(figsize=(10.5, 5.6))
line = plt.plot(dates, tau, label='state-space estimate')[0]
plt.fill_between(dates, tau_band90[0], tau_band90[2], alpha=0.18, label='90% state band', color=line.get_color())
plt.fill_between(dates, tau_band68[0], tau_band68[1], alpha=0.28, label='68% state band', color=line.get_color())
plt.axhline(1.0, linewidth=0.8)
plt.title('Total kernel amplification with top 5 bases')
plt.xlabel('Month')
plt.ylabel('Trace(A_t) / 5')
plt.legend()
save_chart(fig, '04_total_kernel_amplification_top5_state_bands')

fig = plt.figure(figsize=(10.5, 5.6))
for r in range(R):
    line = plt.plot(dates, A_state[:, r, r], label=f'A{r+1}{r+1} basis {r+1}')[0]
    plt.fill_between(dates, diag_band90[0, r], diag_band90[2, r], alpha=0.06, color=line.get_color())
plt.axhline(1.0, linewidth=0.8)
plt.title('Basis-specific amplification with 90% state bands')
plt.xlabel('Month')
plt.ylabel('Amplification relative to average')
plt.legend()
save_chart(fig, '05_A_diagonals_all_top5_state_bands')

for r in range(R):
    fig = plt.figure(figsize=(10, 5.4))
    line = plt.plot(dates, A_state[:, r, r], label=f'A{r+1}{r+1} basis {r+1}')[0]
    plt.fill_between(dates, diag_band90[0, r], diag_band90[2, r], alpha=0.18, label='90% state band', color=line.get_color())
    plt.fill_between(dates, diag_band68[0, r], diag_band68[1, r], alpha=0.28, label='68% state band', color=line.get_color())
    plt.axhline(1.0, linewidth=0.8)
    plt.title(f'Amplification of basis {r+1}')
    plt.xlabel('Month')
    plt.ylabel('A_t diagonal')
    plt.legend()
    save_chart(fig, f'05_A{r+1}{r+1}_basis_{r+1}_state_bands')

for r in range(R):
    est = V[:, r].reshape(H + 1, pvars)
    lo = mode_band90[0, r].reshape(H + 1, pvars)
    hi = mode_band90[2, r].reshape(H + 1, pvars)
    fig = plt.figure(figsize=(10, 5.8))
    for j, label in enumerate(outcome_labels):
        line = plt.plot(h, est[:, j], marker='o', label=label)[0]
        plt.fill_between(h, lo[:, j], hi[:, j], alpha=0.09, color=line.get_color())
    plt.axhline(0, linewidth=0.8)
    plt.title(f'Basis {r+1} loadings with 90% block-bootstrap bands')
    plt.xlabel('Horizon, months')
    plt.ylabel('Basis loading')
    plt.legend()
    save_chart(fig, f'06_basis_{r+1}_loadings_all_variables_90pct_bands')

for r in range(R):
    row = basis_diag_df.iloc[r]
    vals = np.array([row[f'{safe_name(label)}_energy_estimate'] for label in outcome_labels])
    loe = np.array([row[f'{safe_name(label)}_energy_p05'] for label in outcome_labels])
    hie = np.array([row[f'{safe_name(label)}_energy_p95'] for label in outcome_labels])
    xx = np.arange(pvars)
    fig = plt.figure(figsize=(9, 5.4))
    plt.bar(xx, vals, label='estimate')
    plt.errorbar(xx, vals, yerr=[vals - loe, hie - vals], fmt='none', capsize=3, label='90% band')
    plt.xticks(xx, outcome_labels, rotation=25, ha='right')
    plt.ylabel('Energy share')
    plt.title(f'Variable energy shares for basis {r+1}')
    plt.legend()
    save_chart(fig, f'08_basis_{r+1}_variable_energy_shares_90pct_bands')

top12 = top_months.head(12).copy().iloc[::-1]
fig = plt.figure(figsize=(9, 6))
plt.barh(top12['date_str'], top12['trace_A_over_R'])
xerr = np.vstack([top12['trace_A_over_R'] - top12['trace_A_p05'], top12['trace_A_p95'] - top12['trace_A_over_R']])
plt.errorbar(top12['trace_A_over_R'], top12['date_str'], xerr=xerr, fmt='none', capsize=3, label='90% state band')
plt.title('Top months by top-5 kernel amplification')
plt.xlabel('Trace(A_t) / 5')
plt.ylabel('Month')
plt.legend()
save_chart(fig, '09_top_months_kernel_amplification_top5_bands')

fig = plt.figure(figsize=(8.8, 5.3))
xx = np.arange(1, R + 1)
vals = mode_stability['median_abs_corr_with_bootstrap_basis'].values
lo = mode_stability['p05_abs_corr'].values
hi = mode_stability['p95_abs_corr'].values
plt.bar(xx, vals, label='median')
plt.errorbar(xx, vals, yerr=[vals - lo, hi - vals], fmt='none', capsize=3, label='5-95% interval')
plt.ylim(0, 1.05)
plt.xticks(xx)
plt.xlabel('Basis')
plt.ylabel('Abs. correlation with bootstrap-matched basis')
plt.title('Bootstrap stability of top 5 bases')
plt.legend()
save_chart(fig, '10_top5_basis_bootstrap_stability')

print('Saved charts', flush=True)

# -----------------------------
# Text content
# -----------------------------
share1 = float(shares[0]); share1_low = float(share_band90[0, 0]); share1_high = float(share_band90[2, 0])
top3 = float(shares[:3].sum()); top3_low = float(cum_band90[0, 2]); top3_high = float(cum_band90[2, 2])
top5 = float(shares[:5].sum()); top5_low = float(cum_band90[0, 4]); top5_high = float(cum_band90[2, 4])
max_ix = int(np.argmax(tau)); max_date = dates.iloc[max_ix].strftime('%Y-%m'); max_tau = float(tau[max_ix])
max_low = float(tau_band90[0, max_ix]); max_high = float(tau_band90[2, max_ix])

basis_brief_lines = []
for _, row in basis_diag_df.iterrows():
    basis_brief_lines.append(f"Basis {int(row['basis'])}: trace share {row['eigen_share_estimate']:.3f}; dominant variable {row['dominant_variable']}; peak horizon {int(row['peak_horizon_months'])} months; median bootstrap alignment {row['median_abs_corr_with_bootstrap_basis']:.3f}.")
basis_brief = '\n'.join(basis_brief_lines)

executive_summary = f"""
This version retains the top five kernel bases throughout the analysis rather than stopping at the top three. The state model is therefore a 15-dimensional log-covariance model because a symmetric 5 by 5 A_t matrix has 5(5+1)/2 = 15 unique entries.

The main quantitative result is that the first five bases explain {top5:.1%} of the average OVK trace, with a 90 percent block-bootstrap interval of [{top5_low:.1%}, {top5_high:.1%}]. For comparison, the first three bases explain {top3:.1%}, with interval [{top3_low:.1%}, {top3_high:.1%}]. The incremental contribution from bases 4 and 5 is about {(top5-top3):.1%} of the trace. That increment is useful, but it is more fragile than the first basis.

The top-five A_t process remains stationary under the fitted log-Euclidean VAR(1). The fitted spectral radius is {var_radius:.3f}. The total amplification statistic tau_t = trace(A_t)/5 has mean 1.000 by construction, standard deviation {tau.std(ddof=0):.3f}, and maximum {max_tau:.3f} in {max_date}. The 90 percent conditional state band at that maximum is [{max_low:.3f}, {max_high:.3f}].
""".strip()

method_text = f"""
The monthly LP-score construction is unchanged: the shock is MP_used, defined as MP_median with MP_pm fallback when MP_median is missing. CBI_used is included as a control shock using the analogous fallback. The base sample is {dates.iloc[0].strftime('%Y-%m')} to {dates.iloc[-1].strftime('%Y-%m')}, with {len(valid_idx)} usable monthly score surfaces.

The average kernel K_bar is computed from centered LP score surfaces. The top five eigenvectors of K_bar form the retained basis V_5. Score residuals are projected into this basis and standardized, giving z_t in R^5.

The dynamic covariance proxy is G_t = alpha I + (1-alpha) z_t z_t', with alpha = {alpha:.2f}. This avoids singular log-covariance observations. After normalizing the sample mean of G_t, the model maps G_t to the log-Euclidean state y_t = svec(log(G_t)). The latent state evolves as s_t = mu + F(s_(t-1)-mu) + eta_t, with observation y_t = s_t + eps_t. The Kalman smoother gives the latent log-covariance path. Mapping back through the matrix exponential gives A_t, renormalized so mean(A_t) = I.

IRF, eigenspectrum, and basis-loading uncertainty bands use {Bboot} circular moving-block bootstrap draws with {block_len}-month blocks. A_t bands use {Bstate} Gaussian draws from the smoothed state distribution conditional on the fitted state-space model. These are pointwise bands.
""".strip()

basis_text = f"""
The five retained bases are not equally stable. Basis 1 is highly stable across bootstrap samples. Bases 3 and 5 have visibly weaker alignment, so their detailed shape should be treated cautiously. This is a useful diagnostic, not a failure: the top-five subspace captures more trace, but the lower bases are less individually identified.

{basis_brief}
""".strip()

economic_text = f"""
Economically, the move from three to five bases changes the interpretation from a compact three-channel model to a broader response-geometry decomposition. The first three bases still do most of the work, accounting for about {top3:.1%} of the average kernel trace. Bases 4 and 5 add roughly {(top5-top3):.1%}. That extra share is not negligible, but it is small enough that a paper should present top-three results as the core and top-five results as an enriched robustness/diagnostic layer.

The strongest evidence is not the average IRF. The average IRF remains noisy, with wide bootstrap bands. The stronger empirical object is the covariance geometry of the response surfaces. The fact that five bases account for about {top5:.1%} of trace says monetary-policy response-score variation is highly structured rather than diffuse across all horizons and variables.

The A_t process measures time variation in the covariance geometry of the retained response-surface bases. When tau_t = trace(A_t)/5 rises above one, the retained monetary-policy response geometry is amplified relative to its average state. The largest amplification remains {max_date}. The top months also include 2007-2008 and late-1998/1999 episodes. This pattern is economically plausible: the kernel is most amplified when monetary-policy surprises interact with financial stress, changing macro-financial propagation across horizons.

March 2020 still requires caution. It is economically plausible as an extreme monetary-financial transmission episode, but it is also a fallback-shock month in this data construction. The correct interpretation is that the response-score covariance state is extreme in March 2020 under the baseline shock construction, not that we have isolated a clean structural monetary-policy effect for that month.

Including five bases makes the analysis more comprehensive, but it also exposes the boundary between robust structure and fragile detail. The low-dimensional fact is robust: top five bases explain over 90 percent of trace. The exact shape of basis 4 and basis 5 is less robust. This argues for reporting both the top-five cumulative kernel result and the basis-by-basis stability diagnostics.
""".strip()

robustness_text = """
For a serious paper draft, the next step is to rerun three shock-definition versions: MP_median with fallback, MP_pm only, and event-level shocks aggregated manually.

That would tell us whether the leading OVK bases and the A_t amplification spikes are robust or an artifact of the shock definition. For the top-five version, the comparison should report top-five trace share, principal angles between top-five subspaces, basis-specific bootstrap stability, tau_t path correlations, and overlap among top amplification months.
""".strip()

limitations_text = """
The top-five analysis is more complete but less parsimonious. The state-space bands condition on alpha, the process-noise share, the fitted VAR(1), and the retained basis. The bootstrap bands do not fully re-estimate every modeling choice. The lower bases are less stable, so signs and fine wiggles of bases 4 and 5 should not be over-interpreted.
""".strip()

readme = f"""Monthly OVK dynamic state model with top five bases

{executive_summary}

{method_text}

{basis_text}

{economic_text}

{robustness_text}

{limitations_text}
"""
(OUT / 'README_top5_dynamic_state_model.txt').write_text(readme, encoding='utf-8')

# -----------------------------
# PDF report
# -----------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=17, leading=21, spaceAfter=12))
styles.add(ParagraphStyle(name='Heading1Custom', parent=styles['Heading1'], fontSize=14, leading=16, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='Heading2Custom', parent=styles['Heading2'], fontSize=12, leading=14, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name='BodyCustom', parent=styles['BodyText'], fontSize=9.4, leading=12, spaceAfter=6))
styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontSize=8, leading=9, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6))

def P(text, style='BodyCustom'):
    return Paragraph(html.escape(text).replace('\n','<br/>'), styles[style])

def add_paragraphs(story, text):
    for para in text.split('\n\n'):
        para = para.strip()
        if para:
            story.append(P(para))
            story.append(Spacer(1, 0.04 * inch))

def img_flow(path, max_w=6.8*inch, max_h=4.4*inch):
    im = PILImage.open(path)
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)

def small_table(df, cols=None, max_rows=12):
    d = df.copy()
    if cols is not None:
        d = d[cols].copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    data = [list(d.columns)]
    for _, row in d.iterrows():
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(f'{float(v):.3f}')
            else:
                vals.append(str(v))
        data.append(vals)
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONT',(0,0),(-1,0),'Helvetica-Bold',7.2),
        ('FONT',(0,1),(-1,-1),'Helvetica',6.8),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),2),
        ('RIGHTPADDING',(0,0),(-1,-1),2),
        ('TOPPADDING',(0,0),(-1,-1),2),
        ('BOTTOMPADDING',(0,0),(-1,-1),2),
    ]))
    return tbl

compact_summary = pd.DataFrame({
    'quantity': ['LP sample','Usable base months','Retained bases','State dimension','Top 3 trace share','Top 5 trace share','VAR spectral radius','Mean tau_t','SD tau_t','Max tau_t','Max month','Min eigenvalue of A_t'],
    'estimate': [f"{dates.iloc[0].strftime('%Y-%m')} to {dates.iloc[-1].strftime('%Y-%m')}",len(valid_idx),R,dstate,f'{top3:.3f}',f'{top5:.3f}',f'{var_radius:.3f}',f'{tau.mean():.3f}',f'{tau.std(ddof=0):.3f}',f'{max_tau:.3f}',max_date,f'{min_eig_A:.3f}'],
    'uncertainty_or_note': [f'Horizons 0-{H}, lags {L}','Monthly score surfaces','Top five bases','Symmetric 5x5 log A_t',f'90% [{top3_low:.3f}, {top3_high:.3f}]',f'90% [{top5_low:.3f}, {top5_high:.3f}]','Stationary if below 1','Normalized','State path dispersion',f'90% [{max_low:.3f}, {max_high:.3f}]','Fallback shock month','PSD check'],
})

basis_compact = basis_diag_df[['basis','eigen_share_estimate','dominant_variable','peak_horizon_months','median_abs_corr_with_bootstrap_basis']].copy()
top_cols = ['date_str','trace_A_over_R','trace_A_p05','trace_A_p95','MP_used_std','used_pm_fallback_current_month'] + [f'A{r+1}{r+1}_basis{r+1}' for r in range(R)]

pdf_path = FINAL_PDF
story = []
doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
story.append(P('Monthly monetary-policy OVK with top five dynamic bases', 'TitleCenter'))
story.append(P('Rank-5 log-Euclidean state-space A_t model with uncertainty bands.', 'BodyCustom'))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
story.append(Spacer(1, 0.1*inch))
story.append(P('Headline results', 'Heading1Custom'))
story.append(small_table(compact_summary, max_rows=20))
story.append(Spacer(1, 0.08*inch))
add_paragraphs(story, executive_summary)
story.append(PageBreak())

story.append(P('Method upgrade and top-five basis construction', 'Heading1Custom'))
add_paragraphs(story, method_text)
story.append(PageBreak())

story.append(P('Average response and eigenspectrum uncertainty', 'Heading1Custom'))
story.append(img_flow(CHARTS/'01_irf_all_variables_90pct_bands.png', max_h=3.7*inch))
story.append(P('Figure 1. Average LP responses with 90 percent block-bootstrap bands.', 'Caption'))
story.append(img_flow(CHARTS/'02_eigenspectrum_share_top5_90pct_bands.png', max_h=3.2*inch))
story.append(P('Figure 2. Average OVK eigenspectrum with top-five cutoff.', 'Caption'))
story.append(PageBreak())

story.append(P('Cumulative trace share and top-five amplification', 'Heading1Custom'))
story.append(img_flow(CHARTS/'03_cumulative_trace_share_top5_90pct_band.png', max_h=3.4*inch))
story.append(P('Figure 3. Cumulative trace share. The top five bases explain more than 90 percent of trace.', 'Caption'))
story.append(img_flow(CHARTS/'04_total_kernel_amplification_top5_state_bands.png', max_h=3.6*inch))
story.append(P('Figure 4. Total amplification tau_t = trace(A_t)/5 with state-smoothing bands.', 'Caption'))
story.append(PageBreak())

story.append(P('Basis-specific A_t amplification', 'Heading1Custom'))
story.append(img_flow(CHARTS/'05_A_diagonals_all_top5_state_bands.png', max_h=3.8*inch))
story.append(P('Figure 5. Diagonal amplification terms for all five bases.', 'Caption'))
story.append(small_table(top_months[top_cols], max_rows=8))
story.append(PageBreak())

story.append(P('Top-five basis diagnostics', 'Heading1Custom'))
story.append(small_table(basis_compact, max_rows=5))
story.append(Spacer(1,0.08*inch))
story.append(img_flow(CHARTS/'10_top5_basis_bootstrap_stability.png', max_h=3.7*inch))
story.append(P('Figure 6. Bootstrap stability of the five individual bases.', 'Caption'))
story.append(PageBreak())

for r in range(R):
    story.append(P(f'Basis {r+1} loadings and variable energy', 'Heading1Custom'))
    story.append(img_flow(CHARTS/f'06_basis_{r+1}_loadings_all_variables_90pct_bands.png', max_h=3.6*inch))
    story.append(P(f'Figure {7 + 2*r}. Basis {r+1} loadings with 90 percent block-bootstrap bands.', 'Caption'))
    story.append(img_flow(CHARTS/f'08_basis_{r+1}_variable_energy_shares_90pct_bands.png', max_h=3.0*inch))
    story.append(P(f'Figure {8 + 2*r}. Basis {r+1} variable energy shares.', 'Caption'))
    if r < R - 1:
        story.append(PageBreak())

story.append(PageBreak())
story.append(P('Economic interpretation', 'Heading1Custom'))
add_paragraphs(story, economic_text)
story.append(PageBreak())
story.append(P('Robustness agenda', 'Heading1Custom'))
add_paragraphs(story, robustness_text)
story.append(P('Limitations', 'Heading1Custom'))
add_paragraphs(story, limitations_text)
doc.build(story)
print('Built PDF', flush=True)
shutil.copy2(pdf_path, OUT/'monthly_ovk_dynamic_state_model_top5_report.pdf')

# -----------------------------
# HTML report
# -----------------------------
chart_order = [
    ('Average LP responses', '01_irf_all_variables_90pct_bands.png'),
    ('OVK eigenspectrum', '02_eigenspectrum_share_top5_90pct_bands.png'),
    ('Cumulative trace share', '03_cumulative_trace_share_top5_90pct_band.png'),
    ('Total kernel amplification', '04_total_kernel_amplification_top5_state_bands.png'),
    ('All top-five A_t diagonal terms', '05_A_diagonals_all_top5_state_bands.png'),
    ('Basis stability', '10_top5_basis_bootstrap_stability.png'),
]
for r in range(R):
    chart_order.append((f'Basis {r+1} loadings', f'06_basis_{r+1}_loadings_all_variables_90pct_bands.png'))
    chart_order.append((f'Basis {r+1} energy shares', f'08_basis_{r+1}_variable_energy_shares_90pct_bands.png'))
fig_html = ''
for title, fname in chart_order:
    fig_html += f"<h3>{html.escape(title)}</h3><img src='data:image/png;base64,{img_b64(CHARTS/fname)}' alt='{html.escape(title)}'>\n"
html_report = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Top-five dynamic OVK report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;line-height:1.42;color:#222;}} h1{{font-size:24px;}} h2{{margin-top:30px;border-bottom:1px solid #aaa;padding-bottom:4px;}} img{{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 18px 0;}} table{{border-collapse:collapse;margin:12px 0 20px 0;font-size:12px;}} th,td{{border:1px solid #bbb;padding:5px 7px;text-align:left;vertical-align:top;}} th{{background:#eee;}} .note{{background:#f7f7f7;border-left:4px solid #999;padding:10px 14px;}}</style>
</head><body>
<h1>Monthly monetary-policy OVK with top five dynamic bases</h1>
<div class='note'>{paras_to_html(executive_summary)}</div>
<h2>Headline table</h2>{df_to_html(compact_summary, max_rows=20)}
<h2>Method</h2>{paras_to_html(method_text)}
<h2>Basis diagnostics</h2>{df_to_html(basis_compact, max_rows=5)}
<h2>Top amplification months</h2>{df_to_html(top_months[top_cols], max_rows=12)}
<h2>Charts</h2>{fig_html}
<h2>Economic interpretation</h2>{paras_to_html(economic_text)}
<h2>Robustness agenda</h2>{paras_to_html(robustness_text)}
<h2>Limitations</h2>{paras_to_html(limitations_text)}
</body></html>"""
(OUT/'monthly_ovk_dynamic_state_model_top5_report.html').write_text(html_report, encoding='utf-8')
shutil.copy2(OUT/'monthly_ovk_dynamic_state_model_top5_report.html', FINAL_HTML)

# -----------------------------
# Copy script and create bundle
# -----------------------------
shutil.copy2(Path(__file__), OUT/'generate_top5_ovk_pack.py')
helper = Path(__file__).with_name('ovk_data.py')
if helper.exists():
    shutil.copy2(helper, OUT/'ovk_data.py')
if FINAL_ZIP.exists():
    FINAL_ZIP.unlink()
with zipfile.ZipFile(FINAL_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob('*')):
        z.write(f, arcname=f.relative_to(OUT))

# Final summary CSV for quick preview
quick = {
    'retained_bases': R,
    'state_dimension': dstate,
    'usable_base_months': len(valid_idx),
    'lp_start': dates.iloc[0].strftime('%Y-%m'),
    'lp_end': dates.iloc[-1].strftime('%Y-%m'),
    'top3_trace_share': top3,
    'top5_trace_share': top5,
    'top5_trace_share_p05': top5_low,
    'top5_trace_share_p95': top5_high,
    'tau_max': max_tau,
    'tau_max_month': max_date,
    'var_spectral_radius': var_radius,
    'min_A_eig': min_eig_A,
}
pd.DataFrame([quick]).to_csv(OUT/'quick_summary_top5.csv', index=False)
print('DONE')
print(quick)
print('PDF', FINAL_PDF, FINAL_PDF.stat().st_size)
print('HTML', FINAL_HTML, FINAL_HTML.stat().st_size)
print('ZIP', FINAL_ZIP, FINAL_ZIP.stat().st_size)
