import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageOps, ImageDraw
import base64, html, shutil, zipfile, os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib import colors

ROOT = Path('/mnt/data/monthly_ovk_dynamic_state_model_top5_report')
OUT = ROOT/'outputs'
CHARTS = OUT/'charts'
TABLES = OUT/'tables'
CHARTS.mkdir(parents=True, exist_ok=True)
for f in CHARTS.glob('*'):
    f.unlink()

FINAL_PDF = Path('/mnt/data/monthly_ovk_dynamic_state_model_top5_report.pdf')
FINAL_HTML = Path('/mnt/data/monthly_ovk_dynamic_state_model_top5_report.html')
FINAL_ZIP = Path('/mnt/data/monthly_ovk_dynamic_state_model_top5_bundle.zip')
FINAL_CONTACT = Path('/mnt/data/monthly_ovk_dynamic_state_model_top5_contact_sheet.jpg')

# Read tables
irf = pd.read_csv(TABLES/'average_irf_with_block_bootstrap_bands.csv')
eigen = pd.read_csv(TABLES/'average_ovk_eigenspectrum_with_bands.csv')
basis_diag = pd.read_csv(TABLES/'top5_basis_diagnostics_with_bootstrap_bands.csv')
basis_load = pd.read_csv(TABLES/'top5_basis_loadings_with_bootstrap_bands.csv')
stability = pd.read_csv(TABLES/'top5_basis_bootstrap_stability.csv')
drift = pd.read_csv(TABLES/'state_space_A_t_top5_drift_estimates_with_bands.csv', parse_dates=['date'])
top_months = pd.read_csv(TABLES/'top_months_by_top5_state_space_kernel_amplification.csv', parse_dates=['date'])
state_summary = pd.read_csv(TABLES/'top5_state_space_model_summary.csv')
sample_summary = pd.read_csv(TABLES/'sample_and_specification_top5.csv')
subspace = pd.read_csv(TABLES/'top5_subspace_bootstrap_stability.csv')

R=5
outcome_labels = irf['variable'].drop_duplicates().tolist()
H = int(irf['horizon_months'].max())
h = np.arange(H+1)

def get_value(item):
    vals = state_summary.loc[state_summary['item']==item,'value']
    if len(vals)==0:
        return None
    v = vals.iloc[0]
    try:
        return float(v)
    except Exception:
        return v

top3 = get_value('top3_trace_share_estimate')
top3_low = get_value('top3_trace_share_p05')
top3_high = get_value('top3_trace_share_p95')
top5 = get_value('top5_trace_share_estimate')
top5_low = get_value('top5_trace_share_p05')
top5_high = get_value('top5_trace_share_p95')
max_tau = get_value('max_trace_A_over_R')
max_month = get_value('max_month')
max_low = get_value('max_month_trace_p05')
max_high = get_value('max_month_trace_p95')
var_radius = get_value('VAR_spectral_radius')
state_dim = int(get_value('state_dimension_R_times_Rplus1_over_2'))
min_eig = get_value('min_eigenvalue_across_A_t')
tau_sd = get_value('sd_trace_A_over_R')

# Plot helpers
x_date = np.arange(len(drift))
dates = drift['date']
tick_pos = np.linspace(0, len(drift)-1, 9, dtype=int)
tick_labels = dates.iloc[tick_pos].dt.strftime('%Y').tolist()

def savefig(fig, name):
    path = CHARTS/f'{name}.png'
    fig.savefig(path, dpi=165, bbox_inches='tight')
    plt.close(fig)
    return path

# IRF all variables
fig, ax = plt.subplots(figsize=(10,5.8))
for label in outcome_labels:
    d = irf[irf['variable']==label]
    line = ax.plot(d['horizon_months'], d['estimate'], marker='o', label=label)[0]
    ax.fill_between(d['horizon_months'].to_numpy(), d['boot_p05'].to_numpy(), d['boot_p95'].to_numpy(), alpha=0.10, color=line.get_color())
ax.axhline(0, linewidth=0.8)
ax.set_title('Average LP responses with 90% block-bootstrap bands')
ax.set_xlabel('Horizon, months')
ax.set_ylabel('Response')
ax.legend()
savefig(fig, '01_irf_all_variables_90pct_bands')

# Eigenspectrum and cumulative
fig, ax = plt.subplots(figsize=(8.8,5.3))
x = eigen['rank'].to_numpy()
share = eigen['share_estimate'].to_numpy()
ax.bar(x, share, label='estimate')
ax.errorbar(x, share, yerr=[share-eigen['share_p05'].to_numpy(), eigen['share_p95'].to_numpy()-share], fmt='none', capsize=3, label='90% band')
ax.axvline(5, linestyle='--', linewidth=1.0, label='top 5 cutoff')
ax.set_title('Average OVK eigenspectrum with uncertainty')
ax.set_xlabel('Eigen-rank')
ax.set_ylabel('Share of kernel trace')
ax.set_xticks(x)
ax.legend()
savefig(fig, '02_eigenspectrum_share_top5_90pct_bands')

fig, ax = plt.subplots(figsize=(8.8,5.3))
line = ax.plot(x, eigen['cumulative_estimate'], marker='o', label='estimate')[0]
ax.fill_between(x, eigen['cumulative_p05'], eigen['cumulative_p95'], alpha=0.25, label='90% band', color=line.get_color())
ax.axvline(5, linestyle='--', linewidth=1.0, label='top 5 cutoff')
ax.axhline(top5, linewidth=0.8)
ax.set_title('Cumulative OVK trace share with uncertainty')
ax.set_xlabel('Eigen-rank')
ax.set_ylabel('Cumulative share of kernel trace')
ax.set_ylim(0,1.02)
ax.set_xticks(x)
ax.legend()
savefig(fig, '03_cumulative_trace_share_top5_90pct_band')

# Total A amplification
fig, ax = plt.subplots(figsize=(10.5,5.6))
line = ax.plot(x_date, drift['trace_A_over_R'], label='state-space estimate')[0]
ax.fill_between(x_date, drift['trace_A_p05'], drift['trace_A_p95'], alpha=0.18, label='90% state band', color=line.get_color())
ax.fill_between(x_date, drift['trace_A_p16'], drift['trace_A_p84'], alpha=0.28, label='68% state band', color=line.get_color())
ax.axhline(1.0, linewidth=0.8)
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labels)
ax.set_title('Total kernel amplification with top 5 bases')
ax.set_xlabel('Month')
ax.set_ylabel('Trace(A_t) / 5')
ax.legend()
savefig(fig, '04_total_kernel_amplification_top5_state_bands')

# Diagonals all and individual
fig, ax = plt.subplots(figsize=(10.5,5.6))
for r in range(1, R+1):
    y = drift[f'A{r}{r}_basis{r}']
    line = ax.plot(x_date, y, label=f'A{r}{r} basis {r}')[0]
    ax.fill_between(x_date, drift[f'A{r}{r}_p05'], drift[f'A{r}{r}_p95'], alpha=0.06, color=line.get_color())
ax.axhline(1.0, linewidth=0.8)
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labels)
ax.set_title('Basis-specific amplification with 90% state bands')
ax.set_xlabel('Month')
ax.set_ylabel('Amplification relative to average')
ax.legend()
savefig(fig, '05_A_diagonals_all_top5_state_bands')

for r in range(1, R+1):
    fig, ax = plt.subplots(figsize=(10,5.4))
    line = ax.plot(x_date, drift[f'A{r}{r}_basis{r}'], label=f'A{r}{r} basis {r}')[0]
    ax.fill_between(x_date, drift[f'A{r}{r}_p05'], drift[f'A{r}{r}_p95'], alpha=0.18, label='90% state band', color=line.get_color())
    ax.fill_between(x_date, drift[f'A{r}{r}_p16'], drift[f'A{r}{r}_p84'], alpha=0.28, label='68% state band', color=line.get_color())
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_title(f'Amplification of basis {r}')
    ax.set_xlabel('Month')
    ax.set_ylabel('A_t diagonal')
    ax.legend()
    savefig(fig, f'05_A{r}{r}_basis_{r}_state_bands')

# Basis loading and energy charts
for r in range(1, R+1):
    fig, ax = plt.subplots(figsize=(10,5.8))
    for label in outcome_labels:
        d = basis_load[(basis_load['basis']==r) & (basis_load['variable']==label)]
        line = ax.plot(d['horizon_months'], d['loading_estimate'], marker='o', label=label)[0]
        ax.fill_between(d['horizon_months'].to_numpy(), d['boot_p05'].to_numpy(), d['boot_p95'].to_numpy(), alpha=0.09, color=line.get_color())
    ax.axhline(0, linewidth=0.8)
    ax.set_title(f'Basis {r} loadings with 90% block-bootstrap bands')
    ax.set_xlabel('Horizon, months')
    ax.set_ylabel('Basis loading')
    ax.legend()
    savefig(fig, f'06_basis_{r}_loadings_all_variables_90pct_bands')

    row = basis_diag[basis_diag['basis']==r].iloc[0]
    vals=[]; lo=[]; hi=[]
    for label in outcome_labels:
        key = label.lower().replace(' ','_').replace('-','').replace('/','_')
        vals.append(row[f'{key}_energy_estimate'])
        lo.append(row[f'{key}_energy_p05'])
        hi.append(row[f'{key}_energy_p95'])
    vals=np.array(vals); lo=np.array(lo); hi=np.array(hi)
    xx=np.arange(len(outcome_labels))
    fig, ax = plt.subplots(figsize=(9,5.4))
    ax.bar(xx, vals, label='estimate')
    ax.errorbar(xx, vals, yerr=[vals-lo, hi-vals], fmt='none', capsize=3, label='90% band')
    ax.set_xticks(xx)
    ax.set_xticklabels(outcome_labels, rotation=25, ha='right')
    ax.set_ylabel('Energy share')
    ax.set_title(f'Variable energy shares for basis {r}')
    ax.legend()
    savefig(fig, f'08_basis_{r}_variable_energy_shares_90pct_bands')

# Top months and stability
plot_top = top_months.head(12).copy().iloc[::-1]
fig, ax = plt.subplots(figsize=(9,6))
ax.barh(plot_top['date_str'], plot_top['trace_A_over_R'])
xerr=np.vstack([plot_top['trace_A_over_R']-plot_top['trace_A_p05'], plot_top['trace_A_p95']-plot_top['trace_A_over_R']])
ax.errorbar(plot_top['trace_A_over_R'], plot_top['date_str'], xerr=xerr, fmt='none', capsize=3, label='90% state band')
ax.set_title('Top months by top-5 kernel amplification')
ax.set_xlabel('Trace(A_t) / 5')
ax.set_ylabel('Month')
ax.legend()
savefig(fig, '09_top_months_kernel_amplification_top5_bands')

fig, ax = plt.subplots(figsize=(8.8,5.3))
xx=stability['basis'].to_numpy()
vals=stability['median_abs_corr_with_bootstrap_basis'].to_numpy()
lo=stability['p05_abs_corr'].to_numpy()
hi=stability['p95_abs_corr'].to_numpy()
ax.bar(xx, vals, label='median')
ax.errorbar(xx, vals, yerr=[vals-lo, hi-vals], fmt='none', capsize=3, label='5-95% interval')
ax.set_ylim(0,1.05)
ax.set_xticks(xx)
ax.set_xlabel('Basis')
ax.set_ylabel('Abs. correlation with bootstrap-matched basis')
ax.set_title('Bootstrap stability of top 5 bases')
ax.legend()
savefig(fig, '10_top5_basis_bootstrap_stability')

# Text content
basis_brief_lines=[]
for _, row in basis_diag.iterrows():
    basis_brief_lines.append(f"Basis {int(row['basis'])}: trace share {row['eigen_share_estimate']:.3f}; dominant variable {row['dominant_variable']}; peak horizon {int(row['peak_horizon_months'])} months; median bootstrap alignment {row['median_abs_corr_with_bootstrap_basis']:.3f}.")
basis_brief='\n'.join(basis_brief_lines)

executive_summary=f"""
This version retains the top five kernel bases throughout the analysis rather than stopping at the top three. The state model is therefore a 15-dimensional log-covariance model because a symmetric 5 by 5 A_t matrix has 15 unique entries.

The first five bases explain {top5:.1%} of the average OVK trace, with a 90 percent block-bootstrap interval of [{top5_low:.1%}, {top5_high:.1%}]. For comparison, the first three bases explain {top3:.1%}, with interval [{top3_low:.1%}, {top3_high:.1%}]. The incremental contribution from bases 4 and 5 is about {(top5-top3):.1%} of the trace.

The top-five A_t process remains stationary under the fitted log-Euclidean VAR(1). The fitted spectral radius is {var_radius:.3f}. The total amplification statistic tau_t = trace(A_t)/5 has maximum {max_tau:.3f} in {max_month}. The 90 percent conditional state band at that maximum is [{max_low:.3f}, {max_high:.3f}].
""".strip()

method_text=f"""
The LP-score construction uses MP_used, defined as MP_median with MP_pm fallback when MP_median is missing. CBI_used is included as a control shock using the analogous fallback. The average kernel K_bar is computed from centered monthly LP score surfaces, and the top five eigenvectors form the retained basis V_5.

The dynamic covariance proxy is G_t = alpha I + (1-alpha) z_t z_t', with alpha = 0.25. After normalizing its sample mean, the model maps G_t to y_t = svec(log(G_t)). The latent state evolves as s_t = mu + F(s_(t-1)-mu) + eta_t, with observation y_t = s_t + eps_t. The Kalman smoother gives the latent log-covariance path. Mapping back through the matrix exponential gives A_t, renormalized so mean(A_t) = I.

IRF, eigenspectrum, and basis-loading uncertainty bands use 400 circular moving-block bootstrap draws with 18-month blocks. A_t bands use 250 Gaussian draws from the smoothed state distribution conditional on the fitted state-space model. These are pointwise bands.
""".strip()

basis_text=f"""
The five retained bases are not equally stable. Basis 1 is highly stable across bootstrap samples. Bases 3 and 5 have visibly weaker alignment, so their detailed shape should be treated cautiously. This is a useful diagnostic: the top-five subspace captures more trace, but the lower bases are less individually identified.

{basis_brief}
""".strip()

economic_text=f"""
Economically, the move from three to five bases changes the interpretation from a compact three-channel model to a broader response-geometry decomposition. The first three bases still do most of the work, accounting for about {top3:.1%} of the average kernel trace. Bases 4 and 5 add roughly {(top5-top3):.1%}. That extra share is useful, but it is small enough that a paper should present top-three results as the core and top-five results as an enriched robustness layer.

The strongest evidence is not the average IRF. The average IRF remains noisy, with wide bootstrap bands. The stronger empirical object is the covariance geometry of the response surfaces. The fact that five bases account for about {top5:.1%} of trace says monetary-policy response-score variation is structured rather than diffuse across all horizons and variables.

The A_t process measures time variation in the covariance geometry of the retained response-surface bases. When tau_t = trace(A_t)/5 rises above one, the retained monetary-policy response geometry is amplified relative to its average state. The largest amplification remains {max_month}. The top months also include 2007-2008 and late-1998/1999 episodes. This pattern is economically plausible: the kernel is most amplified when monetary-policy surprises interact with financial stress, changing macro-financial propagation across horizons.

March 2020 still requires caution. It is economically plausible as an extreme monetary-financial transmission episode, but it is also a fallback-shock month in this data construction. The correct interpretation is that the response-score covariance state is extreme in March 2020 under the baseline shock construction, not that we have isolated a clean structural monetary-policy effect for that month.

Including five bases makes the analysis more comprehensive, but it also exposes the boundary between robust structure and fragile detail. The low-dimensional fact is robust: top five bases explain over 90 percent of trace. The exact shape of basis 4 and basis 5 is less robust. This argues for reporting both the top-five cumulative kernel result and the basis-by-basis stability diagnostics.
""".strip()

robustness_text="""
For a serious paper draft, the next step is to rerun three shock-definition versions: MP_median with fallback, MP_pm only, and event-level shocks aggregated manually.

That would tell us whether the leading OVK bases and the A_t amplification spikes are robust or an artifact of the shock definition. For the top-five version, the comparison should report top-five trace share, principal angles between top-five subspaces, basis-specific bootstrap stability, tau_t path correlations, and overlap among top amplification months.
""".strip()

limitations_text="""
The top-five analysis is more complete but less parsimonious. The state-space bands condition on alpha, the process-noise share, the fitted VAR(1), and the retained basis. The bootstrap bands do not fully re-estimate every modeling choice. The lower bases are less stable, so signs and fine wiggles of bases 4 and 5 should not be over-interpreted.
""".strip()

readme = f"Monthly OVK dynamic state model with top five bases\n\n{executive_summary}\n\n{method_text}\n\n{basis_text}\n\n{economic_text}\n\n{robustness_text}\n\n{limitations_text}\n"
(OUT/'README_top5_dynamic_state_model.txt').write_text(readme, encoding='utf-8')

# PDF report
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=17, leading=21, spaceAfter=12))
styles.add(ParagraphStyle(name='Heading1Custom', parent=styles['Heading1'], fontSize=14, leading=16, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='BodyCustom', parent=styles['BodyText'], fontSize=9.4, leading=12, spaceAfter=6))
styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontSize=8, leading=9, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6))

def P(text, style='BodyCustom'):
    return Paragraph(html.escape(str(text)).replace('\n','<br/>'), styles[style])

def add_paragraphs(story, text):
    for para in text.split('\n\n'):
        if para.strip():
            story.append(P(para.strip()))
            story.append(Spacer(1, 0.04*inch))

def img_flow(path, max_w=6.8*inch, max_h=4.4*inch):
    im=PILImage.open(path)
    w,h=im.size
    scale=min(max_w/w, max_h/h)
    return Image(str(path), width=w*scale, height=h*scale)

def small_table(df, cols=None, max_rows=12):
    d=df.copy()
    if cols is not None:
        d=d[cols].copy()
    if len(d)>max_rows:
        d=d.head(max_rows)
    data=[list(d.columns)]
    for _, row in d.iterrows():
        vals=[]
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(f'{float(v):.3f}')
            else:
                vals.append(str(v))
        data.append(vals)
    tbl=Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.25,colors.grey),('FONT',(0,0),(-1,0),'Helvetica-Bold',7.1),('FONT',(0,1),(-1,-1),'Helvetica',6.7),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)
    ]))
    return tbl

compact_summary=pd.DataFrame({
    'quantity':['LP sample','Usable months','Retained bases','State dimension','Top 3 trace share','Top 5 trace share','VAR spectral radius','SD tau_t','Max tau_t','Max month','Min eig(A_t)'],
    'estimate':[sample_summary.loc[sample_summary['item']=='LP_base_month_range','value'].iloc[0], sample_summary.loc[sample_summary['item']=='LP_usable_base_months','value'].iloc[0], 5, state_dim, f'{top3:.3f}', f'{top5:.3f}', f'{var_radius:.3f}', f'{tau_sd:.3f}', f'{max_tau:.3f}', max_month, f'{min_eig:.3f}'],
    'note':[f'Horizons 0-24, lags 12','Monthly score surfaces','Top five bases','Symmetric 5x5 log A_t',f'90% [{top3_low:.3f}, {top3_high:.3f}]',f'90% [{top5_low:.3f}, {top5_high:.3f}]','Stationary if below 1','State path dispersion',f'90% [{max_low:.3f}, {max_high:.3f}]','Fallback shock month','PSD check']
})
basis_compact=basis_diag[['basis','eigen_share_estimate','dominant_variable','peak_horizon_months','median_abs_corr_with_bootstrap_basis']]
top_cols=['date_str','trace_A_over_R','trace_A_p05','trace_A_p95','MP_used_std','used_pm_fallback_current_month'] + [f'A{r}{r}_basis{r}' for r in range(1,6)]

story=[]
doc=SimpleDocTemplate(str(FINAL_PDF), pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
story.append(P('Monthly monetary-policy OVK with top five dynamic bases','TitleCenter'))
story.append(P('Rank-5 log-Euclidean state-space A_t model with uncertainty bands.','BodyCustom'))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
story.append(Spacer(1,0.1*inch))
story.append(P('Headline results','Heading1Custom'))
story.append(small_table(compact_summary, max_rows=20))
story.append(Spacer(1,0.08*inch))
add_paragraphs(story, executive_summary)
story.append(PageBreak())
story.append(P('Method and top-five basis construction','Heading1Custom'))
add_paragraphs(story, method_text)
story.append(PageBreak())
story.append(P('Average response and eigenspectrum uncertainty','Heading1Custom'))
story.append(img_flow(CHARTS/'01_irf_all_variables_90pct_bands.png', max_h=3.7*inch))
story.append(P('Figure 1. Average LP responses with 90 percent block-bootstrap bands.','Caption'))
story.append(img_flow(CHARTS/'02_eigenspectrum_share_top5_90pct_bands.png', max_h=3.2*inch))
story.append(P('Figure 2. Average OVK eigenspectrum with top-five cutoff.','Caption'))
story.append(PageBreak())
story.append(P('Cumulative trace share and top-five amplification','Heading1Custom'))
story.append(img_flow(CHARTS/'03_cumulative_trace_share_top5_90pct_band.png', max_h=3.4*inch))
story.append(P('Figure 3. Cumulative trace share.','Caption'))
story.append(img_flow(CHARTS/'04_total_kernel_amplification_top5_state_bands.png', max_h=3.6*inch))
story.append(P('Figure 4. Total amplification tau_t = trace(A_t)/5.','Caption'))
story.append(PageBreak())
story.append(P('Basis-specific A_t amplification','Heading1Custom'))
story.append(img_flow(CHARTS/'05_A_diagonals_all_top5_state_bands.png', max_h=3.7*inch))
story.append(P('Figure 5. Diagonal amplification terms for all five bases.','Caption'))
story.append(small_table(top_months[top_cols], max_rows=8))
story.append(PageBreak())
story.append(P('Top-five basis diagnostics','Heading1Custom'))
story.append(small_table(basis_compact, max_rows=5))
story.append(Spacer(1,0.08*inch))
story.append(img_flow(CHARTS/'10_top5_basis_bootstrap_stability.png', max_h=3.5*inch))
story.append(P('Figure 6. Bootstrap stability of the five individual bases.','Caption'))
story.append(PageBreak())
for r in range(1,6):
    story.append(P(f'Basis {r} loadings and variable energy','Heading1Custom'))
    story.append(img_flow(CHARTS/f'06_basis_{r}_loadings_all_variables_90pct_bands.png', max_h=3.4*inch))
    story.append(P(f'Basis {r} loadings with 90 percent block-bootstrap bands.','Caption'))
    story.append(img_flow(CHARTS/f'08_basis_{r}_variable_energy_shares_90pct_bands.png', max_h=2.9*inch))
    story.append(P(f'Basis {r} variable energy shares.','Caption'))
    if r<5:
        story.append(PageBreak())
story.append(PageBreak())
story.append(P('Economic interpretation','Heading1Custom'))
add_paragraphs(story, economic_text)
story.append(PageBreak())
story.append(P('Robustness agenda','Heading1Custom'))
add_paragraphs(story, robustness_text)
story.append(P('Limitations','Heading1Custom'))
add_paragraphs(story, limitations_text)
doc.build(story)
shutil.copy2(FINAL_PDF, OUT/'monthly_ovk_dynamic_state_model_top5_report.pdf')



def paras_to_html(text):
    return '\n'.join(f'<p>{html.escape(p.strip())}</p>' for p in str(text).split('\n\n') if p.strip())

def df_to_html(df, cols=None, max_rows=20):
    d = df.copy()
    if cols is not None:
        d = d[cols].copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    rows = []
    rows.append('<tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in d.columns) + '</tr>')
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(f'<td>{float(v):.3f}</td>')
            else:
                cells.append(f'<td>{html.escape(str(v))}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '<table>' + '\n'.join(rows) + '</table>'

# HTML
chart_order=[('Average LP responses','01_irf_all_variables_90pct_bands.png'),('OVK eigenspectrum','02_eigenspectrum_share_top5_90pct_bands.png'),('Cumulative trace share','03_cumulative_trace_share_top5_90pct_band.png'),('Total kernel amplification','04_total_kernel_amplification_top5_state_bands.png'),('All top-five A_t diagonal terms','05_A_diagonals_all_top5_state_bands.png'),('Basis stability','10_top5_basis_bootstrap_stability.png')]
for r in range(1,6):
    chart_order.append((f'Basis {r} loadings', f'06_basis_{r}_loadings_all_variables_90pct_bands.png'))
    chart_order.append((f'Basis {r} energy shares', f'08_basis_{r}_variable_energy_shares_90pct_bands.png'))
fig_html=''
for title, fname in chart_order:
    b64=base64.b64encode((CHARTS/fname).read_bytes()).decode('ascii')
    fig_html += f"<h3>{html.escape(title)}</h3><img src='data:image/png;base64,{b64}' alt='{html.escape(title)}'>\n"
html_report=f"""<!doctype html><html><head><meta charset='utf-8'><title>Top-five dynamic OVK report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;line-height:1.42;color:#222;}} h1{{font-size:24px;}} h2{{margin-top:30px;border-bottom:1px solid #aaa;padding-bottom:4px;}} img{{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 18px 0;}} table{{border-collapse:collapse;margin:12px 0 20px 0;font-size:12px;}} th,td{{border:1px solid #bbb;padding:5px 7px;text-align:left;vertical-align:top;}} th{{background:#eee;}} .note{{background:#f7f7f7;border-left:4px solid #999;padding:10px 14px;}}</style>
</head><body><h1>Monthly monetary-policy OVK with top five dynamic bases</h1><div class='note'>{paras_to_html(executive_summary)}</div><h2>Headline table</h2>{df_to_html(compact_summary, max_rows=20)}<h2>Method</h2>{paras_to_html(method_text)}<h2>Basis diagnostics</h2>{df_to_html(basis_compact, max_rows=5)}<h2>Top amplification months</h2>{df_to_html(top_months[top_cols], max_rows=12)}<h2>Charts</h2>{fig_html}<h2>Economic interpretation</h2>{paras_to_html(economic_text)}<h2>Robustness agenda</h2>{paras_to_html(robustness_text)}<h2>Limitations</h2>{paras_to_html(limitations_text)}</body></html>"""
(OUT/'monthly_ovk_dynamic_state_model_top5_report.html').write_text(html_report, encoding='utf-8')
shutil.copy2(OUT/'monthly_ovk_dynamic_state_model_top5_report.html', FINAL_HTML)

# Copy scripts into output
shutil.copy2('/mnt/data/generate_top5_ovk_pack.py', OUT/'generate_top5_ovk_pack.py')
shutil.copy2('/mnt/data/finalize_top5_from_tables.py', OUT/'finalize_top5_from_tables.py')

# Contact sheet from key images (not PDF render yet)
thumbs=[]
for title, fname in chart_order[:12]:
    im=PILImage.open(CHARTS/fname).convert('RGB')
    im=ImageOps.contain(im,(280,190))
    canvas=PILImage.new('RGB',(300,230),'white')
    canvas.paste(im,((300-im.width)//2,30))
    d=ImageDraw.Draw(canvas)
    d.text((8,8),title[:38],fill='black')
    thumbs.append(canvas)
cols=2; rows=(len(thumbs)+cols-1)//cols
sheet=PILImage.new('RGB',(cols*300,rows*230),'white')
for i,t in enumerate(thumbs):
    sheet.paste(t,((i%cols)*300,(i//cols)*230))
sheet.save(OUT/'chart_contact_sheet_top5.jpg', quality=88)

# Final zip
if FINAL_ZIP.exists(): FINAL_ZIP.unlink()
with zipfile.ZipFile(FINAL_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob('*')):
        z.write(f, arcname=f.relative_to(OUT))

print('DONE')
print('Charts', len(list(CHARTS.glob('*.png'))))
print('PDF', FINAL_PDF, FINAL_PDF.stat().st_size)
print('HTML', FINAL_HTML, FINAL_HTML.stat().st_size)
print('ZIP', FINAL_ZIP, FINAL_ZIP.stat().st_size)
