from __future__ import annotations

import base64
import html
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage, ImageOps, ImageDraw
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable

BASELINE_OUT = Path(os.environ.get('OVK_TOP5_OUT', '/mnt/data/monthly_ovk_dynamic_state_model_top5_report/outputs'))
BASELINE_PDF = Path(os.environ.get('OVK_TOP5_FINAL_PDF', '/mnt/data/monthly_ovk_dynamic_state_model_top5_report.pdf'))
BASELINE_HTML = Path(os.environ.get('OVK_TOP5_FINAL_HTML', '/mnt/data/monthly_ovk_dynamic_state_model_top5_report.html'))
ROBUST_OUT = Path(os.environ.get('OVK_ROBUST_OUT', '/mnt/data/top5_robustness_point_estimates/outputs'))
DATA_ZIP = Path(os.environ.get('OVK_DATA_ZIP', '/mnt/data/data.zip'))

PACK = Path(os.environ.get('OVK_FULL_PACK', '/mnt/data/monthly_ovk_top5_full_appended_results_pack'))
REPORTS = PACK / 'reports'
DATA_RAW = PACK / 'data_raw'
DATA_PROCESSED = PACK / 'data_processed'
BASELINE_DIR = PACK / 'top5_baseline_state_space_results'
ROBUST_DIR = PACK / 'robustness_comparison_results'
CODE_DIR = PACK / 'code'
for d in [REPORTS, DATA_RAW, DATA_PROCESSED, BASELINE_DIR, ROBUST_DIR, CODE_DIR]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

APPENDIX_PDF = REPORTS / 'robustness_appendix.pdf'
FULL_PDF = Path(os.environ.get('OVK_FULL_FINAL_PDF', '/mnt/data/monthly_ovk_top5_full_appended_report.pdf'))
FULL_HTML = Path(os.environ.get('OVK_FULL_FINAL_HTML', '/mnt/data/monthly_ovk_top5_full_appended_report.html'))
FULL_ZIP = Path(os.environ.get('OVK_FULL_FINAL_ZIP', '/mnt/data/monthly_ovk_top5_full_appended_results_pack.zip'))
CONTACT = Path(os.environ.get('OVK_FULL_FINAL_CONTACT', '/mnt/data/monthly_ovk_top5_full_appended_contact_sheet.jpg'))
for p in [FULL_PDF, FULL_HTML, FULL_ZIP, CONTACT]:
    p.parent.mkdir(parents=True, exist_ok=True)

# Copy raw uploaded data and extract it.
shutil.copy2(DATA_ZIP, DATA_RAW / 'data.zip')
with zipfile.ZipFile(DATA_ZIP) as z:
    z.extractall(DATA_RAW / 'extracted')

# Copy existing top-five baseline pack outputs.
if BASELINE_OUT.exists():
    shutil.copytree(BASELINE_OUT, BASELINE_DIR, dirs_exist_ok=True)
if BASELINE_PDF.exists():
    shutil.copy2(BASELINE_PDF, REPORTS / 'top5_baseline_state_space_report.pdf')
if BASELINE_HTML.exists():
    shutil.copy2(BASELINE_HTML, REPORTS / 'top5_baseline_state_space_report.html')

# Copy robustness outputs.
shutil.copytree(ROBUST_OUT, ROBUST_DIR, dirs_exist_ok=True)

# Copy key processed panels to data_processed.
for p in [
    BASELINE_OUT / 'ovk_monetary_panel_monthly_fixed_full.csv',
    BASELINE_OUT / 'ovk_monetary_panel_monthly_fixed_overlap.csv',
    ROBUST_OUT / 'processed_panel_three_shock_definitions.csv',
    ROBUST_OUT / 'monthly_shocks_repaired.csv',
    ROBUST_OUT / 'event_shocks_with_manual_fields.csv',
    ROBUST_OUT / 'event_level_manual_monthly_aggregation.csv',
]:
    if p.exists():
        shutil.copy2(p, DATA_PROCESSED / p.name)

# Copy code used.
for p in [
    Path(os.environ.get('OVK_GENERATE_SCRIPT', '/mnt/data/generate_top5_ovk_pack.py')),
    Path(os.environ.get('OVK_FINALIZE_SCRIPT', '/mnt/data/finalize_top5_from_tables.py')),
    BASELINE_OUT / 'generate_top5_ovk_pack.py',
    BASELINE_OUT / 'finalize_top5_from_tables.py',
    ROBUST_OUT / 'code' / 'run_top5_robustness_point_estimates.py',
    ROBUST_OUT / 'code' / 'ovk_data.py',
    BASELINE_OUT / 'ovk_data.py',
    Path(__file__).with_name('ovk_data.py'),
    Path(os.environ.get('OVK_ROBUST_SCRIPT', '/mnt/data/run_top5_robustness_point_estimates.py')),
    Path(os.environ.get('OVK_FINAL_PACK_SCRIPT', '/mnt/data/make_final_appended_top5_pack.py')),
]:
    if p.exists():
        target = CODE_DIR / p.name
        if target.exists():
            target = CODE_DIR / f'{p.stem}_copy{p.suffix}'
        shutil.copy2(p, target)
(CODE_DIR / 'requirements.txt').write_text('numpy\npandas\nmatplotlib\nscipy\nscikit-learn\nPillow\nreportlab\npypdf\n', encoding='utf-8')
(CODE_DIR / 'RUN_ORDER.txt').write_text(
    '1. python generate_top5_ovk_pack.py\n'
    '2. python finalize_top5_from_tables.py\n'
    '3. python run_top5_robustness_point_estimates.py\n'
    '4. python make_final_appended_top5_pack.py\n',
    encoding='utf-8'
)

# Load tables.
T = ROBUST_OUT / 'tables'
comp = pd.read_csv(T / 'robustness_comparison_metrics.csv')
summary = pd.read_csv(T / 'robustness_variant_summary.csv')
diag = pd.read_csv(T / 'basis_specific_A_diag_path_correlations.csv')
top = pd.read_csv(T / 'top15_amplification_months_by_variant.csv')
diff_months = pd.read_csv(T / 'months_where_monthly_fallback_differs_from_event_manual.csv')
basis_energy = pd.read_csv(T / 'basis_energy_all_variants.csv')

# Abbreviated display columns. Keep PDF tables narrow enough to avoid clipping.
short_map = {
    'MP_median with fallback': 'median fallback',
    'MP_pm only': 'MP_pm',
    'Event-level shocks aggregated manually': 'event manual',
}
summary_display = summary[['label','top1_trace_share','top3_trace_share','top5_trace_share','tau_sd','tau_max','tau_max_month','state_spectral_radius']].copy()
summary_display['variant'] = summary_display['label'].map(short_map).fillna(summary_display['label'])
summary_pdf = summary_display[['variant','top1_trace_share','top3_trace_share','top5_trace_share','tau_sd','tau_max','tau_max_month','state_spectral_radius']].rename(columns={
    'top1_trace_share':'top1', 'top3_trace_share':'top3', 'top5_trace_share':'top5',
    'tau_max_month':'max_month', 'state_spectral_radius':'radius'
})
comp_display = comp[['label','top5_trace_share','max_principal_angle_degrees','tau_path_corr_with_baseline','top10_overlap_with_baseline','march_2020_tau','march_2020_rank','A44_diag_corr_with_baseline','A55_diag_corr_with_baseline']].copy()
comp_display['variant'] = comp_display['label'].map(short_map).fillna(comp_display['label'])
comp_pdf = comp_display[['variant','top5_trace_share','max_principal_angle_degrees','tau_path_corr_with_baseline','top10_overlap_with_baseline','march_2020_tau','march_2020_rank','A44_diag_corr_with_baseline','A55_diag_corr_with_baseline']].rename(columns={
    'top5_trace_share':'top5', 'max_principal_angle_degrees':'max_angle',
    'tau_path_corr_with_baseline':'tau_corr', 'top10_overlap_with_baseline':'top10',
    'march_2020_tau':'Mar2020_tau', 'march_2020_rank':'Mar_rank',
    'A44_diag_corr_with_baseline':'A44_corr', 'A55_diag_corr_with_baseline':'A55_corr'
})
diag_display = diag[['label','baseline_basis','matched_variant_basis','basis_vector_abs_corr','A_diag_path_corr']].copy()
diag_display['variant'] = diag_display['label'].map(short_map).fillna(diag_display['label'])
diag_pdf = diag_display[['variant','baseline_basis','matched_variant_basis','basis_vector_abs_corr','A_diag_path_corr']].rename(columns={
    'baseline_basis':'basis', 'matched_variant_basis':'match',
    'basis_vector_abs_corr':'vec_corr', 'A_diag_path_corr':'diag_corr'
})
top_display = top[['label','rank','date','tau','A11','A22','A33','A44','A55']].copy()
top_display['variant'] = top_display['label'].map(short_map).fillna(top_display['label'])
top_pdf = top_display.groupby('variant', sort=False).head(5)[['variant','rank','date','tau','A11','A22','A33','A44','A55']]

# Robustness interpretation from actual numbers.
base = comp[comp['variant'] == 'median_fallback'].iloc[0]
pm = comp[comp['variant'] == 'mp_pm_only'].iloc[0]
ev = comp[comp['variant'] == 'event_manual'].iloc[0]
march_rows = comp[['label', 'march_2020_tau', 'march_2020_rank']].copy()
march_summary = '; '.join(
    f"{row['label']}: rank {int(row['march_2020_rank']) if pd.notna(row['march_2020_rank']) else 'NA'}, tau {row['march_2020_tau']:.3f}"
    for _, row in march_rows.iterrows()
)
top_month_summary = '; '.join(
    f"{row['label']}: {row['tau_max_month']}"
    for _, row in summary.iterrows()
)

agenda_text = """
Robustness agenda remains essential. The current baseline uses MP_median with fallback. For a serious draft, rerun MP_median with fallback, MP_pm only, and event-level shocks aggregated manually. The correct robustness comparison is not only whether the average IRF survives. It should compare the leading five-dimensional subspace, the top-five trace share, the top amplification months, the tau_t path, and the basis-specific A_t diagonal paths. A stable result would show similar top-five subspace geometry and repeated crisis amplification across shock definitions. A fragile result would show that basis 4, basis 5, or the March 2020 spike depends on the fallback rule.
""".strip()

executive_text = f"""
This appended pack implements the requested robustness comparison rather than leaving it as a prose-only agenda. I reran the same rank-five log-Euclidean state-space A_t pipeline under three shock definitions: MP_median with fallback, MP_pm only, and manually aggregated event-level shocks.

The low-rank conclusion is stable. The top-five trace share is {base['top5_trace_share']:.3f} in the baseline, {pm['top5_trace_share']:.3f} under MP_pm only, and {ev['top5_trace_share']:.3f} under event-level manual aggregation.

The subspace and dynamic-path diagnostics are more discriminating. Event-level manual aggregation is very close to the baseline, with max principal angle {ev['max_principal_angle_degrees']:.1f} degrees and tau_t path correlation {ev['tau_path_corr_with_baseline']:.3f}. MP_pm-only is still low-rank, but less geometrically stable: max principal angle {pm['max_principal_angle_degrees']:.1f} degrees, tau_t correlation {pm['tau_path_corr_with_baseline']:.3f}, and basis-5 diagonal correlation only {pm['A55_diag_corr_with_baseline']:.3f}.

March 2020 is reported explicitly as a fallback-sensitive stress-test month rather than assumed to be the top amplification month. In the upgraded run, March 2020 diagnostics are: {march_summary}. The top amplification months by variant are: {top_month_summary}.
""".strip()

econ_text = """
The economic message is now sharper. The top-five subspace is robust enough to treat as a stable low-dimensional response-score geometry. That means the average covariance structure of monthly monetary-policy response surfaces is not diffuse across all horizons and variables. It is concentrated in a small propagation space.

The fragile part is the interpretation of the lower-ranked bases and crisis-specific allocation of amplification. Basis 1, basis 2, and basis 3 are relatively stable across definitions. Basis 4 is moderately sensitive. Basis 5 is highly sensitive under MP_pm-only. That means bases 4 and 5 should be described as lower-ranked covariance rotations, not named as invariant structural channels.

The tau_t comparison is the main crisis-amplification diagnostic. Total amplification remains high around 2007-2008 and 2020 under all definitions. That suggests crisis amplification is not simply an artifact of the baseline fallback rule. However, the exact diagonal decomposition of A_t changes across definitions. The paper should therefore emphasize robust subspace-level amplification before making basis-specific economic claims.

The event-level manual aggregation is especially informative because it isolates mixed-event months. The monthly baseline and event-manual definitions differ in January 2008, October 2008, and March 2020. These are precisely periods where shock construction and macro-financial transmission are both unstable. The manual event aggregation shows that the baseline is not wildly off, but the differences are large enough to require transparent reporting.
""".strip()

# ReportLab helpers.
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=12))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontSize=14, leading=16, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontSize=11, leading=13, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontSize=9.2, leading=11.5, spaceAfter=5))
styles.add(ParagraphStyle(name='Captionx', parent=styles['BodyText'], fontSize=7.5, leading=8.5, alignment=TA_CENTER, spaceAfter=6))


def P(text, style='Bodyx'):
    return Paragraph(str(text).replace('\n', '<br/>'), styles[style])


def add_text(story, text):
    for para in str(text).split('\n\n'):
        if para.strip():
            story.append(P(para.strip()))
            story.append(Spacer(1, 0.04 * inch))


PDF_TABLE_WIDTH = 7.15 * inch


def _pdf_table_cell(value, header=False):
    if isinstance(value, float):
        text = '' if pd.isna(value) else f'{value:.3f}'
    else:
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        text = '' if is_missing else str(value)
    text = html.escape(text)
    if header:
        text = text.replace('_', '_<br/>')
    else:
        text = text.replace(', ', ',<br/>').replace('; ', ';<br/>')
    style = ParagraphStyle(
        'PdfTableHeader' if header else 'PdfTableCell',
        fontName='Helvetica-Bold' if header else 'Helvetica',
        fontSize=5.7 if header else 5.4,
        leading=6.4 if header else 6.1,
        wordWrap='CJK',
    )
    return Paragraph(text, style)


def _pdf_table_widths(columns):
    weights = []
    for col in columns:
        name = str(col).lower()
        if any(token in name for token in ['variant', 'label', 'source', 'outcome']):
            weights.append(1.45)
        elif any(token in name for token in ['month', 'date', 'basis']):
            weights.append(1.05)
        else:
            weights.append(0.85)
    scale = PDF_TABLE_WIDTH / max(sum(weights), 1.0)
    return [w * scale for w in weights]


def make_table(df, max_rows=12, cols=None, max_cols=9):
    d = df.copy()
    if cols is not None:
        keep = [c for c in cols if c in d.columns]
        d = d[keep].copy() if keep else d.iloc[:, :max_cols].copy()
    elif len(d.columns) > max_cols:
        d = d.iloc[:, :max_cols].copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    if d.empty and len(d.columns) == 0:
        d = pd.DataFrame({'note': ['No rows available.']})
    data = [[_pdf_table_cell(c, header=True) for c in d.columns]]
    for _, row in d.iterrows():
        data.append([_pdf_table_cell(v) for v in row.tolist()])
    tbl = Table(data, repeatRows=1, colWidths=_pdf_table_widths(list(d.columns)), hAlign='LEFT', splitByRow=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 1.3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1.3),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
    ]))
    return tbl


def img_flow(path, max_w=6.9 * inch, max_h=4.4 * inch):
    im = PILImage.open(path)
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)

CH = ROBUST_OUT / 'charts'
story = []
story.append(P('Appendix: shock-definition robustness for top-five OVK', 'TitleCenter'))
story.append(P('Appended to the rank-five monthly monetary-policy OVK state-space report.', 'Bodyx'))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
story.append(Spacer(1, 0.08 * inch))
story.append(P('Requested robustness agenda', 'H1x'))
add_text(story, agenda_text)
story.append(P('Executive robustness results', 'H1x'))
add_text(story, executive_text)
story.append(PageBreak())

story.append(P('Variant summary', 'H1x'))
story.append(make_table(summary_pdf, max_rows=5))
story.append(Spacer(1, 0.08 * inch))
story.append(img_flow(CH / 'robustness_top5_trace_share_by_variant.png', max_h=3.7 * inch))
story.append(P('Figure A1. Top-five trace share is stable across shock definitions.', 'Captionx'))
story.append(PageBreak())

story.append(P('Leading five-dimensional subspace comparison', 'H1x'))
story.append(make_table(comp_pdf, max_rows=5))
story.append(Spacer(1, 0.08 * inch))
story.append(img_flow(CH / 'robustness_principal_angles_vs_baseline.png', max_h=3.7 * inch))
story.append(P('Figure A2. Principal angles versus the baseline top-five subspace.', 'Captionx'))
story.append(PageBreak())

story.append(P('Total amplification path', 'H1x'))
add_text(story, f'The tau_t path is the direct comparison of crisis amplification. The upgraded estimator reports March 2020 explicitly rather than assuming it is always the top month: {march_summary}. The top amplification months by variant are: {top_month_summary}.')
story.append(img_flow(CH / 'robustness_tau_paths_by_variant.png', max_h=4.3 * inch))
story.append(P('Figure A3. tau_t = trace(A_t)/5 under each shock definition.', 'Captionx'))
story.append(P('Top amplification months', 'H2x'))
story.append(make_table(top_pdf, max_rows=18))
story.append(PageBreak())

story.append(P('Basis-specific A_t diagonal path correlations', 'H1x'))
add_text(story, 'The basis-specific diagonal paths are the most demanding diagnostic. A stable total tau_t path can still hide reallocation across bases. The weak MP_pm-only basis-5 correlation is the clearest fragility result.')
story.append(make_table(diag_pdf, max_rows=15))
story.append(PageBreak())

for j in range(1, 6):
    story.append(P(f'Basis {j}: A_t diagonal robustness', 'H1x'))
    story.append(img_flow(CH / f'robustness_A{j}{j}_path_by_variant.png', max_h=4.6 * inch))
    story.append(P(f'Figure A{3+j}. Matched diagonal path for baseline basis {j}.', 'Captionx'))
    if j in [2, 4]:
        story.append(PageBreak())
story.append(PageBreak())

story.append(P('Monthly fallback versus event-level manual aggregation', 'H1x'))
add_text(story, 'The baseline monthly fallback and manual event-level aggregation differ only in mixed months where one event has missing median shocks and another event in the same month has nonmissing median shocks. Those months are January 2008, October 2008, and March 2020.')
story.append(make_table(diff_months, max_rows=5))
story.append(Spacer(1, 0.08 * inch))
story.append(img_flow(CH / 'shock_definition_monthly_fallback_vs_event_manual_differences.png', max_h=3.8 * inch))
story.append(P('Figure A9. Mixed-event months where monthly fallback and event-level manual aggregation differ.', 'Captionx'))
story.append(PageBreak())

story.append(P('Economic interpretation of the robustness results', 'H1x'))
add_text(story, econ_text)
story.append(P('Data and code included', 'H1x'))
add_text(story, 'The ZIP bundle contains the raw uploaded data.zip, extracted raw CSV files, repaired monthly panels, manually aggregated event-level shocks, all tables, all charts, the original top-five baseline report outputs, and the exact Python scripts used to reproduce the baseline and appended robustness results.')

SimpleDocTemplate(str(APPENDIX_PDF), pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch).build(story)

# Merge previous top-five baseline PDF with appendix.
writer = PdfWriter()
for src in [BASELINE_PDF, APPENDIX_PDF]:
    reader = PdfReader(str(src))
    for page in reader.pages:
        writer.add_page(page)
with open(FULL_PDF, 'wb') as f:
    writer.write(f)
shutil.copy2(FULL_PDF, REPORTS / 'monthly_ovk_top5_full_appended_report.pdf')
# APPENDIX_PDF is already inside REPORTS.

# Build a compact HTML report that points to both sections and embeds robustness charts.
def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def df_html(df, max_rows=20):
    d = df.head(max_rows).copy()
    rows = ['<tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in d.columns) + '</tr>']
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f'<td>{v:.3f}</td>')
            else:
                cells.append(f'<td>{html.escape(str(v))}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '<table>' + '\n'.join(rows) + '</table>'

imgs = ['robustness_top5_trace_share_by_variant.png','robustness_principal_angles_vs_baseline.png','robustness_tau_paths_by_variant.png'] + [f'robustness_A{j}{j}_path_by_variant.png' for j in range(1, 6)]
fig_html = ''.join(f'<h3>{name}</h3><img src="data:image/png;base64,{b64(CH / name)}" />' for name in imgs)
html_text = f"""<!doctype html><html><head><meta charset='utf-8'><title>Top-five OVK full appended report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;line-height:1.42;color:#222;}} h1{{font-size:24px;}} h2{{margin-top:30px;border-bottom:1px solid #999;padding-bottom:4px;}} table{{border-collapse:collapse;font-size:12px;margin:12px 0 20px 0;}} th,td{{border:1px solid #bbb;padding:5px 7px;vertical-align:top;}} th{{background:#eee;}} img{{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 18px 0;}}</style></head><body>
<h1>Monthly monetary-policy OVK: top-five full appended pack</h1>
<h2>Requested robustness agenda</h2><p>{html.escape(agenda_text)}</p>
<h2>Executive robustness results</h2><p>{html.escape(executive_text).replace(chr(10), '<br/>')}</p>
<h2>Variant summary</h2>{df_html(summary_display, 5)}
<h2>Subspace and dynamic-path comparison</h2>{df_html(comp_display, 5)}
<h2>Basis-specific diagonal path correlations</h2>{df_html(diag_display, 20)}
<h2>Charts</h2>{fig_html}
<h2>Economic interpretation</h2><p>{html.escape(econ_text).replace(chr(10), '<br/>')}</p>
</body></html>"""
FULL_HTML.write_text(html_text, encoding='utf-8')
shutil.copy2(FULL_HTML, REPORTS / 'monthly_ovk_top5_full_appended_report.html')

# README and manifest.
readme = f"""Monthly OVK top-five full appended results pack

Main files:
- reports/monthly_ovk_top5_full_appended_report.pdf
- reports/robustness_appendix.pdf
- reports/top5_baseline_state_space_report.pdf
- reports/monthly_ovk_top5_full_appended_report.html

Raw data:
- data_raw/data.zip
- data_raw/extracted/data/*.csv

Processed data:
- data_processed/ovk_monetary_panel_monthly_fixed_full.csv
- data_processed/processed_panel_three_shock_definitions.csv
- data_processed/event_level_manual_monthly_aggregation.csv

Code:
- code/generate_top5_ovk_pack.py
- code/finalize_top5_from_tables.py
- code/run_top5_robustness_point_estimates.py
- code/make_final_appended_top5_pack.py

Robustness headline:
- Baseline top-five trace share: {base['top5_trace_share']:.3f}
- MP_pm only top-five trace share: {pm['top5_trace_share']:.3f}
- Event-level manual top-five trace share: {ev['top5_trace_share']:.3f}
- March 2020 diagnostics by variant: {march_summary}.
- Basis 5 is fragile under MP_pm only; its matched A_t diagonal path correlation is {pm['A55_diag_corr_with_baseline']:.3f}.
"""
(PACK / 'README.md').write_text(readme, encoding='utf-8')
manifest = []
for f in sorted(PACK.rglob('*')):
    if f.is_file():
        manifest.append({'relative_path': str(f.relative_to(PACK)), 'bytes': f.stat().st_size})
pd.DataFrame(manifest).to_csv(PACK / 'file_manifest.csv', index=False)

# Zip full pack.
if FULL_ZIP.exists():
    FULL_ZIP.unlink()
with zipfile.ZipFile(FULL_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(PACK.rglob('*')):
        if f.is_file():
            z.write(f, arcname=f.relative_to(PACK))
    z.write(FULL_PDF, arcname='monthly_ovk_top5_full_appended_report.pdf')
    z.write(FULL_HTML, arcname='monthly_ovk_top5_full_appended_report.html')

print('Created', FULL_PDF)
print('Created', FULL_HTML)
print('Created', FULL_ZIP)
