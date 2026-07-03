from pathlib import Path
import shutil, zipfile, pandas as pd
from pypdf import PdfReader, PdfWriter

base_pack = Path('/mnt/data/monthly_ovk_top5_full_appended_results_pack')
out_root = Path('/mnt/data/monthly_ovk_top5_full_math_results_pack')
math_dir = out_root/'math_appendix'
reports_dir = out_root/'reports'
reports_dir.mkdir(exist_ok=True)
base_report = Path('/mnt/data/monthly_ovk_top5_full_appended_report.pdf')
appendix_pdf = math_dir/'technical_math_appendix.pdf'
tex_path = math_dir/'technical_math_appendix.tex'

# Copy user-facing appendix files.
shutil.copy2(appendix_pdf, '/mnt/data/monthly_ovk_top5_math_technical_appendix.pdf')
shutil.copy2(tex_path, '/mnt/data/monthly_ovk_top5_math_technical_appendix.tex')

# Merge base report + appendix.
merged_pdf = Path('/mnt/data/monthly_ovk_top5_full_appended_math_report.pdf')
writer = PdfWriter()
for pdf in [base_report, appendix_pdf]:
    reader = PdfReader(str(pdf))
    for page in reader.pages:
        writer.add_page(page)
with open(merged_pdf,'wb') as f:
    writer.write(f)
shutil.copy2(merged_pdf, reports_dir/'monthly_ovk_top5_full_appended_math_report.pdf')
shutil.copy2(appendix_pdf, reports_dir/'technical_math_appendix.pdf')
shutil.copy2(tex_path, reports_dir/'technical_math_appendix.tex')

# Update README with math appendix section.
readme_path = out_root/'README.md'
old = readme_path.read_text() if readme_path.exists() else ''
if '## Mathematical Appendix Added' not in old:
    addition = '''

## Mathematical Appendix Added

This version appends a full technical mathematical appendix to the report and includes source files under `math_appendix/` and `reports/`.
The appendix covers:

- local-projection score derivation via Frisch-Waugh-Lovell residualization;
- finite-dimensional operator-valued kernel construction and positive-definiteness proofs;
- spectral finite-rank truncation and standardized factor-score identities;
- construction and interpretation of the dynamic PSD drift matrix `A_t`;
- strict-positive-definiteness of the shrinkage proxy `G_t`;
- log-Euclidean VAR(1) state-space model, stationarity proof, Kalman filter/smoother equations;
- reconstruction proof for `K_t = V Lambda^(1/2) A_t Lambda^(1/2) V'`;
- moving-block bootstrap and state-smoothing uncertainty bands;
- principal-angle, trace-share, tau-path, top-month, and basis-specific robustness metrics;
- code-to-math crosswalk and pseudocode.

Primary new files:

- `reports/monthly_ovk_top5_full_appended_math_report.pdf`
- `reports/technical_math_appendix.pdf`
- `reports/technical_math_appendix.tex`
- `math_appendix/technical_math_appendix.pdf`
- `math_appendix/technical_math_appendix.tex`
- `math_appendix/algorithm_math_to_code_crosswalk.csv`
- `math_appendix/algorithm_pseudocode.py`
'''
    readme_path.write_text(old + addition)

# Add create scripts to code folder for reproducibility.
code_dir=out_root/'code'
code_dir.mkdir(exist_ok=True)
for src in ['/mnt/data/create_math_appendix.py','/mnt/data/finalize_math_pack.py']:
    p=Path(src)
    if p.exists():
        shutil.copy2(p, code_dir/p.name)

# Build manifest.
manifest_rows=[]
for f in sorted(out_root.rglob('*')):
    if f.is_file():
        rel=f.relative_to(out_root)
        manifest_rows.append({'path':str(rel), 'bytes':f.stat().st_size})
manifest_df=pd.DataFrame(manifest_rows)
manifest_df.to_csv(out_root/'file_manifest.csv', index=False)
shutil.copy2(out_root/'file_manifest.csv','/mnt/data/monthly_ovk_top5_full_math_file_manifest.csv')

# Zip full pack.
zip_out = Path('/mnt/data/monthly_ovk_top5_full_math_results_pack.zip')
if zip_out.exists(): zip_out.unlink()
with zipfile.ZipFile(zip_out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out_root.rglob('*')):
        if f.is_file():
            z.write(f, arcname=f.relative_to(out_root))

print('appendix_pdf', appendix_pdf, appendix_pdf.stat().st_size)
print('merged_pdf', merged_pdf, merged_pdf.stat().st_size)
print('zip_out', zip_out, zip_out.stat().st_size)
print('manifest', out_root/'file_manifest.csv')
