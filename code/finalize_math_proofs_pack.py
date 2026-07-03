from pathlib import Path
import shutil, zipfile, os, csv
from pypdf import PdfReader, PdfWriter
from PIL import Image, ImageOps, ImageDraw

BASE=Path('/mnt/data')
PACK=BASE/'monthly_ovk_top5_full_math_results_pack'
OLD_REPORT=BASE/'monthly_ovk_top5_full_appended_report.pdf'
TECH_APPENDIX=PACK/'math_appendix'/'technical_math_appendix.pdf'
FINAL_PDF=BASE/'monthly_ovk_top5_full_appended_math_proofs_report.pdf'
FINAL_ZIP=BASE/'monthly_ovk_top5_full_appended_math_proofs_results_pack.zip'
FINAL_CONTACT=BASE/'monthly_ovk_top5_full_appended_math_proofs_contact_sheet.jpg'
RENDER_DIR=BASE/'math_proofs_final_render_check'

# Remove unnecessary LaTeX aux files from deliverable, keep tex/pdf/summary/crosswalk/pseudocode.
for ext in ['*.aux','*.log','*.out','*.toc']:
    for f in (PACK/'math_appendix').glob(ext):
        try: f.unlink()
        except FileNotFoundError: pass

# Combine report + technical appendix.
writer=PdfWriter()
for pdf in [OLD_REPORT, TECH_APPENDIX]:
    reader=PdfReader(str(pdf))
    for page in reader.pages:
        writer.add_page(page)
with open(FINAL_PDF,'wb') as f:
    writer.write(f)
reports=PACK/'reports'
reports.mkdir(exist_ok=True)
shutil.copy2(FINAL_PDF, reports/'monthly_ovk_top5_full_appended_math_proofs_report.pdf')
shutil.copy2(TECH_APPENDIX, reports/'technical_math_appendix.pdf')

# Update README.
readme=PACK/'README.md'
existing=readme.read_text() if readme.exists() else ''
block='''

## Mathematical derivations and proofs appendix

This final version includes a technical mathematical appendix that operationalizes the algorithm. It gives full notation, derivations, proof sketches, and implementation formulas for the residualized LP score surface, average operator-valued kernel, finite-rank top-five basis, whitened score factors, dynamic positive-definite K_t construction, log-Euclidean state-space A_t evolution, Kalman smoothing equations, uncertainty bands, and robustness diagnostics across shock definitions.

Primary added files:

```text
reports/monthly_ovk_top5_full_appended_math_proofs_report.pdf
reports/technical_math_appendix.pdf
math_appendix/technical_math_appendix.pdf
math_appendix/technical_math_appendix.tex
math_appendix/technical_math_appendix_summary.md
math_appendix/algorithm_math_to_code_crosswalk.csv
math_appendix/algorithm_pseudocode.py
```
'''
if '## Mathematical derivations and proofs appendix' not in existing:
    readme.write_text(existing + block)

# Update manifest.
manifest=[]
for f in sorted(PACK.rglob('*')):
    if f.is_file() and f.exists():
        rel=f.relative_to(PACK)
        manifest.append({'relative_path':str(rel), 'bytes':f.stat().st_size, 'category':str(rel).split(os.sep)[0]})
with open(PACK/'file_manifest.csv','w',newline='',encoding='utf-8') as fh:
    w=csv.DictWriter(fh, fieldnames=['relative_path','bytes','category'])
    w.writeheader(); w.writerows(manifest)
shutil.copy2(PACK/'file_manifest.csv', BASE/'monthly_ovk_top5_full_appended_math_proofs_file_manifest.csv')

# Render final PDF and contact sheet.
if RENDER_DIR.exists(): shutil.rmtree(RENDER_DIR)
RENDER_DIR.mkdir(parents=True)
os.system(f'python /home/oai/skills/pdfs/scripts/render_pdf.py {FINAL_PDF} --out_dir {RENDER_DIR} --dpi 110 >/tmp/render_math_final.log 2>&1')
imgs=[]
for p in sorted(RENDER_DIR.glob('page-*.png')):
    try:
        imgs.append(Image.open(p).convert('RGB'))
    except Exception:
        pass
if imgs:
    thumbs=[]
    for i,im in enumerate(imgs, start=1):
        th=ImageOps.contain(im,(190,250))
        canv=Image.new('RGB',(210,280),'white')
        canv.paste(th,((210-th.width)//2,24))
        d=ImageDraw.Draw(canv); d.text((8,6),f'p. {i}',fill='black')
        thumbs.append(canv)
    cols=4; rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new('RGB',(cols*210, rows*280),'white')
    for i,th in enumerate(thumbs):
        sheet.paste(th,((i%cols)*210,(i//cols)*280))
    sheet.save(FINAL_CONTACT, quality=88)
    shutil.copy2(FINAL_CONTACT, reports/'monthly_ovk_top5_full_appended_math_proofs_contact_sheet.jpg')

# Rebuild manifest after contact sheet copied.
manifest=[]
for f in sorted(PACK.rglob('*')):
    if f.is_file() and f.exists():
        rel=f.relative_to(PACK)
        manifest.append({'relative_path':str(rel), 'bytes':f.stat().st_size, 'category':str(rel).split(os.sep)[0]})
with open(PACK/'file_manifest.csv','w',newline='',encoding='utf-8') as fh:
    w=csv.DictWriter(fh, fieldnames=['relative_path','bytes','category'])
    w.writeheader(); w.writerows(manifest)
shutil.copy2(PACK/'file_manifest.csv', BASE/'monthly_ovk_top5_full_appended_math_proofs_file_manifest.csv')

# Zip, robustly skipping broken or non-file paths.
if FINAL_ZIP.exists(): FINAL_ZIP.unlink()
with zipfile.ZipFile(FINAL_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(PACK.rglob('*')):
        if f.is_file() and f.exists():
            z.write(f, arcname=f.relative_to(PACK))

print('FINAL_PDF', FINAL_PDF, FINAL_PDF.stat().st_size, 'pages', len(PdfReader(str(FINAL_PDF)).pages))
print('TECH_APPENDIX', TECH_APPENDIX, TECH_APPENDIX.stat().st_size, 'pages', len(PdfReader(str(TECH_APPENDIX)).pages))
print('ZIP', FINAL_ZIP, FINAL_ZIP.stat().st_size)
print('CONTACT', FINAL_CONTACT, FINAL_CONTACT.stat().st_size if FINAL_CONTACT.exists() else 'missing')
print('MANIFEST', BASE/'monthly_ovk_top5_full_appended_math_proofs_file_manifest.csv')
