"""Reporting, charting, manifests, PDF merging, and ZIP packaging for SEC OVK."""
from __future__ import annotations

import hashlib
import html
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file SHA256 digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def savefig(fig: plt.Figure, path_no_suffix: Path) -> tuple[Path, Path]:
    """Save a matplotlib figure as PNG and SVG."""
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    png = path_no_suffix.with_suffix(".png")
    svg = path_no_suffix.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, dpi=170, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _safe_series(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(dtype=float)


def make_all_charts(results: dict[str, dict[str, Any]], tables: dict[str, pd.DataFrame], charts_dir: Path) -> dict[str, Path]:
    """Create the required SEC robustness charts."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, Path] = {}

    fig = plt.figure(figsize=(8.8, 5.2))
    for key, res in results.items():
        vals = np.asarray(res["geom"].eigenvalues[: res["selected_L"] + 1])
        plt.plot(np.arange(len(vals)), vals, marker="o", label=res["label"])
    plt.xlabel("Graph-Laplacian eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title("SEC graph-Laplacian spectra")
    plt.legend()
    chart_paths["spectrum"] = savefig(fig, charts_dir / "sec_laplacian_spectrum")[0]

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, max(4.2, 3.0 * n)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (_, res) in zip(axes, results.items()):
        dates = pd.to_datetime(res["dates"])
        Lplot = min(4, res["selected_L"])
        for j in range(1, Lplot + 1):
            ax.plot(dates, res["geom"].eigenfunctions[:, j], label=f"phi_{j}")
        ax.set_title(res["label"])
        ax.legend(loc="best")
    axes[-1].set_xlabel("Date")
    chart_paths["eigenfunctions"] = savefig(fig, charts_dir / "sec_eigenfunctions_over_time")[0]

    for key, res in results.items():
        dates = pd.to_datetime(res["dates"])
        base = res["baseline_path"].copy()
        base["date"] = pd.to_datetime(base["date"])
        merged = pd.DataFrame({"date": dates, "tau_sec": res["tau_sec"]}).merge(
            base[["date", "tau"]].rename(columns={"tau": "tau_publication_grade"}),
            on="date",
            how="inner",
        )
        fig = plt.figure(figsize=(10.5, 5.4))
        plt.plot(merged["date"], merged["tau_publication_grade"], label="publication-grade tau_t")
        plt.plot(merged["date"], merged["tau_sec"], label="SEC tau_t")
        bands = tables.get("sec_bootstrap_tau_bands", pd.DataFrame())
        b = bands[bands["variant"].eq(key)] if "variant" in bands.columns else pd.DataFrame()
        if not b.empty:
            b = b.copy()
            b["date"] = pd.to_datetime(b["date"])
            plt.fill_between(b["date"], b["tau_sec_p05"], b["tau_sec_p95"], alpha=0.18, label="SEC 90% block band")
        plt.axhline(1.0, linewidth=0.8)
        plt.ylabel("tau_t")
        plt.title(f"Publication-grade versus SEC tau_t: {res['label']}")
        plt.legend()
        chart_paths[f"tau_{key}"] = savefig(fig, charts_dir / f"sec_tau_overlay_{key}")[0]

    fig = plt.figure(figsize=(10.5, 5.4))
    for key, res in results.items():
        plt.plot(pd.to_datetime(res["dates"]), res["tau_sec"], label=res["label"])
    plt.axhline(1.0, linewidth=0.8)
    plt.ylabel("SEC tau_t")
    plt.title("SEC tau_t by shock definition")
    plt.legend()
    chart_paths["tau_all"] = savefig(fig, charts_dir / "sec_tau_all_shocks_overlay")[0]

    for key, res in results.items():
        dates = pd.to_datetime(res["dates"])
        base = res["baseline_path"].copy()
        base["date"] = pd.to_datetime(base["date"])
        for j in range(res["A_sec"].shape[1]):
            col = f"A{j+1}{j+1}"
            if col not in base.columns:
                continue
            merged = pd.DataFrame({"date": dates, "sec": res["A_sec"][:, j, j]}).merge(
                base[["date", col]].rename(columns={col: "publication_grade"}),
                on="date",
                how="inner",
            )
            fig = plt.figure(figsize=(10.5, 5.2))
            plt.plot(merged["date"], merged["publication_grade"], label="publication-grade")
            plt.plot(merged["date"], merged["sec"], label="SEC")
            plt.axhline(1.0, linewidth=0.8)
            plt.ylabel(col)
            plt.title(f"{col} path: {res['label']}")
            plt.legend()
            chart_paths[f"{key}_{col}"] = savefig(fig, charts_dir / f"sec_{key}_{col}_overlay")[0]

    for basis in [4, 5]:
        fig = plt.figure(figsize=(10.5, 5.4))
        for key, res in results.items():
            plt.plot(pd.to_datetime(res["dates"]), res["A_sec"][:, basis - 1, basis - 1], label=f"{res['label']} SEC")
        plt.axhline(1.0, linewidth=0.8)
        plt.ylabel(f"A{basis}{basis}")
        plt.title(f"SEC basis {basis} fragility across shock definitions")
        plt.legend()
        chart_paths[f"basis{basis}_fragility"] = savefig(fig, charts_dir / f"sec_basis{basis}_fragility")[0]

    top = tables.get("sec_top_amplification_months", pd.DataFrame()).copy()
    if not top.empty:
        baseline_top = top[top["source"].eq("publication_grade")].head(10)
        sec_top = top[top["source"].eq("SEC")].head(10)
        labels = list(dict.fromkeys(baseline_top["date"].tolist() + sec_top["date"].tolist()))[:15]
        fig = plt.figure(figsize=(11.0, 5.7))
        x = np.arange(len(labels))
        base_vals = [float(baseline_top.loc[baseline_top["date"].eq(m), "tau"].iloc[0]) if baseline_top["date"].eq(m).any() else np.nan for m in labels]
        sec_vals = [float(sec_top.loc[sec_top["date"].eq(m), "tau"].iloc[0]) if sec_top["date"].eq(m).any() else np.nan for m in labels]
        plt.bar(x - 0.18, base_vals, width=0.36, label="publication-grade")
        plt.bar(x + 0.18, sec_vals, width=0.36, label="SEC")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("tau_t")
        plt.title("Top amplification months: publication-grade versus SEC")
        plt.legend()
        chart_paths["top_months"] = savefig(fig, charts_dir / "sec_top_amplification_months")[0]

    base_key = "median_fallback" if "median_fallback" in results else next(iter(results))
    res = results[base_key]
    emb = res["embedding"]
    fig = plt.figure(figsize=(8.4, 6.0))
    scale = 30.0 + 120.0 * (res["tau_sec"] / max(float(np.max(res["tau_sec"])), 1e-12))
    plt.scatter(emb[:, 0], emb[:, 1], s=scale, alpha=0.65)
    dates = pd.to_datetime(res["dates"])
    labels = ["1998-10", "2007-09", "2020-03"]
    top_idx = np.argsort(res["tau_sec"])[::-1][:3]
    for idx in top_idx:
        plt.annotate(dates.iloc[idx].strftime("%Y-%m"), (emb[idx, 0], emb[idx, 1]))
    for month in labels:
        mask = dates.dt.strftime("%Y-%m").eq(month).to_numpy()
        if mask.any():
            idx = int(np.where(mask)[0][0])
            plt.annotate(month, (emb[idx, 0], emb[idx, 1]))
    plt.xlabel("phi_1")
    plt.ylabel("phi_2")
    plt.title("SEC macro-state manifold map by tau_t")
    chart_paths["manifold"] = savefig(fig, charts_dir / "sec_state_manifold_tau_map")[0]

    oos = tables.get("sec_oos_loss_comparison", pd.DataFrame())
    if not oos.empty:
        fig = plt.figure(figsize=(10.5, 5.4))
        xlabels = [f"{r.variant}\n{r.model}" for r in oos.itertuples()]
        plt.bar(np.arange(len(oos)), oos["loss"])
        plt.xticks(np.arange(len(oos)), xlabels, rotation=45, ha="right")
        plt.ylabel("Loss")
        plt.title("Out-of-sample loss comparison")
        chart_paths["oos"] = savefig(fig, charts_dir / "sec_oos_loss_comparison")[0]

    angles = tables.get("sec_principal_angles", pd.DataFrame())
    if not angles.empty:
        fig = plt.figure(figsize=(9.0, 5.2))
        for _, row in angles.iterrows():
            vals = [row.get(f"angle_{j}_degrees", np.nan) for j in range(1, 6)]
            plt.plot(np.arange(1, 6), vals, marker="o", label=row["label"])
        plt.xlabel("Principal angle index")
        plt.ylabel("Degrees")
        plt.title("Baseline versus SEC-filtered score-kernel subspace")
        plt.legend()
        chart_paths["angles"] = savefig(fig, charts_dir / "sec_principal_angles")[0]
    return chart_paths


PDF_TABLE_WIDTH = 7.1 * inch


def _reportlab_table_cell(value: object, *, header: bool = False) -> Paragraph:
    text = _format_cell(value)
    text = html.escape(text)
    if header:
        text = text.replace("_", "_<br/>")
    else:
        text = text.replace(", ", ",<br/>").replace("; ", ";<br/>")
    style = ParagraphStyle(
        "PdfTableHeader" if header else "PdfTableCell",
        fontName="Helvetica-Bold" if header else "Helvetica",
        fontSize=5.9 if header else 5.6,
        leading=6.6 if header else 6.3,
        wordWrap="CJK",
    )
    return Paragraph(text, style)


def _reportlab_table_widths(columns: list[object]) -> list[float]:
    weights = []
    for col in columns:
        name = str(col).lower()
        if any(token in name for token in ["variant", "label", "source", "model"]):
            weights.append(1.35)
        elif any(token in name for token in ["month", "date", "basis"]):
            weights.append(1.0)
        else:
            weights.append(0.85)
    scale = PDF_TABLE_WIDTH / max(sum(weights), 1.0)
    return [w * scale for w in weights]


def df_to_reportlab_table(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 6) -> Table:
    """Convert a small DataFrame slice to a reportlab table."""
    view = df.copy().head(max_rows)
    if len(view.columns) > max_cols:
        view = view.iloc[:, :max_cols]
    if view.empty and len(view.columns) == 0:
        view = pd.DataFrame({"note": ["No rows available."]})
    values = [[_reportlab_table_cell(c, header=True) for c in view.columns]]
    for _, row in view.iterrows():
        values.append([_reportlab_table_cell(v) for v in row.tolist()])
    table = Table(values, repeatRows=1, colWidths=_reportlab_table_widths(list(view.columns)), hAlign="LEFT", splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
            ]
        )
    )
    return table


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def build_sec_pdf(
    pdf_path: Path,
    tables: dict[str, pd.DataFrame],
    chart_paths: dict[str, Path],
    summary_text: list[str],
) -> None:
    """Build the standalone SEC robustness appendix PDF."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=16, leading=20))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    story: list[Any] = [
        Paragraph("SEC robustness: macro-state geometry and state-conditioned OVK", styles["TitleCenter"]),
        Spacer(1, 0.12 * inch),
    ]
    for text in summary_text:
        story.append(Paragraph(html.escape(text), styles["BodyText"]))
        story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph("Key comparison tables", styles["Heading1"]))
    for name in [
        "sec_top5_trace_share_comparison",
        "sec_principal_angles",
        "sec_tau_path_comparison",
        "sec_basis_diag_path_correlations",
        "sec_oos_loss_comparison",
    ]:
        df = tables.get(name, pd.DataFrame())
        if df.empty:
            continue
        story.append(Paragraph(name.replace("_", " "), styles["Heading2"]))
        story.append(df_to_reportlab_table(df))
        story.append(Spacer(1, 0.12 * inch))
    for key in ["tau_median_fallback", "tau_all", "basis4_fragility", "basis5_fragility", "manifold", "oos", "angles"]:
        path = chart_paths.get(key)
        if path and Path(path).exists():
            story.append(PageBreak())
            story.append(Paragraph(Path(path).stem.replace("_", " "), styles["Heading1"]))
            story.append(Image(str(path), width=6.7 * inch, height=3.8 * inch, kind="proportional"))
    doc.build(story)


def build_sec_html(
    html_path: Path,
    tables: dict[str, pd.DataFrame],
    chart_paths: dict[str, Path],
    summary_text: list[str],
) -> None:
    """Build the standalone SEC robustness appendix HTML."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SEC robustness appendix</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;max-width:1100px} table{border-collapse:collapse;font-size:12px} td,th{border:1px solid #ccc;padding:4px 6px} img{max-width:100%;height:auto}</style>",
        "</head><body>",
        "<h1>SEC robustness: macro-state geometry and state-conditioned OVK</h1>",
    ]
    for text in summary_text:
        parts.append(f"<p>{html.escape(text)}</p>")
    for name in [
        "sec_top5_trace_share_comparison",
        "sec_principal_angles",
        "sec_tau_path_comparison",
        "sec_basis_diag_path_correlations",
        "sec_oos_loss_comparison",
    ]:
        df = tables.get(name, pd.DataFrame())
        if not df.empty:
            parts.append(f"<h2>{html.escape(name.replace('_', ' '))}</h2>")
            parts.append(df.head(20).to_html(index=False, escape=True))
    for key, path in chart_paths.items():
        rel = os.path.relpath(path, html_path.parent).replace("\\", "/")
        parts.append(f"<h2>{html.escape(Path(path).stem.replace('_', ' '))}</h2><img src='{html.escape(rel)}'>")
    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")


def write_math_appendix(tex_path: Path, pdf_path: Path) -> None:
    """Write a SEC math appendix TeX file and a readable PDF rendering."""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    content = r"""
\section{SEC robustness appendix}
This appendix documents a final append-only robustness exercise. It does not
replace the publication-grade OVK estimator. It uses macro-state graph geometry
to regularize response-score covariance objects and compares the resulting
state-conditioned OVK to the publication-grade tau_t path.

\subsection{Graph Laplacian and eigenfunctions}
Let S_t be the predetermined macro-financial state vector built only from
information dated t-1 or earlier. A self-tuning k-nearest-neighbor graph has
weights W_{ij}=\exp(-\|S_i-S_j\|^2/(\sigma_i\sigma_j)). Density normalization
uses W_\alpha=D_q^{-\alpha}WD_q^{-\alpha}. The symmetric normalized Laplacian
is L_{sym}=I-D^{-1/2}W_\alpha D^{-1/2}. Diffusion eigenfunctions
\phi_j=D^{-1/2}u_j provide smooth scalar coordinates on the sampled macro-state
manifold.

\subsection{Local gradients and SEC directional features}
For each nontrivial eigenfunction, local linear regression on the k-neighborhood
approximates \phi_j(S_i)-\phi_j(S_t)\approx \nabla\phi_j(S_t)'(S_i-S_t). With
\Delta S_t=S_{t+1}-S_t, directional SEC features are
\xi_{ij,t}=\phi_i(S_t)\langle\nabla\phi_j(S_t),\Delta S_t\rangle.

\subsection{SEC-filtered score kernel}
Centered publication-grade LP score surfaces e_t are regressed on scalar SEC
features \Phi_t=[1,\phi_1(S_t),...,\phi_L(S_t)]. The fitted score surfaces
\hat e_t define K_{SEC,score}=T^{-1}\sum_t \hat e_t\hat e_t'. This matrix is
positive semidefinite because it is an average of outer products.

\subsection{SEC-conditioned A_t}
With publication-grade retained basis V_5 and eigenvalues Lambda_5, whitened
factors z_t produce SPD proxies G_t=\alpha I+(1-\alpha)z_tz_t'. After sample
mean normalization, y_t=svec(\log \tilde G_t) is regressed on \Phi_t. Fitted
values map back through A_{raw}(S_t)=\exp(smat(\hat y_t)). Defining
C=T^{-1}\sum_t A_{raw}(S_t), the normalized path is
A_{SEC}(S_t)=C^{-1/2}A_{raw}(S_t)C^{-1/2}. Thus every A_{SEC}(S_t) is positive
definite and T^{-1}\sum_t A_{SEC}(S_t)=I_5.

\subsection{Retained mean-kernel recovery}
The state-conditioned retained kernel is
K_{SEC,t}=V_5\Lambda_5^{1/2}A_{SEC}(S_t)\Lambda_5^{1/2}V_5'. Since
mean_t A_{SEC}(S_t)=I_5, the sample mean of K_{SEC,t} equals
V_5\Lambda_5V_5'. Principal angles compare the baseline and SEC-filtered
subspaces invariantly to sign changes and rotations.

\subsection{Interpretation and limits}
SEC estimates state-conditioned response-score covariance geometry, not a
standalone structural causal IRF. Lower-ranked basis-specific A_{jj,t} paths,
especially bases 4 and 5, can rotate or become fragile even when the leading
five-dimensional subspace remains stable. Limitations include finite sample
noise, state-vector sensitivity, graph-gradient approximation error, and
fallback-shock measurement uncertainty.
"""
    tex_path.write_text(content.strip() + "\n", encoding="utf-8")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=16))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    story = [
        Paragraph("SEC mathematical appendix", styles["TitleCenter"]),
        Spacer(1, 0.12 * inch),
    ]
    for block in content.split(r"\subsection"):
        clean = (
            block.replace(r"\section{SEC robustness appendix}", "SEC robustness appendix")
            .replace("{", "")
            .replace("}", "")
            .replace("\\", "")
            .replace("_", " ")
        )
        if clean.strip():
            story.append(Paragraph(html.escape(clean.strip()), styles["BodyText"]))
            story.append(Spacer(1, 0.10 * inch))
    doc.build(story)


def merge_pdfs(base_pdf: Path, appendix_pdf: Path, out_pdf: Path) -> int:
    """Merge an existing report PDF and SEC appendix PDF without modifying either source."""
    writer = PdfWriter()
    for src in [base_pdf, appendix_pdf]:
        reader = PdfReader(str(src))
        for page in reader.pages:
            writer.add_page(page)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as fh:
        writer.write(fh)
    return len(PdfReader(str(out_pdf)).pages)


def verify_pdf_readable(path: Path) -> int:
    """Verify that a PDF can be opened and has at least one page."""
    reader = PdfReader(str(path))
    n_pages = len(reader.pages)
    if n_pages < 1:
        raise ValueError(f"PDF has no pages: {path}")
    return n_pages


def create_contact_sheet(chart_paths: dict[str, Path], out_path: Path) -> None:
    """Create a JPEG contact sheet from representative SEC charts."""
    selected = [p for p in chart_paths.values() if Path(p).exists()][:9]
    if not selected:
        raise ValueError("No charts available for contact sheet")
    rows = int(np.ceil(len(selected) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, selected):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(Path(path).stem, fontsize=9)
        ax.axis("off")
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_manifest(root: Path, out_csv: Path, descriptions: dict[str, str] | None = None) -> pd.DataFrame:
    """Write a recursive file manifest for a result root."""
    descriptions = descriptions or {}
    rows = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        rel = file.relative_to(root).as_posix()
        stat = file.stat()
        rows.append(
            {
                "relative_path": rel,
                "file_size_bytes": int(stat.st_size),
                "modified_time": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
                "sha256": file_sha256(file),
                "description": descriptions.get(rel, ""),
            }
        )
    manifest = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)
    return manifest


def create_final_zip(repo_root: Path, sec_root: Path, out_zip: Path, extra_files: list[Path]) -> None:
    """Create the final SEC robustness ZIP pack."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    include_dirs = [
        repo_root / "data_raw",
        repo_root / "data_processed",
        repo_root / "code",
        repo_root / "tests",
        repo_root / "results" / "top5_full_appended_results_pack",
        repo_root / "results" / "reports",
        sec_root,
    ]
    seen: set[Path] = set()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for directory in include_dirs:
            if not directory.exists():
                continue
            for file in sorted(directory.rglob("*")):
                if not file.is_file():
                    continue
                if file.resolve() == out_zip.resolve():
                    continue
                arc = file.relative_to(repo_root).as_posix()
                if file.resolve() in seen:
                    continue
                seen.add(file.resolve())
                zf.write(file, arcname=arc)
        for file in extra_files:
            if file.exists() and file.resolve() not in seen:
                zf.write(file, arcname=file.relative_to(repo_root).as_posix())


def copy_sec_code(repo_root: Path, sec_code_dir: Path) -> None:
    """Copy SEC source modules into the SEC results code directory."""
    sec_code_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_sec_robustness.py",
        "sec_geometry.py",
        "sec_ovk.py",
        "sec_comparisons.py",
        "sec_reporting.py",
    ]:
        src = repo_root / "code" / name
        if src.exists():
            shutil.copy2(src, sec_code_dir / name)
