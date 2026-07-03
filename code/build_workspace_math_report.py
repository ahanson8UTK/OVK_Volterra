from __future__ import annotations

import html
import json
import math
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUT_DIR / "ovk_workspace_math_report.pdf"
REPORT_DATE = datetime.now().strftime("%B %d, %Y")


def rel(path: str | Path) -> Path:
    return ROOT / Path(path)


def read_csv(path: str | Path) -> pd.DataFrame:
    p = rel(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def read_json(path: str | Path) -> dict[str, Any]:
    p = rel(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def first_row(df: pd.DataFrame, **filters: Any) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns:
            return None
        mask &= df[key].astype(str).eq(str(value))
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def fmt_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (float,)):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.{digits}f}"
    text = str(value)
    try:
        numeric = float(text)
        if math.isfinite(numeric) and text.strip() not in {"", "True", "False"}:
            return fmt_value(numeric, digits=digits)
    except ValueError:
        pass
    return text


def shorten(text: Any, max_chars: int = 82) -> str:
    s = fmt_value(text)
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "..."


def escape(text: Any) -> str:
    return html.escape(shorten(text, 140))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=14,
            textColor=colors.HexColor("#1f2933"),
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=14,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#111827"),
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=9,
            spaceAfter=5,
            textColor=colors.HexColor("#1f2937"),
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=12.6,
            spaceAfter=5,
            textColor=colors.HexColor("#1f2937"),
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.2,
            spaceAfter=4,
            textColor=colors.HexColor("#374151"),
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.6,
            leading=9.5,
            spaceBefore=2,
            spaceAfter=7,
            textColor=colors.HexColor("#4b5563"),
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.1,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.7,
            leading=8,
            alignment=TA_LEFT,
            textColor=colors.white,
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.3,
            leading=8.8,
            leftIndent=0,
            rightIndent=0,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f7f7f2"),
            borderColor=colors.HexColor("#d9d6c8"),
            borderWidth=0.4,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=7,
        ),
    }


STYLES = make_styles()


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(text: str, level: int = 1) -> Paragraph:
    return P(escape(text), "H1" if level == 1 else "H2")


def formula(text: str) -> Preformatted:
    clean = textwrap.dedent(text).strip("\n")
    clean = "\n".join(line.rstrip() for line in clean.splitlines())
    return Preformatted(clean, STYLES["Code"])


def df_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    rename: dict[str, str] | None = None,
    max_rows: int = 12,
    max_chars: int = 64,
    col_widths: list[float] | None = None,
) -> Table:
    if df.empty:
        data = [[Paragraph("No table found in this workspace.", STYLES["TableCell"])]]
        return Table(data, colWidths=[7.0 * inch])
    work = df.copy()
    if columns:
        work = work[[c for c in columns if c in work.columns]]
    if max_rows and len(work) > max_rows:
        work = work.head(max_rows)
    if rename:
        work = work.rename(columns={k: v for k, v in rename.items() if k in work.columns})
    headers = [Paragraph(escape(c), STYLES["TableHead"]) for c in work.columns]
    rows = []
    for _, row in work.iterrows():
        rows.append(
            [
                Paragraph(html.escape(shorten(row[c], max_chars=max_chars)), STYLES["TableCell"])
                for c in work.columns
            ]
        )
    data = [headers] + rows
    if col_widths is None:
        n = max(len(work.columns), 1)
        col_widths = [7.0 * inch / n] * n
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def key_value_table(rows: list[tuple[str, Any]], key_width: float = 2.55 * inch) -> Table:
    df = pd.DataFrame(rows, columns=["Quantity", "Value"])
    return df_table(
        df,
        max_rows=len(df),
        max_chars=105,
        col_widths=[key_width, 7.0 * inch - key_width],
    )


def add_image(story: list[Any], path: str | Path, caption: str, max_width: float = 6.8 * inch, max_height: float = 3.65 * inch) -> None:
    p = rel(path)
    if not p.exists():
        return
    with PILImage.open(p) as img:
        w, h = img.size
    scale = min(max_width / float(w), max_height / float(h))
    story.append(Image(str(p), width=w * scale, height=h * scale))
    story.append(P(escape(caption), "Caption"))


def page_header(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, letter[1] - 0.42 * inch, "OVK Workspace Math Report")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def source_inventory() -> pd.DataFrame:
    rows = [
        {
            "Source": "code/run_publication_grade_ovk.py",
            "Math role": "LP score construction; average OVK; eigensystem; robust log-Euclidean state-space; FFBS and full-pipeline bootstrap.",
        },
        {
            "Source": "code/ovk_nested_workflow.py",
            "Math role": "Nested M0-M3 mean/covariance comparison; moving mean gamma_t filter; A_t survival after mean adjustment.",
        },
        {
            "Source": "code/run_top5_robustness_point_estimates.py",
            "Math role": "Shock-definition robustness surfaces, now fed by the upgraded publication-grade estimator.",
        },
        {
            "Source": "code/sec_geometry.py, code/sec_ovk.py, code/run_sec_robustness.py",
            "Math role": "Graph Laplacian macro-state geometry, SEC features, SEC-conditioned A_t, and comparison diagnostics.",
        },
        {
            "Source": "code/iv_ovk.py",
            "Math role": "Scalar proxy-IV LP influence scores, IV OVK eigensystem, IV A_t path, score-energy decomposition, and IV bootstrap.",
        },
        {
            "Source": "code/create_full_math_appendix_pack.py",
            "Math role": "Earlier standalone math appendix generator for the top-five OVK derivations.",
        },
        {
            "Source": "sec_robustness_results/math_appendix/sec_math_appendix.tex",
            "Math role": "Concise SEC mathematical appendix source.",
        },
        {
            "Source": "results/reports/*.pdf and monthly_ovk_top5_with_SEC_robustness_report.pdf",
            "Math role": "Previously generated narrative reports and appendices assembled from the tables and charts.",
        },
    ]
    return pd.DataFrame(rows)


def build_report() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pub_rank = read_csv("results/publication_grade_ovk/outputs/tables/publication_grade_rank_summary.csv")
    pub_unc = read_csv("results/publication_grade_ovk/outputs/tables/publication_grade_uncertainty_summary.csv")
    pub_boot = read_csv("results/publication_grade_ovk/outputs/tables/publication_grade_full_pipeline_bootstrap_summary.csv")
    same_sample = read_csv("results/publication_grade_ovk/outputs/tables/same_sample_outcome_comparison.csv")
    policy_split = read_csv("results/publication_grade_ovk/outputs/tables/policy_cbi_split_comparison.csv")
    placebo = read_csv("results/publication_grade_ovk/outputs/tables/placebo_shock_comparison.csv")
    variant_registry = read_csv("results/publication_grade_ovk/outputs/tables/publication_grade_variant_registry.csv")

    nested_overview = read_csv("results/nested_mean_cov_robustness/tables/nested_mean_covariance_variant_summary.csv")
    nested_comp = read_csv("results/nested_mean_cov_robustness/tables/model_comparisons_block_bootstrap_by_variant.csv")
    nested_survival = read_csv("results/nested_mean_cov_robustness/tables/A_t_survival_summary_by_variant.csv")
    nested_meta = read_json("results/nested_mean_cov_robustness/metadata.json")

    shock_summary = read_csv("results/top5_shock_robustness/outputs/tables/robustness_variant_summary.csv")
    shock_metrics = read_csv("results/top5_shock_robustness/outputs/tables/robustness_comparison_metrics.csv")

    sec_trace = read_csv("sec_robustness_results/tables/sec_top5_trace_share_comparison.csv")
    sec_tau = read_csv("sec_robustness_results/tables/sec_tau_path_comparison.csv")
    sec_angles = read_csv("sec_robustness_results/tables/sec_principal_angles.csv")
    sec_oos = read_csv("sec_robustness_results/tables/sec_oos_loss_comparison.csv")

    iv_sample = read_csv("results/iv_ovk/tables/iv_lp_sample_summary.csv")
    iv_first = read_csv("results/iv_ovk/tables/iv_first_stage_summary.csv")
    iv_rank = read_csv("results/iv_ovk/tables/iv_rank_summary.csv")
    iv_base = read_csv("results/iv_ovk/tables/iv_vs_baseline_tau_comparison.csv")
    iv_decomp = read_csv("results/iv_ovk/tables/iv_score_decomposition_summary.csv")
    iv_nested = read_csv("results/iv_ovk/tables/iv_nested_log_score_comparison.csv")

    headline = first_row(pub_rank, variant="base5_headline", rank=5)
    rank3 = first_row(pub_rank, variant="base5_headline", rank=3)
    rank7 = first_row(pub_rank, variant="base5_headline", rank=7)
    all8 = first_row(pub_rank, variant="all8_expectation_overlap", rank=5)
    iv_headline = first_row(iv_rank, rank=5)

    story: list[Any] = []
    story.append(P("All Workspace Math So Far", "Title"))
    story.append(
        P(
            f"Consolidated mathematical report for the OVK LP workspace. Generated locally from current result tables on {REPORT_DATE}.",
            "Subtitle",
        )
    )
    story.append(
        P(
            "Scope: this PDF consolidates the math encoded in the workspace scripts, tables, appendices, and reports. "
            "It is a math-first ledger of the work already done: local-projection score construction, operator-valued kernels, "
            "dynamic positive-definite A_t paths, uncertainty bands, nested mean/covariance checks, shock robustness, SEC state geometry, "
            "and the proxy-IV extension.",
            "Body",
        )
    )
    story.append(
        P(
            "Important interpretation: the central object is time-varying covariance geometry of LP response-score surfaces. "
            "The headline A_t and tau_t paths measure amplification of response-score covariance, not unrestricted time-varying structural IRFs.",
            "Body",
        )
    )

    quick_rows: list[tuple[str, Any]] = []
    if headline is not None:
        quick_rows.extend(
            [
                ("Headline sample", f"{headline.get('label')} ; n={fmt_value(headline.get('n_valid'))}, M=125"),
                ("Headline retained rank", f"R={fmt_value(headline.get('rank'))}, state dimension={fmt_value(headline.get('state_dim'))}"),
                ("Top-five trace share", headline.get("retained_trace_share")),
                ("Selected alpha", headline.get("alpha_hat")),
                ("Transition spectral radius", headline.get("transition_spectral_radius")),
                ("Max tau_t", f"{fmt_value(headline.get('tau_max'))} in {headline.get('tau_max_month')}"),
            ]
        )
    if not nested_overview.empty:
        row = first_row(nested_overview, variant="base5_headline")
        if row is not None:
            quick_rows.extend(
                [
                    ("Nested dynamic covariance advantage M1-M0", row.get("M1_minus_M0")),
                    ("Nested moving mean advantage M2-M0", row.get("M2_minus_M0")),
                    ("A_t survival correlation M1 vs M3", row.get("survival_corr_trace_M1_M3")),
                ]
            )
    if iv_headline is not None:
        quick_rows.append(("Proxy-IV max tau_t", f"{fmt_value(iv_headline.get('tau_max'))} in {iv_headline.get('tau_max_month')}"))
    story.append(key_value_table(quick_rows))
    story.append(Spacer(1, 0.14 * inch))
    story.append(heading("Source Map"))
    story.append(df_table(source_inventory(), max_rows=12, max_chars=98, col_widths=[2.45 * inch, 4.55 * inch]))

    story.append(PageBreak())
    story.append(heading("1. Common LP Score Surface"))
    story.append(
        P(
            "All main workflows begin by converting monetary-policy local projections into one-observation score surfaces. "
            "For p outcomes and horizons h=0,...,H, each observation is a stacked response vector Y_t in R^M with M=p(H+1). "
            "In the headline five-outcome run p=5, H=24, and M=125; in the expectations overlap run p=8 and M=200.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            Y_t = stack_{h=0..H} ( y_{t+h} - y_{t-1} )

            M_W = I - W (W'W)^+ W'
            m_tilde = M_W m
            Y_tilde = M_W Y
            sigma_m^2 = T^{-1} sum_t m_tilde_t^2

            q_t = m_tilde_t * Y_tilde_t / sigma_m^2
            beta_hat = T^{-1} sum_t q_t
            e_t = q_t - beta_hat
            """
        )
    )
    story.append(
        P(
            "The second line is the finite-sample Frisch-Waugh-Lovell residualization used by the code. "
            "Because sigma_m^2 is the sample second moment of the residualized shock, the sample mean of q_t is exactly the multivariate LP coefficient surface beta_hat.",
            "Body",
        )
    )
    story.append(heading("FWL Identity", 2))
    story.append(
        formula(
            r"""
            T^{-1} sum_t q_t
              = T^{-1} sum_t [m_tilde_t Y_tilde_t / sigma_m^2]
              = (m_tilde' Y_tilde) / (m_tilde' m_tilde)
              = beta_hat.
            """
        )
    )

    story.append(heading("2. Average Operator-Valued Kernel"))
    story.append(
        P(
            "The average OVK is the covariance matrix of centered score surfaces. "
            "Written by horizon blocks, K_bar(h,h') is a p by p matrix describing how score variation at horizon h co-moves with score variation at horizon h'.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            K_bar = T^{-1} E' E = T^{-1} sum_t e_t e_t'
            K_bar(h,h') = T^{-1} sum_t e_t(h) e_t(h')'

            For coefficients a_0,...,a_H in R^p:
            sum_{h,h'} a_h' K_bar(h,h') a_{h'}
              = T^{-1} sum_t [sum_h a_h' e_t(h)]^2 >= 0.
            """
        )
    )
    story.append(
        P(
            "This proves K_bar is a finite-sample positive semidefinite operator-valued kernel. "
            "Its trace equals the average squared norm of centered score surfaces, so eigenvalue trace shares are variance-geometry shares.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            K_bar = V Lambda V'
            K_bar,R = V_R Lambda_R V_R'
            trace_share_R = (lambda_1 + ... + lambda_R) / trace(K_bar)
            """
        )
    )
    if headline is not None:
        overview_rows = [
            ("R=3 trace share", rank3.get("retained_trace_share") if rank3 is not None else ""),
            ("R=5 trace share", headline.get("retained_trace_share")),
            ("R=7 trace share", rank7.get("retained_trace_share") if rank7 is not None else ""),
            ("R=5 eigengap R to R+1", headline.get("eigengap_R_to_Rplus1")),
            ("R=5 robust log likelihood", headline.get("robust_loglik")),
            ("R=5 factor log score", headline.get("factor_log_score")),
        ]
        story.append(key_value_table(overview_rows))
    add_image(
        story,
        "results/publication_grade_ovk/outputs/charts/06_rank_sensitivity.png",
        "Publication-grade rank sensitivity for retained response-score covariance geometry.",
    )

    story.append(PageBreak())
    story.append(heading("3. Dynamic Log-Euclidean A_t Model"))
    story.append(
        P(
            "After retaining the leading R-dimensional eigenspace, the code whitens the finite-rank score coordinates. "
            "The sample covariance of z_t is I_R, so a time-varying positive-definite matrix A_t can be read as local amplification relative to the average retained geometry.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            z_t = Lambda_R^{-1/2} V_R' e_t

            T^{-1} sum_t z_t z_t'
              = Lambda_R^{-1/2} V_R' K_bar V_R Lambda_R^{-1/2}
              = I_R.

            G_t = alpha I_R + (1-alpha) z_t z_t'       with alpha > 0
            H_t = G_bar^{-1/2} G_t G_bar^{-1/2}
            y_t^log = svec( log(H_t) )
            """
        )
    )
    story.append(
        P(
            "The shrinkage proxy G_t is strictly positive definite because u'G_tu >= alpha ||u||^2 for every nonzero u. "
            "That is the step that makes matrix logarithms well-defined even though z_t z_t' is rank one.",
            "Body",
        )
    )
    story.append(heading("State Equation and Reconstruction", 2))
    story.append(
        formula(
            r"""
            x_t = mu + F (x_{t-1} - mu) + eta_t
            y_t^log = x_t + eps_t

            A_raw,t = exp( smat(x_{t|T}) )
            C = T^{-1} sum_t A_raw,t
            A_t = C^{-1/2} A_raw,t C^{-1/2}

            K_t = V_R Lambda_R^{1/2} A_t Lambda_R^{1/2} V_R'
            tau_t = trace(A_t) / R
            S_t = A_t / tau_t
            """
        )
    )
    story.append(
        P(
            "The normalization gives sample mean(A_t)=I_R, hence sample mean(K_t)=K_bar,R. "
            "Student-t robust observation weights, structured transition shrinkage, EM-style F/Q/R estimation, FFBS state draws, and full-pipeline moving-block bootstrap draws are all implemented in the publication-grade script.",
            "Body",
        )
    )
    if not pub_unc.empty:
        story.append(df_table(pub_unc, max_rows=12, max_chars=75, col_widths=[3.15 * inch, 3.85 * inch]))
    add_image(
        story,
        "results/publication_grade_ovk/outputs/charts/01_publication_tau_ffbs_state_bands.png",
        "Headline rank-five tau_t with FFBS state uncertainty bands.",
    )
    add_image(
        story,
        "results/publication_grade_ovk/outputs/charts/02_publication_tau_full_pipeline_bands.png",
        "Headline rank-five tau_t with full-pipeline bootstrap uncertainty bands.",
    )

    story.append(PageBreak())
    story.append(heading("4. Publication-Grade Robustness Ledger"))
    story.append(
        P(
            "The publication-grade run treats the base-five rank-five estimator as the headline and then refits the score geometry under alternate samples, outcome sets, transformations, shock definitions, placebo shocks, policy/CBI splits, smooth LP surfaces, and SF Fed/Bauer-Swanson appendix shocks.",
            "Body",
        )
    )
    keep_variants = [
        "base5_headline",
        "all8_expectation_overlap",
        "base5_headline_standardized",
        "base5_mp_pm_only",
        "base5_event_manual",
        "policy_without_cbi",
        "cbi_with_policy",
        "placebo_permuted",
        "placebo_shift84",
        "base5_headline_smooth",
        "sf_fed_raw",
        "sf_fed_orthogonalized",
    ]
    if not pub_rank.empty:
        pub_rank5 = pub_rank[(pub_rank["rank"].astype(str) == "5") & pub_rank["variant"].isin(keep_variants)].copy()
        story.append(
            df_table(
                pub_rank5,
                columns=[
                    "variant",
                    "group",
                    "outcome_count",
                    "retained_trace_share",
                    "tau_sd",
                    "tau_max",
                    "tau_max_month",
                    "transition_spectral_radius",
                ],
                rename={
                    "retained_trace_share": "trace share",
                    "tau_sd": "tau sd",
                    "tau_max": "max tau",
                    "tau_max_month": "max month",
                    "transition_spectral_radius": "rho(F)",
                },
                max_rows=14,
                max_chars=48,
                col_widths=[1.55 * inch, 1.05 * inch, 0.55 * inch, 0.76 * inch, 0.62 * inch, 0.68 * inch, 0.76 * inch, 1.03 * inch],
            )
        )
    if not pub_boot.empty:
        selected_boot = pub_boot[pub_boot["variant"].isin(["base5_headline", "all8_expectation_overlap", "base5_mp_pm_only", "base5_event_manual", "placebo_permuted", "sf_fed_raw"])].copy()
        story.append(heading("Full-Pipeline Bootstrap Ranges", 2))
        story.append(
            df_table(
                selected_boot,
                columns=[
                    "variant",
                    "bootstrap_draws_valid",
                    "trace_share_p05",
                    "trace_share_p95",
                    "max_tau_p05",
                    "max_tau_p95",
                    "march_2020_top10_probability_full_pipeline",
                ],
                rename={
                    "bootstrap_draws_valid": "valid draws",
                    "trace_share_p05": "share p05",
                    "trace_share_p95": "share p95",
                    "max_tau_p05": "max tau p05",
                    "max_tau_p95": "max tau p95",
                    "march_2020_top10_probability_full_pipeline": "P(Mar20 top10)",
                },
                max_rows=8,
                max_chars=44,
            )
        )
    add_image(
        story,
        "results/publication_grade_ovk/outputs/charts/07_shock_construction_envelope.png",
        "Publication-grade shock-construction envelope for tau_t variants.",
    )
    add_image(
        story,
        "results/publication_grade_ovk/outputs/charts/08_student_t_weights.png",
        "Student-t robust observation weights used by the headline state-space fit.",
    )
    story.append(heading("Variant Registry", 2))
    if not variant_registry.empty:
        story.append(
            df_table(
                variant_registry[variant_registry["available"].astype(str).eq("True")],
                columns=["variant", "group", "outcome_columns", "shock_col", "control_col", "transform"],
                max_rows=16,
                max_chars=50,
                col_widths=[1.35 * inch, 0.95 * inch, 1.65 * inch, 1.15 * inch, 1.15 * inch, 0.75 * inch],
            )
        )

    story.append(PageBreak())
    story.append(heading("5. Mean Versus Covariance: Nested Models"))
    story.append(
        P(
            "The nested workflow tests whether the main A_t movement is really covariance movement, or whether it is mostly an omitted moving LP-response center. "
            "It evaluates one-step Gaussian quasi-log scores in a compact W-coordinate basis.",
            "Body",
        )
    )
    story.append(
        df_table(
            pd.DataFrame(
                [
                    ("M0", "fixed beta", "fixed K", "fixed mean and fixed covariance benchmark"),
                    ("M1", "fixed beta", "dynamic K_t", "dynamic covariance around fixed average response"),
                    ("M2", "beta + B_beta gamma_t", "fixed K", "moving response center only"),
                    ("M3", "beta + B_beta gamma_t", "dynamic K_t", "joint moving mean and moving covariance"),
                ],
                columns=["Model", "Mean", "Covariance", "Meaning"],
            ),
            max_rows=4,
            max_chars=70,
            col_widths=[0.55 * inch, 1.55 * inch, 1.25 * inch, 3.65 * inch],
        )
    )
    story.append(
        formula(
            r"""
            gamma_pred,t = phi gamma_{t-1}
            mean_pred,t = beta + B_beta gamma_pred,t
            gamma_obs,t = B_beta' (q_t - beta)
            gamma_t = (1-k_mean) gamma_pred,t + k_mean gamma_obs,t

            S_pred,t = (1-k_target) S_{t-1} + k_target S_target
            S_t = (1-k_cov) S_pred,t + k_cov e_t e_t'
            log_score_t = log N(q_t ; mean_pred,t, S_pred,t)
            """
        )
    )
    if not nested_overview.empty:
        story.append(
            df_table(
                nested_overview,
                columns=[
                    "variant",
                    "outcome_count",
                    "score_surface_dimension",
                    "M1_minus_M0",
                    "M2_minus_M0",
                    "M3_minus_M1",
                    "survival_corr_trace_M1_M3",
                    "top12_overlap",
                ],
                rename={
                    "score_surface_dimension": "M",
                    "M1_minus_M0": "M1-M0",
                    "M2_minus_M0": "M2-M0",
                    "M3_minus_M1": "M3-M1",
                    "survival_corr_trace_M1_M3": "A_t corr",
                    "top12_overlap": "top12 overlap",
                },
                max_rows=8,
                max_chars=44,
            )
        )
    if not nested_comp.empty:
        selected = nested_comp[nested_comp["comparison"].isin(["M1 - M0", "M2 - M0", "M3 - M1", "M3 - M2"])].copy()
        story.append(
            df_table(
                selected,
                columns=["variant", "comparison", "avg_log_score_diff", "p05", "p95", "prob_diff_gt_0"],
                rename={"avg_log_score_diff": "avg diff", "prob_diff_gt_0": "P(diff>0)"},
                max_rows=12,
                max_chars=42,
            )
        )
    story.append(
        P(
            "The base-five result is the clearest anchor: dynamic covariance (M1-M0) is strongly positive, moving mean alone (M2-M0) is negative, and the M1 vs M3 A_t paths are almost identical after subtracting the estimated moving center.",
            "Body",
        )
    )
    add_image(
        story,
        "results/nested_mean_cov_robustness/charts/base5_headline/04_A_t_survival_M1_vs_M3.png",
        "Base-five A_t survival: covariance amplification after estimating the moving mean.",
    )
    add_image(
        story,
        "results/nested_mean_cov_robustness/charts/base5_headline/05_mean_drift_vs_cov_amplification.png",
        "Base-five mean-drift norm versus residual covariance amplification.",
    )
    if nested_meta:
        variants = nested_meta.get("variants", [])
        if variants:
            rows = []
            for v in variants:
                core = v.get("upgraded_state_space_core", {})
                rows.append(
                    (
                        v.get("variant"),
                        v.get("N"),
                        v.get("M"),
                        core.get("student_t_degrees_of_freedom"),
                        core.get("M1_alpha_hat"),
                        core.get("M3_alpha_hat"),
                        v.get("survival_corr_trace"),
                    )
                )
            story.append(heading("Nested State-Space Core", 2))
            story.append(
                df_table(
                    pd.DataFrame(rows, columns=["variant", "N", "M", "nu", "alpha M1", "alpha M3", "A_t corr"]),
                    max_rows=8,
                )
            )

    story.append(PageBreak())
    story.append(heading("6. Shock-Definition Robustness"))
    story.append(
        P(
            "The shock robustness layer compares the median-fallback baseline, MP_pm-only shocks, and event-level manually aggregated shocks. "
            "It uses invariant subspace diagnostics, tau-path correlations, top-month overlap, and basis-specific A_jj path correlations.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            principal angles: singular values of V_base' V_variant
            tau correlation: corr( tau_base,t, tau_variant,t )
            top-k overlap: | Top_k(tau_base) cap Top_k(tau_variant) |
            basis diagonal correlation: corr( A_base,jj,t, A_variant,matched(j)matched(j),t )
            """
        )
    )
    if not shock_summary.empty:
        story.append(
            df_table(
                shock_summary,
                columns=[
                    "variant",
                    "label",
                    "top5_trace_share",
                    "tau_sd",
                    "tau_max",
                    "tau_max_month",
                    "state_spectral_radius",
                ],
                rename={
                    "top5_trace_share": "trace share",
                    "tau_sd": "tau sd",
                    "tau_max": "max tau",
                    "tau_max_month": "max month",
                    "state_spectral_radius": "rho(F)",
                },
                max_rows=6,
                max_chars=54,
            )
        )
    if not shock_metrics.empty:
        story.append(
            df_table(
                shock_metrics,
                columns=[
                    "variant",
                    "top5_trace_share_diff_vs_baseline",
                    "max_principal_angle_degrees",
                    "tau_path_corr_with_baseline",
                    "top10_overlap_with_baseline",
                    "march_2020_tau",
                    "march_2020_rank",
                    "A55_diag_corr_with_baseline",
                ],
                rename={
                    "top5_trace_share_diff_vs_baseline": "share diff",
                    "max_principal_angle_degrees": "max angle",
                    "tau_path_corr_with_baseline": "tau corr",
                    "top10_overlap_with_baseline": "top10",
                    "march_2020_tau": "Mar20 tau",
                    "march_2020_rank": "Mar20 rank",
                    "A55_diag_corr_with_baseline": "A55 corr",
                },
                max_rows=6,
                max_chars=44,
            )
        )
    add_image(
        story,
        "results/top5_shock_robustness/outputs/charts/robustness_tau_paths_by_variant.png",
        "Tau paths across the three shock-definition variants.",
    )
    add_image(
        story,
        "results/top5_shock_robustness/outputs/charts/robustness_principal_angles_vs_baseline.png",
        "Principal angles versus the baseline retained response-score subspace.",
    )

    story.append(PageBreak())
    story.append(heading("7. SEC Macro-State Geometry Robustness"))
    story.append(
        P(
            "SEC is an append-only robustness check. It asks whether a smooth function of predetermined macro-financial state geometry can recover related response-score covariance geometry. "
            "It does not replace the publication-grade estimator.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            State vector: S_t uses only information dated t-1 or earlier.

            W_ij = exp( -||S_i-S_j||^2 / (sigma_i sigma_j) )
            W_alpha = D_q^{-alpha} W D_q^{-alpha}
            L_sym = I - D^{-1/2} W_alpha D^{-1/2}
            phi_j = D^{-1/2} u_j

            Phi_t = [1, phi_1(S_t), ..., phi_L(S_t)]
            E_hat = Phi B_score
            K_SEC,score = T^{-1} E_hat' E_hat
            """
        )
    )
    story.append(
        formula(
            r"""
            y_t = svec( log(H_t) )
            y_hat_t = Phi_t B_level
            A_raw(S_t) = exp( smat(y_hat_t) )
            C = T^{-1} sum_t A_raw(S_t)
            A_SEC(S_t) = C^{-1/2} A_raw(S_t) C^{-1/2}

            K_SEC,t = V_5 Lambda_5^{1/2} A_SEC(S_t) Lambda_5^{1/2} V_5'
            tau_SEC,t = trace(A_SEC(S_t)) / 5
            """
        )
    )
    if not sec_trace.empty:
        story.append(
            df_table(
                sec_trace,
                columns=[
                    "variant",
                    "baseline_top5_trace_share",
                    "sec_score_top5_trace_share_of_total",
                    "sec_score_internal_top5_share",
                    "sec_rank5_kernel_trace_share",
                ],
                rename={
                    "baseline_top5_trace_share": "base share",
                    "sec_score_top5_trace_share_of_total": "SEC-score share",
                    "sec_score_internal_top5_share": "SEC internal share",
                    "sec_rank5_kernel_trace_share": "SEC rank5 mean share",
                },
                max_rows=6,
                max_chars=44,
            )
        )
    if not sec_tau.empty:
        story.append(
            df_table(
                sec_tau,
                columns=[
                    "variant",
                    "tau_corr",
                    "tau_rmse",
                    "tau_crisis_window_corr",
                    "baseline_tau_max_month",
                    "sec_tau_max_month",
                    "top10_overlap_count",
                    "march_2020_sec_rank",
                    "march_2020_baseline_rank",
                ],
                rename={
                    "tau_crisis_window_corr": "crisis corr",
                    "baseline_tau_max_month": "base max",
                    "sec_tau_max_month": "SEC max",
                    "top10_overlap_count": "top10 overlap",
                    "march_2020_sec_rank": "Mar20 SEC rank",
                    "march_2020_baseline_rank": "Mar20 base rank",
                },
                max_rows=6,
                max_chars=46,
            )
        )
    if not sec_angles.empty:
        story.append(
            df_table(
                sec_angles,
                columns=["variant", "max_angle_degrees", "mean_angle_degrees", "angle_1_degrees", "angle_5_degrees"],
                rename={
                    "max_angle_degrees": "max angle",
                    "mean_angle_degrees": "mean angle",
                    "angle_1_degrees": "angle 1",
                    "angle_5_degrees": "angle 5",
                },
                max_rows=6,
            )
        )
    add_image(
        story,
        "sec_robustness_results/charts/sec_tau_all_shocks_overlay.png",
        "SEC-conditioned tau_t paths across shock definitions.",
    )
    add_image(
        story,
        "sec_robustness_results/charts/sec_state_manifold_tau_map.png",
        "SEC macro-state manifold with tau_t intensity.",
    )

    story.append(PageBreak())
    story.append(heading("8. Proxy-IV LP/OVK Extension"))
    story.append(
        P(
            "The IV appendix replaces the OLS-style shock score with exactly identified scalar-IV influence score surfaces. "
            "The external instrument is the SF Fed/Bauer-Swanson monetary-policy surprise and the endogenous policy indicator is DGS1.",
            "Body",
        )
    )
    story.append(
        formula(
            r"""
            Residualize x_t, z_t, and Y_t on the common LP control matrix C.

            q_zx = mean_t( z_res,t x_res,t )
            beta_IV = mean_t( z_res,t Y_res,t ) / q_zx
            u_IV,t = Y_res,t - x_res,t beta_IV

            psi_IV,t = z_res,t u_IV,t / q_zx
            K_IV = T^{-1} sum_t (psi_IV,t - beta_psi)(psi_IV,t - beta_psi)'
            """
        )
    )
    if not iv_sample.empty:
        story.append(df_table(iv_sample, max_rows=16, max_chars=90, col_widths=[2.65 * inch, 4.35 * inch]))
    if not iv_first.empty:
        story.append(
            df_table(
                iv_first,
                columns=[
                    "nobs",
                    "q_zx",
                    "corr_zx",
                    "pi_hat",
                    "first_stage_partial_r2",
                    "first_stage_f_stat",
                    "weak_iv_warning",
                ],
                max_rows=2,
            )
        )
    if not iv_rank.empty:
        story.append(
            df_table(
                iv_rank,
                columns=[
                    "rank",
                    "state_dim",
                    "retained_trace_share",
                    "tau_sd",
                    "tau_max",
                    "tau_max_month",
                    "transition_spectral_radius",
                ],
                rename={
                    "retained_trace_share": "trace share",
                    "tau_sd": "tau sd",
                    "tau_max": "max tau",
                    "tau_max_month": "max month",
                    "transition_spectral_radius": "rho(F)",
                },
                max_rows=8,
            )
        )
    if not iv_decomp.empty:
        story.append(
            df_table(
                iv_decomp,
                columns=[
                    "max_tau_month",
                    "max_tau",
                    "driver_label_for_max_tau_month",
                    "top10_first_stage_driven_count",
                    "top10_residual_driven_count",
                    "top10_mixed_count",
                    "corr_tau_log_first_stage_leverage",
                    "corr_tau_log_retained_residual_energy",
                ],
                rename={
                    "driver_label_for_max_tau_month": "driver",
                    "top10_first_stage_driven_count": "top10 first-stage",
                    "top10_residual_driven_count": "top10 residual",
                    "top10_mixed_count": "top10 mixed",
                    "corr_tau_log_first_stage_leverage": "corr first-stage",
                    "corr_tau_log_retained_residual_energy": "corr residual",
                },
                max_rows=2,
                max_chars=42,
            )
        )
    if not iv_base.empty:
        story.append(
            df_table(
                iv_base,
                columns=["overlap_observations", "tau_correlation", "top10_overlap", "iv_max_month", "iv_max_tau", "baseline_max_month", "baseline_max_tau"],
                rename={
                    "overlap_observations": "overlap n",
                    "tau_correlation": "tau corr",
                    "top10_overlap": "top10",
                    "iv_max_month": "IV max",
                    "iv_max_tau": "IV max tau",
                    "baseline_max_month": "base max",
                    "baseline_max_tau": "base max tau",
                },
                max_rows=2,
            )
        )
    if not iv_nested.empty:
        story.append(heading("IV Nested Mean/Covariance Check", 2))
        story.append(
            df_table(
                iv_nested,
                columns=["comparison", "avg_log_score_diff", "p05", "p95", "prob_diff_gt_0"],
                rename={"avg_log_score_diff": "avg diff", "prob_diff_gt_0": "P(diff>0)"},
                max_rows=8,
            )
        )
    add_image(
        story,
        "results/iv_ovk/charts/iv_tau_overlay_baseline.png",
        "Proxy-IV tau_t overlaid against the baseline publication-grade tau_t.",
    )
    add_image(
        story,
        "results/iv_ovk/charts/iv_score_decomposition_top_tau.png",
        "Score-energy decomposition for top proxy-IV tau months.",
    )
    add_image(
        story,
        "results/iv_ovk/charts/iv_bootstrap_tau_bands.png",
        "Proxy-IV full-pipeline bootstrap tau bands.",
    )

    story.append(PageBreak())
    story.append(heading("9. What Was Proven or Checked"))
    proof_rows = [
        ("FWL score identity", "The average score surface q_t equals the residualized multivariate LP coefficient beta_hat."),
        ("OVK positive semidefiniteness", "K_bar is an average of score outer products, so all finite block quadratic forms are nonnegative."),
        ("Trace interpretation", "trace(K_bar) is average squared centered score-surface norm; trace shares are variance-geometry shares."),
        ("Rank-R optimality", "The leading eigenbasis is the best rank-R approximation of K_bar in Frobenius norm."),
        ("Whitened factors", "z_t has sample covariance I_R inside the retained eigenspace."),
        ("SPD proxy", "G_t = alpha I + (1-alpha) z_t z_t' is strictly positive definite for alpha > 0."),
        ("Mean preservation", "Normalizing A_t to sample mean I_R makes mean(K_t) equal the retained average kernel."),
        ("Positive dynamic kernel", "K_t is PSD whenever A_t is PSD; it is positive on the retained subspace when A_t is SPD."),
        ("Log-state stationarity", "rho(F)<1 gives a stable VAR(1) log-covariance state representation."),
        ("Uncertainty bands", "FFBS gives conditional state uncertainty; moving-block full-pipeline bootstrap rebuilds scores, basis, parameters, and tau."),
        ("Nested mean/covariance separation", "M1, M2, and M3 isolate dynamic covariance, moving mean, and their joint contribution."),
        ("SEC state conditioning", "A_SEC(S_t) remains SPD and normalized; SEC rank-five mean kernel preserves the retained mean-kernel construction."),
        ("Proxy-IV score identity", "Exactly identified IV scores use z_res * u_IV / mean(z_res*x_res), then enter the same OVK machinery."),
    ]
    story.append(df_table(pd.DataFrame(proof_rows, columns=["Item", "Result"]), max_rows=20, max_chars=100, col_widths=[2.15 * inch, 4.85 * inch]))

    story.append(heading("Existing PDF Reports in Workspace", 2))
    pdf_rows = []
    for p in [
        "results/reports/publication_grade_ovk_report.pdf",
        "results/reports/nested_mean_cov_robustness_report.pdf",
        "results/reports/top5_full_appended_report.pdf",
        "results/reports/iv_ovk_report.pdf",
        "sec_robustness_results/reports/sec_robustness_appendix.pdf",
        "sec_robustness_results/math_appendix/sec_math_appendix.pdf",
        "monthly_ovk_top5_with_SEC_robustness_report.pdf",
    ]:
        path = rel(p)
        pdf_rows.append((p, "present" if path.exists() else "missing", path.stat().st_size if path.exists() else ""))
    story.append(df_table(pd.DataFrame(pdf_rows, columns=["Report", "Status", "Bytes"]), max_rows=12, max_chars=95, col_widths=[5.1 * inch, 0.8 * inch, 1.1 * inch]))

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.58 * inch,
        title="OVK Workspace Math Report",
        author="Codex",
    )
    doc.build(story, onFirstPage=page_header, onLaterPages=page_header)


if __name__ == "__main__":
    build_report()
    print(PDF_PATH)
