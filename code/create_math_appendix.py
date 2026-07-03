from pathlib import Path
import pandas as pd
import numpy as np
import shutil, zipfile, os, textwrap, subprocess, json, csv
from datetime import datetime
from pypdf import PdfReader, PdfWriter

base_pack = Path('/mnt/data/monthly_ovk_top5_full_appended_results_pack')
base_report = Path('/mnt/data/monthly_ovk_top5_full_appended_report.pdf')
if not base_pack.exists():
    raise FileNotFoundError(base_pack)
if not base_report.exists():
    raise FileNotFoundError(base_report)

out_root = Path('/mnt/data/monthly_ovk_top5_full_math_results_pack')
if out_root.exists():
    shutil.rmtree(out_root)
shutil.copytree(base_pack, out_root)

math_dir = out_root / 'math_appendix'
math_dir.mkdir(parents=True, exist_ok=True)
reports_dir = out_root / 'reports'
reports_dir.mkdir(exist_ok=True)

# Load key result tables for dynamic values.
eigen = pd.read_csv(base_pack/'top5_baseline_state_space_results/tables/average_ovk_eigenspectrum_with_bands.csv')
state = pd.read_csv(base_pack/'top5_baseline_state_space_results/tables/top5_state_space_model_summary.csv')
rob = pd.read_csv(base_pack/'robustness_comparison_results/tables/robustness_variant_summary.csv')
metrics = pd.read_csv(base_pack/'robustness_comparison_results/tables/robustness_comparison_metrics.csv')
diag_corr = pd.read_csv(base_pack/'robustness_comparison_results/tables/basis_specific_A_diag_path_correlations.csv')

# Helper extract values.
def state_val(key):
    row = state.loc[state['item'].eq(key), 'value']
    return row.iloc[0] if len(row) else ''

R = int(float(state_val('retained_bases_R')))
d = int(float(state_val('state_dimension_R_times_Rplus1_over_2')))
alpha = 0.25
rho = state_val('VAR_spectral_radius')
proc_share = state_val('process_noise_share')
meas_share = state_val('measurement_noise_share')
boot = state_val('bootstrap_draws')
state_draws = state_val('state_uncertainty_draws')
block_len = state_val('bootstrap_block_length_months')
top1 = eigen.loc[eigen['rank'].eq(1), 'share_estimate'].iloc[0]
top3 = eigen.loc[eigen['rank'].eq(3), 'cumulative_estimate'].iloc[0]
top5 = eigen.loc[eigen['rank'].eq(5), 'cumulative_estimate'].iloc[0]
top5_lo = eigen.loc[eigen['rank'].eq(5), 'cumulative_p05'].iloc[0]
top5_hi = eigen.loc[eigen['rank'].eq(5), 'cumulative_p95'].iloc[0]
max_tau = float(state_val('max_trace_A_over_R'))
max_month = state_val('max_month')
max_tau_lo = float(state_val('max_month_trace_p05'))
max_tau_hi = float(state_val('max_month_trace_p95'))

sample = pd.read_csv(base_pack/'top5_baseline_state_space_results/tables/sample_and_specification_top5.csv')
# sample table may have item/value fields
try:
    n_valid = sample.loc[sample['item'].eq('LP usable base months'), 'value'].iloc[0]
except Exception:
    n_valid = '392'

rob_table_tex = rob[['label','top5_trace_share','tau_max','tau_max_month','state_spectral_radius']].copy()
rob_table_tex['top5_trace_share'] = rob_table_tex['top5_trace_share'].map(lambda x: f'{x:.3f}')
rob_table_tex['tau_max'] = rob_table_tex['tau_max'].map(lambda x: f'{x:.3f}')
rob_table_tex['state_spectral_radius'] = rob_table_tex['state_spectral_radius'].map(lambda x: f'{x:.3f}')
rob_rows = '\n'.join([f"{r['label'].replace('&','\\&')} & {r['top5_trace_share']} & {r['tau_max']} & {r['tau_max_month']} & {r['state_spectral_radius']} \\\\" for _, r in rob_table_tex.iterrows()])

metric_rows = metrics[['label','top5_trace_share','max_principal_angle_degrees','tau_path_corr_with_baseline','top10_overlap_with_baseline','march_2020_tau','march_2020_rank']].copy()
for c in ['top5_trace_share','max_principal_angle_degrees','tau_path_corr_with_baseline','march_2020_tau']:
    metric_rows[c] = metric_rows[c].map(lambda x: f'{x:.3f}')
metric_rows_tex = '\n'.join([f"{r['label'].replace('&','\\&')} & {r['top5_trace_share']} & {r['max_principal_angle_degrees']} & {r['tau_path_corr_with_baseline']} & {r['top10_overlap_with_baseline']} & {r['march_2020_tau']} & {r['march_2020_rank']} \\\\" for _, r in metric_rows.iterrows()])

crosswalk_rows = [
    ('T', 'Number of usable monthly score surfaces', 'n_valid, valid_idx', 'sample_and_specification_top5.csv'),
    ('M', 'Stacked response dimension, horizons times variables', 'M=(H+1)*p', 'generate_top5_ovk_pack.py'),
    ('q_t', 'LP score surface', 'Q_scores[t,:]', 'average_irf_with_block_bootstrap_bands.csv'),
    ('hat beta', 'Average LP score / average IRF', 'beta_hat', 'average_irf_with_block_bootstrap_bands.csv'),
    ('hat K', 'Average operator-valued kernel matrix', 'K_bar', 'not stored directly; eigen outputs stored'),
    ('V_R', 'Top R basis matrix', 'V', 'top5_basis_loadings_with_bootstrap_bands.csv'),
    ('lambda_r', 'Kernel eigenvalues / trace shares', 'evals, shares', 'average_ovk_eigenspectrum_with_bands.csv'),
    ('z_t', 'Whitened finite-rank score factor', 'Z', 'computed in code'),
    ('G_t', 'Full-rank SPD proxy for z_t z_t prime', 'G_raw', 'generate_top5_ovk_pack.py'),
    ('y_t', 'Matrix-log observation vector', 'Ylog', 'generate_top5_ovk_pack.py'),
    ('F,Q,R_e', 'State transition, process covariance, measurement covariance', 'F, Qproc, Rmeas', 'top5_state_VAR_F_matrix.csv and covariance CSVs'),
    ('A_t', 'Latent PSD drift matrix', 'A_state', 'state_space_A_t_top5_drift_estimates_with_bands.csv'),
    ('tau_t', 'Total kernel amplification trace(A_t)/R', 'trace_A', 'state_space_A_t_top5_drift_estimates_with_bands.csv'),
]
with open(math_dir/'algorithm_math_to_code_crosswalk.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['math_symbol','meaning','code_variable','pack_location'])
    writer.writerows(crosswalk_rows)

pseudo = r'''# Mathematical algorithm in executable pseudocode
# Inputs: monthly panel, shock definition, horizons H, lags L, retained rank R

# 1. Residualize shock and stacked future responses on controls.
m_res = residualize(m_std, X_controls)
Y_res = residualize_each_column(Y_future_minus_baseline, X_controls)

# 2. Build local-projection score surfaces.
sigma2 = mean(m_res**2)
Q_scores = (m_res[:, None] * Y_res) / sigma2
beta_hat = mean(Q_scores, axis=0)
E = Q_scores - beta_hat

# 3. Estimate the average operator-valued kernel.
K_bar = E.T @ E / T
lambda_all, V_all = eig_sorted_descending(K_bar)
V = V_all[:, :R]
Lambda = diag(lambda_all[:R])

# 4. Whiten finite-rank factor scores.
Z = E @ V @ Lambda^{-1/2}
# Then sample mean of Z_t Z_t' is I_R.

# 5. Create full-rank SPD covariance proxies.
G_t = alpha * I_R + (1-alpha) * Z_t Z_t'
G_t = mean(G_t)^{-1/2} G_t mean(G_t)^{-1/2}
y_t = svec(logm(G_t))

# 6. Fit stationary log-Euclidean state-space model.
y_t = x_t + eps_t
x_t = mu + F(x_{t-1} - mu) + eta_t
# Enforce/check spectral_radius(F) < 1.
# Apply Kalman filter and Rauch-Tung-Striebel smoother.

# 7. Map smoothed state back to SPD drift matrices.
B_t = expm(smat(x_{t|T}))
C = mean(B_t)
A_t = C^{-1/2} B_t C^{-1/2}

# 8. Reconstruct time-varying rank-R OVK.
K_t = V Lambda^{1/2} A_t Lambda^{1/2} V.T
tau_t = trace(A_t) / R

# 9. Robustness comparison across shock definitions.
# Compare top-R subspace angles, trace shares, tau_t correlations,
# top amplification month overlap, and matched diagonal A_jj,t paths.
'''
(math_dir/'algorithm_pseudocode.py').write_text(pseudo)

tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=0.88in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs,longtable,array}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{fancyvrb}
\usepackage{microtype}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\newtheorem{assumption}{Assumption}
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\tr}{\operatorname{tr}}
\newcommand{\rank}{\operatorname{rank}}
\newcommand{\vech}{\operatorname{vech}}
\newcommand{\svec}{\operatorname{svec}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\argmin}{\operatorname*{arg\,min}}
\newcommand{\HS}{\operatorname{HS}}
\title{Technical Mathematical Appendix\\\large Finite-Rank Time-Varying Random Operator-Valued Kernel Model for Monetary-Policy LP Score Surfaces}
\author{Generated appendix for the monthly top-five OVK results pack}
\date{June 2026}
\begin{document}
\maketitle

\begin{abstract}
This appendix states the mathematical estimands, derivations, algorithmic identities, and proof sketches behind the empirical top-five operator-valued kernel (OVK) analysis. The empirical implementation uses monthly local-projection score surfaces, estimates an average OVK, retains the leading five-dimensional subspace, and models time variation through a positive-definite log-Euclidean state-space process for $A_t$. The key empirical dimensions are $R=\VAR_R$ retained bases, state dimension $R(R+1)/2=\VAR_D$, $\widehat\psi_5=\VAR_TOPFIVE$ top-five trace share, and maximum baseline amplification $\max_t \widehat\tau_t=\VAR_MAXTAU$ in \VAR_MAXMONTH.
\end{abstract}

\tableofcontents
\newpage

\section{Purpose and high-level object}

The empirical object is not just an average impulse response. It is a time-varying covariance geometry for entire monetary-policy local-projection response surfaces. At each usable month $t$, let
\[
q_t \in \R^M, \qquad M=(H+1)p,
\]
be a stacked score surface over horizons $h=0,\ldots,H$ and outcome variables $j=1,\ldots,p$. In the monthly run, $H=24$, $p=5$, hence $M=125$. The average local-projection response is
\[
\widehat\beta = \frac{1}{T}\sum_{t=1}^T q_t.
\]
The average OVK is the covariance matrix of centered score surfaces,
\[
\widehat K = \frac{1}{T}\sum_{t=1}^T (q_t-\widehat\beta)(q_t-\widehat\beta)'.
\]
Because the response vector is ordered by horizon and variable, $\widehat K$ can be read blockwise as
\[
\widehat K(h,h') \in \R^{p\times p}.
\]
The block $\widehat K(h,h')$ describes how score variation at horizon $h$ co-moves with score variation at horizon $h'$ across the five outcomes. This is the finite-dimensional empirical version of an operator-valued kernel.

The dynamic finite-rank model is
\[
\widehat K_t^{(R)} = V_R \Lambda_R^{1/2} A_t \Lambda_R^{1/2} V_R', \qquad A_t\in\mathbb{S}_{++}^R,
\]
with sample normalization
\[
\frac{1}{T}\sum_{t=1}^T A_t=I_R.
\]
The scalar amplification statistic is
\[
\tau_t = \frac{1}{R}\tr(A_t).
\]
For the reported top-five run, $R=\VAR_R$, $\widehat\psi_5=\sum_{r=1}^5 \widehat\lambda_r/\tr(\widehat K)=\VAR_TOPFIVE$, with bootstrap interval $[\VAR_TOPFIVELO,\VAR_TOPFIVEHI]$.

\section{Data-to-score derivation}

\subsection{Stacked local-projection responses}

Let $x_{j,t}$ be the transformed monthly outcome $j$ at month $t$. The implementation uses
\[
100\log(IP_t),\quad 100\log(CPI_t),\quad UNRATE_t,\quad GS2_t,\quad BAA10Y_t.
\]
For each base month $t$ and horizon $h$, define the local-projection response
\[
Y_{j,t,h}=x_{j,t+h}-x_{j,t-1}.
\]
Stacking over $h=0,\ldots,H$ and $j=1,\ldots,p$ gives
\[
Y_t = \operatorname{stack}\{Y_{j,t,h}\}_{h,j}\in\R^M.
\]
Let $m_t$ be the standardized monetary-policy shock and let $X_t$ collect controls: an intercept, trend, current central-bank-information shock, lags of monetary and information shocks, and lags of macro levels and macro first differences.

\subsection{Frisch-Waugh-Lovell residualization}

Let $M_X$ denote the residual-maker matrix for the control matrix $X$ in the finite sample:
\[
M_X = I - X(X'X)^{-1}X'.
\]
Define residualized shock and residualized response surfaces:
\[
\widetilde m = M_X m, \qquad \widetilde Y = M_X Y.
\]
Equivalently, $\widetilde m_t$ and $\widetilde Y_t$ are the residuals after projecting $m_t$ and each column of $Y_t$ on the same controls.

The vector local-projection coefficient solves
\[
\min_{\beta,\Gamma}\sum_{t=1}^T \|Y_t-m_t\beta-X_t\Gamma\|_2^2.
\]
By the Frisch-Waugh-Lovell theorem,
\[
\widehat\beta = \frac{\sum_t \widetilde m_t\widetilde Y_t}{\sum_t \widetilde m_t^2}
= \frac{1}{T}\sum_t \frac{\widetilde m_t\widetilde Y_t}{T^{-1}\sum_s \widetilde m_s^2}.
\]
This motivates the score surface
\[
q_t=\frac{\widetilde m_t\widetilde Y_t}{\widehat\sigma_m^2},\qquad \widehat\sigma_m^2=\frac{1}{T}\sum_s \widetilde m_s^2.
\]
Then
\[
\widehat\beta = \frac{1}{T}\sum_t q_t.
\]

\begin{proposition}[LP score mean identity]
The sample mean of the score surfaces equals the Frisch-Waugh-Lovell local-projection coefficient vector.
\end{proposition}
\begin{proof}
By definition,
\[
\frac{1}{T}\sum_t q_t
=\frac{1}{T}\sum_t \frac{\widetilde m_t\widetilde Y_t}{\widehat\sigma_m^2}
=\frac{\sum_t \widetilde m_t\widetilde Y_t}{T\widehat\sigma_m^2}
=\frac{\sum_t \widetilde m_t\widetilde Y_t}{\sum_t \widetilde m_t^2}.
\]
The final expression is the FWL coefficient from regressing the residualized response surface on the residualized shock. \qedhere
\end{proof}

\section{Average operator-valued kernel}

\subsection{Finite-dimensional OVK representation}

Index a stacked coordinate by $a=(h,j)$, with horizon $h$ and variable $j$. The sample covariance matrix $\widehat K\in\R^{M\times M}$ has entries
\[
\widehat K_{ab}=\frac{1}{T}\sum_t (q_{t,a}-\widehat\beta_a)(q_{t,b}-\widehat\beta_b).
\]
Blockwise, for $h,h'\in\{0,\ldots,H\}$,
\[
\widehat K(h,h')_{j,k}=\widehat K_{(h,j),(h',k)}.
\]
Thus $\widehat K(h,h')$ is a linear operator from variable space at horizon $h'$ to variable space at horizon $h$.

\begin{proposition}[Positive-definiteness of the empirical OVK]
The block kernel $\widehat K(h,h')$ is positive semidefinite as an operator-valued kernel.
\end{proposition}
\begin{proof}
Take arbitrary vectors $a_h\in\R^p$ for $h=0,\ldots,H$ and stack them into $a\in\R^M$. Then
\[
\sum_{h,h'} a_h'\widehat K(h,h')a_{h'} = a'\widehat K a
=\frac{1}{T}\sum_t \left[a'(q_t-\widehat\beta)\right]^2\ge 0.
\]
This is exactly the positive-definiteness condition for a finite operator-valued kernel. \qedhere
\end{proof}

\begin{proposition}[Population mean kernel]
Suppose $q_t$ is strictly stationary with $\E\|q_t\|^2<\infty$. Define $\beta=\E q_t$ and $K=\E[(q_t-\beta)(q_t-\beta)']$. Then $K$ is positive semidefinite and the block map $K(h,h')$ is a population OVK.
\end{proposition}
\begin{proof}
For any $a\in\R^M$,
\[
a'Ka = \E\left[ a'(q_t-\beta)(q_t-\beta)'a\right]
=\E\left[ a'(q_t-\beta)\right]^2\ge 0.
\]
The block representation is only a re-indexing of the same positive semidefinite matrix. \qedhere
\end{proof}

\section{Finite-rank reduction}

Let the spectral decomposition of the sample average OVK be
\[
\widehat K = V\Lambda V',
\]
where $V'V=I$, $\Lambda=\diag(\widehat\lambda_1,\ldots,\widehat\lambda_M)$, and $\widehat\lambda_1\ge\cdots\ge\widehat\lambda_M\ge 0$. The rank-$R$ approximation is
\[
\widehat K_R=V_R\Lambda_R V_R',
\]
where $V_R=(v_1,\ldots,v_R)$ and $\Lambda_R=\diag(\widehat\lambda_1,\ldots,\widehat\lambda_R)$.

\begin{proposition}[Optimality of the rank-$R$ average kernel]
Among all matrices $B$ with $\rank(B)\le R$, $\widehat K_R$ solves
\[
\min_{\rank(B)\le R}\|\widehat K-B\|_{\HS}.
\]
\end{proposition}
\begin{proof}
This is the Eckart-Young-Mirsky theorem applied to the symmetric positive semidefinite matrix $\widehat K$. The Hilbert-Schmidt norm in finite dimensions is the Frobenius norm. \qedhere
\end{proof}

\subsection{Whitened finite-rank score factors}

Define centered score surfaces
\[
e_t=q_t-\widehat\beta.
\]
Define standardized factor scores
\[
z_t=\Lambda_R^{-1/2}V_R'e_t\in\R^R.
\]
The code uses the transpose convention
\[
z_t'=e_t'V_R\Lambda_R^{-1/2}.
\]

\begin{proposition}[Sample covariance normalization]
The standardized factor scores satisfy
\[
\frac{1}{T}\sum_t z_tz_t'=I_R.
\]
\end{proposition}
\begin{proof}
Using $\widehat K=T^{-1}\sum_t e_te_t'$,
\[
\frac{1}{T}\sum_t z_tz_t'
=\Lambda_R^{-1/2}V_R'\widehat K V_R\Lambda_R^{-1/2}
=\Lambda_R^{-1/2}\Lambda_R\Lambda_R^{-1/2}=I_R.
\]
\end{proof}

This identity is operationally important. It means all time variation in the rank-$R$ covariance geometry is expressed relative to an average identity metric in the retained factor space.

\section{Time-varying OVK drift matrix $A_t$}

\begin{definition}[Finite-rank time-varying OVK]
For a positive semidefinite matrix $A_t\in\mathbb{S}_+^R$, define
\[
K_t^{(R)}=V_R\Lambda_R^{1/2}A_t\Lambda_R^{1/2}V_R'.
\]
\end{definition}

\begin{proposition}[PSD preservation]
If $A_t\succeq 0$, then $K_t^{(R)}\succeq 0$ and hence defines a positive semidefinite finite operator-valued kernel.
\end{proposition}
\begin{proof}
For any $a\in\R^M$,
\[
a'K_t^{(R)}a = b'A_tb,
\qquad b=\Lambda_R^{1/2}V_R'a.
\]
Since $A_t\succeq0$, $b'A_tb\ge0$. \qedhere
\end{proof}

\begin{proposition}[Mean recovery]
If $T^{-1}\sum_t A_t=I_R$, then the sample average of $K_t^{(R)}$ equals the retained average kernel $\widehat K_R$:
\[
\frac{1}{T}\sum_t K_t^{(R)}=\widehat K_R.
\]
\end{proposition}
\begin{proof}
\[
\frac{1}{T}\sum_t K_t^{(R)}
=V_R\Lambda_R^{1/2}\left(\frac{1}{T}\sum_t A_t\right)\Lambda_R^{1/2}V_R'
=V_R\Lambda_R V_R' = \widehat K_R.
\]
\end{proof}

\section{Full-rank SPD proxy and matrix logarithm}

The raw outer product $z_tz_t'$ is positive semidefinite but rank one. A matrix logarithm requires strict positive definiteness. The implementation therefore uses
\[
G_t=\alpha I_R+(1-\alpha)z_tz_t',\qquad 0<\alpha<1.
\]
In the top-five run, $\alpha=\VAR_ALPHA$.

\begin{lemma}[Strict positive definiteness of $G_t$]
For $0<\alpha<1$, $G_t\in\mathbb{S}_{++}^R$ for every $z_t$.
\end{lemma}
\begin{proof}
For any nonzero $x\in\R^R$,
\[
x'G_tx = \alpha\|x\|^2+(1-\alpha)(x'z_t)^2 \ge \alpha\|x\|^2>0.
\]
Therefore all eigenvalues of $G_t$ are strictly positive. \qedhere
\end{proof}

Normalize the proxy by its sample mean:
\[
\bar G=\frac{1}{T}\sum_t G_t,
\qquad \widetilde G_t=\bar G^{-1/2}G_t\bar G^{-1/2}.
\]
Then map to the symmetric matrix logarithm:
\[
Y_t=\log(\widetilde G_t),
\qquad y_t=\svec(Y_t)\in\R^d,
\qquad d=R(R+1)/2.
\]
For $R=\VAR_R$, $d=\VAR_D$.

\section{Log-Euclidean state-space model}

The latent state is $x_t\in\R^d$. The model is
\[
y_t=x_t+\varepsilon_t,
\qquad \varepsilon_t\sim N(0,R_e),
\]
\[
x_t=\mu+F(x_{t-1}-\mu)+\eta_t,
\qquad \eta_t\sim N(0,Q).
\]
The fitted transition matrix has spectral radius $\rho(F)=\VAR_RHO$, below one.

\begin{theorem}[Stationarity of the log-state process]
If $\rho(F)<1$ and $\eta_t$ is covariance-stationary with finite second moments, then the state equation admits a unique covariance-stationary solution
\[
x_t=\mu+\sum_{k=0}^{\infty}F^k\eta_{t-k}.
\]
If $\eta_t$ has covariance $Q$, then the state covariance $\Omega$ solves the discrete Lyapunov equation
\[
\Omega=F\Omega F'+Q.
\]
\end{theorem}
\begin{proof}
Since $\rho(F)<1$, there exists a matrix norm and constants $C<\infty$, $0<r<1$ such that $\|F^k\|\le Cr^k$. Hence the infinite series converges in mean square. Substituting the series into the recursion verifies that it is a solution. Uniqueness follows because the difference between any two stationary solutions satisfies $d_t=Fd_{t-1}$, which implies $\E\|d_t\|^2\to0$ under $\rho(F)<1$. Taking variances in the recursion gives $\Omega=F\Omega F'+Q$. \qedhere
\end{proof}

\subsection{Kalman filter and smoother}

For the observation equation $y_t=x_t+\varepsilon_t$, the one-step prediction and update are
\[
\hat x_{t|t-1}=\mu+F(\hat x_{t-1|t-1}-\mu),
\]
\[
P_{t|t-1}=FP_{t-1|t-1}F'+Q,
\]
\[
S_t=P_{t|t-1}+R_e,
\qquad K_t^{Kal}=P_{t|t-1}S_t^{-1},
\]
\[
\hat x_{t|t}=\hat x_{t|t-1}+K_t^{Kal}(y_t-\hat x_{t|t-1}),
\]
\[
P_{t|t}=(I-K_t^{Kal})P_{t|t-1}(I-K_t^{Kal})'+K_t^{Kal}R_e(K_t^{Kal})'.
\]
The Rauch-Tung-Striebel backward smoother uses
\[
J_t=P_{t|t}F'P_{t+1|t}^{-1},
\]
\[
\hat x_{t|T}=\hat x_{t|t}+J_t(\hat x_{t+1|T}-\hat x_{t+1|t}),
\]
\[
P_{t|T}=P_{t|t}+J_t(P_{t+1|T}-P_{t+1|t})J_t'.
\]

\subsection{Mapping back to positive-definite $A_t$}

Let $\operatorname{smat}$ invert $\svec$. Define
\[
B_t=\exp\{\operatorname{smat}(\hat x_{t|T})\}.
\]
Each $B_t$ is strictly positive definite. Normalize
\[
C=\frac{1}{T}\sum_t B_t,
\qquad A_t=C^{-1/2}B_tC^{-1/2}.
\]

\begin{proposition}[SPD preservation and mean identity]
Each $A_t$ is strictly positive definite and $T^{-1}\sum_t A_t=I_R$.
\end{proposition}
\begin{proof}
The matrix exponential of a symmetric matrix is strictly positive definite, so $B_t\succ0$. Since $C$ is the average of strictly positive definite matrices, $C\succ0$ and $C^{-1/2}$ exists. Congruence by a nonsingular matrix preserves strict positive definiteness, hence $A_t\succ0$. Finally,
\[
\frac{1}{T}\sum_t A_t=C^{-1/2}\left(\frac{1}{T}\sum_tB_t\right)C^{-1/2}=C^{-1/2}CC^{-1/2}=I_R.
\]
\end{proof}

\section{Uncertainty bands}

\subsection{Moving-block bootstrap for score-based objects}

The score surfaces $q_t$ are serially dependent because macro outcomes overlap across horizons and because shocks and controls are time series. The implementation uses moving-block bootstrap resamples of the score surfaces. For each bootstrap draw $b$:
\begin{enumerate}[leftmargin=1.2em]
\item resample blocks of $q_t$ with block length $\ell=\VAR_BLOCK$ months;
\item recompute $\widehat\beta^{(b)}$, $\widehat K^{(b)}$, eigenvalues, eigenvectors, and retained basis loadings;
\item align eigenvector signs to the baseline; and
\item take pointwise empirical quantiles.
\end{enumerate}
The reported baseline uses $\VAR_BOOT$ bootstrap draws.

\begin{theorem}[Bootstrap validity, operational version]
Assume $(q_t)$ is strictly stationary, strongly mixing with sufficiently fast mixing coefficients, has finite $4+\delta$ moments, and the retained eigenvalues are separated by nonzero eigengaps. If block length $\ell\to\infty$ and $\ell/T\to0$, then the moving-block bootstrap consistently approximates the limiting distribution of smooth functionals of $\widehat K$, including separated eigenvalues and sign-aligned eigenvectors.
\end{theorem}
\begin{proof}[Proof sketch]
Under the stated dependence and moment conditions, a central limit theorem holds for sample means of $q_t$ and $\operatorname{vec}(q_tq_t')$. The moving-block bootstrap consistently reproduces the long-run covariance of these sample moments. The covariance matrix $\widehat K$ is a smooth function of these sample moments. Eigenvalues and eigenvectors are differentiable functions of $\widehat K$ when eigengaps are nonzero. The result follows by the functional delta method and the bootstrap delta method. \qedhere
\end{proof}

\subsection{State-smoothing bands for $A_t$}

Conditional on the fitted state-space model, the smoother gives
\[
x_t\mid y_{1:T}\approx N(\hat x_{t|T},P_{t|T}).
\]
The implementation draws
\[
x_t^{(b)}\sim N(\hat x_{t|T},P_{t|T}),
\]
maps each draw through the matrix exponential and mean normalization, and computes quantiles for $\tau_t$ and the diagonal entries $A_{jj,t}$. These are pointwise conditional state-smoothing bands, not full posterior intervals over all tuning choices.

\section{Robustness comparison mathematics}

The robustness variants are:
\[
\text{MP\_median with fallback},\qquad \text{MP\_pm only},\qquad \text{event-level shocks aggregated manually}.
\]
The comparison is not only about average IRFs. It is about whether the leading response-geometry subspace and crisis-state amplification survive shock redefinition.

\subsection{Subspace principal angles}

Let $V_0,V_1\in\R^{M\times R}$ be two orthonormal top-$R$ basis matrices. Compute the singular values of $V_0'V_1$:
\[
\sigma_1\ge\cdots\ge\sigma_R\ge0.
\]
The principal angles are
\[
\theta_j=\arccos(\sigma_j).
\]
Small maximum angle means the retained subspaces are close. This comparison is invariant to sign changes and rotations within the retained subspace.

\begin{proposition}[Rotation invariance of the subspace comparison]
For any orthogonal $R\times R$ matrices $O_0,O_1$, the singular values of $(V_0O_0)'(V_1O_1)$ equal the singular values of $V_0'V_1$.
\end{proposition}
\begin{proof}
\[
(V_0O_0)'(V_1O_1)=O_0'(V_0'V_1)O_1.
\]
Left and right multiplication by orthogonal matrices preserves singular values. \qedhere
\end{proof}

\subsection{Trace share and amplification paths}

The retained trace share is
\[
\psi_R=\frac{\sum_{r=1}^R\widehat\lambda_r}{\tr(\widehat K)}.
\]
The path comparison uses
\[
\operatorname{corr}(\tau_t^{(0)},\tau_t^{(v)}),
\qquad \tau_t^{(v)}=\frac{1}{R}\tr(A_t^{(v)}).
\]
Top-month overlap compares the sets of largest local amplification months. Basis-specific comparisons use matched basis vectors and correlations of diagonal paths $A_{jj,t}$.

\subsection{Why basis-specific labels are more fragile than the top-five subspace}

Let $B_t=V_R\Lambda_R^{1/2}A_t\Lambda_R^{1/2}V_R'$. In whitened retained coordinates, an orthogonal rotation $O$ changes the coordinate representation but not the kernel:
\[
V_R\Lambda_R^{1/2}A_t\Lambda_R^{1/2}V_R'
= (V_R\Lambda_R^{1/2}O)(O'A_tO)(O'\Lambda_R^{1/2}V_R')
\]
when the retained feature map is represented by $V_R\Lambda_R^{1/2}O$. Thus the full retained subspace is the stable geometric object. A named lower-ranked basis vector, especially basis 4 or basis 5, can rotate without destroying the top-five kernel geometry. This is why the robustness pack compares both subspaces and basis-specific diagonal paths.

\section{Numerical robustness summary}

\begin{table}[h!]
\centering
\begin{tabular}{lrrrr}
\toprule
Shock definition & Top-five share & Max $\tau_t$ & Max month & State radius\\
\midrule
\VAR_ROBROWS
\bottomrule
\end{tabular}
\caption{Top-five trace share and state-space amplification by shock definition.}
\end{table}

\begin{table}[h!]
\centering
\begin{tabular}{lrrrrrr}
\toprule
Shock definition & Top-five share & Max angle & Corr($\tau$) & Top-10 overlap & Mar. 2020 $\tau$ & Rank\\
\midrule
\VAR_METRICROWS
\bottomrule
\end{tabular}
\caption{Rotation-invariant and pathwise robustness diagnostics.}
\end{table}

The low-rank result is stable: the top-five trace share remains close to $0.90$ across all three definitions. The event-level manual aggregation is close to the baseline in both subspace angle and $\tau_t$ path correlation. The MP\_pm-only variant remains low-rank but rotates the fifth basis strongly and weakens the basis-specific $A_{55,t}$ correlation. Therefore, the stable claim is the top-five response-geometry subspace and repeated crisis amplification; the fragile claim is the exact naming of lower-ranked bases, especially bases 4 and 5.

\section{Assumptions needed for interpretation}

\begin{assumption}[Shock relevance]
The residualized shock variance satisfies $\E\widetilde m_t^2>0$.
\end{assumption}

\begin{assumption}[Moment existence]
The score surface satisfies $\E\|q_t\|^{4+\delta}<\infty$ for some $\delta>0$.
\end{assumption}

\begin{assumption}[Weak dependence]
The score process is stationary and weakly dependent enough for laws of large numbers, central limit theory, and block bootstrap approximation of second moments.
\end{assumption}

\begin{assumption}[Eigenvalue separation]
For basis-specific inference, the retained eigenvalues have nonzero local eigengaps. For subspace inference, the gap between $\lambda_R$ and $\lambda_{R+1}$ is the main required separation.
\end{assumption}

\begin{assumption}[Shock exogeneity for causal IRFs]
To interpret $\widehat\beta$ as a causal monetary-policy response, the residualized monetary-policy shock must be conditionally exogenous given controls and the information-shock adjustment. The OVK covariance geometry can still be estimated descriptively without this causal interpretation.
\end{assumption}

\section{Consistency statements}

\begin{proposition}[Consistency of the average OVK]
Under stationarity, ergodicity, and $\E\|q_t\|^2<\infty$,
\[
\widehat K \to K
\]
almost surely entrywise and in finite-dimensional Hilbert-Schmidt norm.
\end{proposition}
\begin{proof}
The entries of $\widehat K$ are finite sums and products of sample averages of $q_{t,a}$ and $q_{t,a}q_{t,b}$. By the ergodic theorem, these sample averages converge almost surely to their expectations. Finite dimensionality gives Hilbert-Schmidt convergence. \qedhere
\end{proof}

\begin{proposition}[Continuity of retained eigenspaces]
If $K$ has an eigengap $\lambda_R(K)>\lambda_{R+1}(K)$ and $\|\widehat K-K\|\to0$, then the projection onto the estimated top-$R$ subspace converges to the population top-$R$ projection.
\end{proposition}
\begin{proof}[Proof sketch]
This is a standard consequence of spectral perturbation theory. The distance between spectral projectors is bounded by a constant times $\|\widehat K-K\|$ divided by the eigengap. \qedhere
\end{proof}

\begin{proposition}[Correctness of the dynamic OVK reconstruction]
For every fitted month $t$, the algorithm returns a positive semidefinite finite-rank OVK $\widehat K_t^{(R)}$. Its sample average equals $\widehat K_R$.
\end{proposition}
\begin{proof}
The log-Euclidean state model returns $A_t\succ0$ after exponentiation and normalization. PSD preservation follows from Proposition 4. Mean recovery follows from Proposition 5. \qedhere
\end{proof}

\section{Operational theorem: what the algorithm estimates}

\begin{theorem}[Operational meaning of the algorithm]
Under the assumptions above, the algorithm consistently estimates the average response-score covariance kernel $K$, its leading retained subspace, and a model-based stationary positive-definite latent path $A_t$ describing time variation in the retained score-covariance geometry. The reconstructed object
\[
\widehat K_t^{(R)}=V_R\Lambda_R^{1/2}A_t\Lambda_R^{1/2}V_R'
\]
is positive semidefinite for every $t$ and averages to the retained sample kernel $\widehat K_R$.
\end{theorem}
\begin{proof}
The score construction gives the LP coefficient mean identity. The centered score covariance is a PSD empirical OVK and converges to its population counterpart under ergodicity. Spectral truncation gives the optimal rank-$R$ Hilbert-Schmidt approximation and, under an eigengap, a stable retained subspace. Whitened scores have identity sample covariance, so time variation can be represented as a positive-definite metric $A_t$ around the identity. The shrinkage proxy is strictly SPD, the matrix logarithm is well-defined, and the stationary state equation is valid because $\rho(F)<1$. Exponentiation and sample mean normalization preserve SPD and enforce $T^{-1}\sum_t A_t=I_R$. Therefore each reconstructed $\widehat K_t^{(R)}$ is PSD and its sample average is $\widehat K_R$. \qedhere
\end{proof}

\section{Code crosswalk}

\begin{longtable}{p{0.18\textwidth}p{0.32\textwidth}p{0.22\textwidth}p{0.20\textwidth}}
\toprule
Math object & Meaning & Code variable & Output file\\
\midrule
\endhead
$T$ & Number of usable monthly score surfaces & \texttt{n\_valid}, \texttt{valid\_idx} & \texttt{sample\_and\_specification\_top5.csv}\\
$M$ & Stacked response dimension & \texttt{M=(H+1)*p} & code\\
$q_t$ & LP score surface & \texttt{Q\_scores[t,:]} & IRF tables\\
$\widehat\beta$ & Average LP response & \texttt{beta\_hat} & \texttt{average\_irf\_with\_...csv}\\
$\widehat K$ & Average OVK & \texttt{K\_bar} & eigen tables\\
$V_R$ & Retained basis matrix & \texttt{V} & basis loading tables\\
$\lambda_r$ & Eigenvalues / trace shares & \texttt{evals}, \texttt{shares} & eigenspectrum table\\
$z_t$ & Whitened rank-$R$ score & \texttt{Z} & computed in code\\
$G_t$ & Full-rank SPD proxy & \texttt{G\_raw} & code\\
$y_t$ & Matrix-log observation vector & \texttt{Ylog} & code\\
$F,Q,R_e$ & State transition and covariance matrices & \texttt{F}, \texttt{Qproc}, \texttt{Rmeas} & state matrices CSVs\\
$A_t$ & Positive-definite drift matrix & \texttt{A\_state} & drift estimates CSV\\
$\tau_t$ & Total amplification & \texttt{trace\_A} & drift estimates CSV\\
\bottomrule
\end{longtable}

\section{Pseudocode}

\begin{Verbatim}[fontsize=\small]
Input: monthly panel, shock definition, horizons H, lags L, rank R.

1. Construct Y_t = stack{x_{j,t+h} - x_{j,t-1}} over horizons and variables.
2. Residualize shock m_t and Y_t on controls X_t.
3. Form q_t = residualized_m_t * residualized_Y_t / mean(residualized_m_t^2).
4. Set beta_hat = mean(q_t). Set e_t = q_t - beta_hat.
5. Estimate K_bar = mean(e_t e_t').
6. Eigendecompose K_bar and retain V_R, Lambda_R.
7. Whiten retained scores: z_t = Lambda_R^{-1/2} V_R' e_t.
8. Build SPD proxy G_t = alpha I + (1-alpha) z_t z_t'.
9. Normalize G_t by its sample mean and map to y_t = svec(log(G_t)).
10. Fit stationary VAR(1) state model in log space.
11. Kalman-smooth latent x_t.
12. Map back: B_t = exp(smat(x_t)); A_t = C^{-1/2}B_t C^{-1/2}.
13. Reconstruct K_t = V_R Lambda_R^{1/2} A_t Lambda_R^{1/2} V_R'.
14. Compute tau_t = trace(A_t)/R and basis-specific A_jj,t paths.
15. Repeat for robustness shock definitions and compare subspaces and paths.
\end{Verbatim}

\section{Interpretation boundary}

The algorithm operationalizes a time-varying covariance geometry for monetary-policy response-score surfaces. It does not, by itself, prove that every spike in $\tau_t$ is a structural monetary-transmission regime. The strongest supported claim is that the leading five-dimensional response-score subspace is stable and that crisis periods repeatedly produce high-amplification covariance states. The weaker and more fragile claim is the exact economic naming of bases 4 and 5.

\end{document}
'''

replacements = {
    'VAR_R': str(R),
    'VAR_D': str(d),
    'VAR_TOPFIVE': f'{top5:.3f}',
    'VAR_TOPFIVELO': f'{top5_lo:.3f}',
    'VAR_TOPFIVEHI': f'{top5_hi:.3f}',
    'VAR_MAXTAU': f'{max_tau:.3f}',
    'VAR_MAXMONTH': str(max_month),
    'VAR_ALPHA': f'{alpha:.2f}',
    'VAR_RHO': f'{float(rho):.3f}',
    'VAR_BLOCK': str(block_len),
    'VAR_BOOT': str(boot),
    'VAR_ROBROWS': rob_rows,
    'VAR_METRICROWS': metric_rows_tex,
}
for k, v in replacements.items():
    tex = tex.replace('\\' + k, v if k in ['VAR_ROBROWS','VAR_METRICROWS'] else v)
# Fix accidental replaced command patterns not expected; using placeholders were \VAR_ tokens.

tex_path = math_dir/'technical_math_appendix.tex'
tex_path.write_text(tex)

# Markdown/plain source too.
md = f'''# Technical Mathematical Appendix - Operational OVK Algorithm

This source accompanies `technical_math_appendix.pdf` and the merged full report.
It documents the score construction, average operator-valued kernel, finite-rank truncation, log-Euclidean state-space model for A_t, uncertainty bands, robustness metrics, and proof sketches.

Key baseline values:
- Retained bases R: {R}
- State dimension R(R+1)/2: {d}
- Top-five trace share: {top5:.3f}, 90 percent bootstrap interval [{top5_lo:.3f}, {top5_hi:.3f}]
- VAR spectral radius: {float(rho):.3f}
- Maximum tau_t: {max_tau:.3f} in {max_month}, state interval [{max_tau_lo:.3f}, {max_tau_hi:.3f}]

See the PDF or TeX source for full derivations and proofs.
'''
(math_dir/'technical_math_appendix_summary.md').write_text(md)

# Compile latex. Run twice for TOC.
for i in range(2):
    result = subprocess.run(['pdflatex','-interaction=nonstopmode','technical_math_appendix.tex'], cwd=math_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        raise RuntimeError('pdflatex failed')
appendix_pdf = math_dir/'technical_math_appendix.pdf'
if not appendix_pdf.exists():
    raise FileNotFoundError(appendix_pdf)

# Copy appendix to top-level /mnt/data for easy download.
shutil.copy2(appendix_pdf, '/mnt/data/monthly_ovk_top5_math_technical_appendix.pdf')
shutil.copy2(tex_path, '/mnt/data/monthly_ovk_top5_math_technical_appendix.tex')

# Merge base full report + appendix.
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

# Update README.
readme_path = out_root/'README.md'
old = readme_path.read_text() if readme_path.exists() else ''
addition = f'''

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
print('out_root', out_root)
