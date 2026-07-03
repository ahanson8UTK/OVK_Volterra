from pathlib import Path
import os, shutil, zipfile, csv, textwrap, html
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted,
    Table, TableStyle, HRFlowable, KeepTogether
)
from pypdf import PdfReader, PdfWriter
from PIL import Image, ImageOps, ImageDraw

BASE = Path(os.environ.get('OVK_BASE_DIR', '/mnt/data'))
OLD_PACK = Path(os.environ.get('OVK_FULL_PACK', str(BASE / 'monthly_ovk_top5_full_appended_results_pack')))
OLD_REPORT = Path(os.environ.get('OVK_FULL_FINAL_PDF', str(BASE / 'monthly_ovk_top5_full_appended_report.pdf')))
NEW_PACK = Path(os.environ.get('OVK_MATH_PACK', str(BASE / 'monthly_ovk_top5_full_math_results_pack')))
OUT_ZIP = Path(os.environ.get('OVK_MATH_FINAL_ZIP', str(BASE / 'monthly_ovk_top5_full_appended_with_math_results_pack.zip')))
COMBINED_PDF = Path(os.environ.get('OVK_MATH_FINAL_PDF', str(BASE / 'monthly_ovk_top5_full_appended_with_math_report.pdf')))
APPENDIX_PDF = Path(os.environ.get('OVK_MATH_APPENDIX_PDF', str(BASE / 'monthly_ovk_top5_math_appendix.pdf')))
CONTACT_SHEET = Path(os.environ.get('OVK_MATH_CONTACT', str(BASE / 'monthly_ovk_top5_full_appended_with_math_contact_sheet.jpg')))
RENDER_DIR = Path(os.environ.get('OVK_MATH_RENDER_DIR', str(BASE / 'monthly_ovk_top5_math_pdf_render_check')))
for p in [OUT_ZIP, COMBINED_PDF, APPENDIX_PDF, CONTACT_SHEET]:
    p.parent.mkdir(parents=True, exist_ok=True)

# Clean pack directory.
if NEW_PACK.exists():
    shutil.rmtree(NEW_PACK)
shutil.copytree(OLD_PACK, NEW_PACK)

MATH_DIR = NEW_PACK / 'math'
MATH_DIR.mkdir(exist_ok=True)
REPORTS_DIR = NEW_PACK / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = NEW_PACK / 'math' / 'tables'
TABLES_DIR.mkdir(exist_ok=True, parents=True)
CODE_DIR = NEW_PACK / 'code'
CODE_DIR.mkdir(exist_ok=True)


def _sample_table_value(item):
    candidates = [
        NEW_PACK / 'top5_baseline_state_space_results' / 'tables' / 'sample_and_specification_top5.csv',
        NEW_PACK / 'tables' / 'sample_and_specification_top5.csv',
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('item') == item:
                    return row.get('value')
    return None


OUTCOME_LABELS_TEXT = _sample_table_value('outcomes') or 'IP, CPI, Unemployment, 2Y yield, BAA-10Y spread'
OUTCOME_COUNT = len([x for x in OUTCOME_LABELS_TEXT.split(',') if x.strip()])

# -----------------------------
# Math appendix structured content
# -----------------------------

notation_rows = [
    ('T', 'Number of usable monthly LP base observations.'),
    ('p', f'Number of outcome variables. Here p = {OUTCOME_COUNT}.'),
    ('H', 'Maximum forecast horizon in months. Here H = 24.'),
    ('M', 'Stacked response dimension M = p(H+1).'),
    ('R', 'Finite rank retained in the OVK basis. Here R = 5.'),
    ('y_t', 'p-vector of macro-financial outcomes in month t.'),
    ('m_t', 'Standardized monetary-policy shock in month t.'),
    ('c_t', 'Central-bank-information shock control in month t.'),
    ('w_t', 'Control vector for month t, including trend, shock lags, and macro lags.'),
    ('Y_t', 'Stacked future response vector: [(y_{t+h}-y_{t-1})] for h=0,...,H.'),
    ('W', 'T by k matrix of controls.'),
    ('M_W', 'Residual-maker I - W(W\'W)^+W\'.'),
    ('tilde m_t', 'Residualized monetary-policy shock.'),
    ('tilde Y_t', 'Residualized stacked future response vector.'),
    ('q_t', 'LP score surface: tilde m_t tilde Y_t / sigma_m^2.'),
    ('beta_hat', 'Average LP coefficient surface, equal to sample mean of q_t.'),
    ('e_t', 'Centered LP score surface q_t - beta_hat.'),
    ('K_bar', 'Average operator-valued kernel, estimated by T^{-1} sum e_t e_t\'.'),
    ('K_bar(h,h\')', 'p by p block mapping horizon h\' responses to horizon h responses.'),
    ('V_R', 'M by R matrix of leading eigenvectors of K_bar.'),
    ('Lambda_R', 'R by R diagonal matrix of leading eigenvalues of K_bar.'),
    ('z_t', 'Whitened finite-rank score factor Lambda_R^{-1/2} V_R\'e_t.'),
    ('A_t', 'R by R positive-definite drift matrix in the finite-rank kernel.'),
    ('tau_t', 'Total kernel amplification trace(A_t)/R.'),
    ('G_t', 'Full-rank raw covariance proxy alpha I + (1-alpha) z_t z_t\'.'),
    ('alpha', 'Shrinkage parameter guaranteeing G_t is positive definite.'),
    ('log(G_t)', 'Matrix logarithm of the SPD proxy.'),
    ('svec', 'Vectorization of the unique elements of a symmetric matrix.'),
    ('x_t', 'Latent log-covariance state in the Kalman model.'),
    ('F', 'VAR(1) transition matrix for x_t.'),
    ('Q', 'State innovation covariance.'),
    ('R_e', 'Measurement-error covariance for the observed log proxy.'),
    ('P_V', 'Projection matrix V_R V_R\' onto a leading subspace.'),
]

proposition_rows = [
    ('1', 'FWL score identity', 'The average LP coefficient equals the sample mean of the residualized score surface q_t.'),
    ('2', 'Average OVK positivity', 'K_bar(h,h\') is a valid finite-sample positive semidefinite operator-valued kernel.'),
    ('3', 'Trace equals score variance', 'tr(K_bar) equals the average squared norm of centered LP score surfaces.'),
    ('4', 'Whitened factors', 'z_t has sample covariance I_R inside the retained subspace.'),
    ('5', 'Rank-R optimality', 'The leading eigenbasis is the optimal rank-R approximation of K_bar in Frobenius norm.'),
    ('6', 'Dynamic kernel positivity', 'K_t = V_R Lambda_R^{1/2} A_t Lambda_R^{1/2} V_R\' is PSD whenever A_t is PSD.'),
    ('7', 'Mean preservation', 'If mean(A_t)=I_R, then mean(K_t)=K_bar,R.'),
    ('8', 'SPD proxy validity', 'G_t = alpha I + (1-alpha) z_t z_t\' is SPD for alpha > 0.'),
    ('9', 'Stationarity of log-state', 'rho(F)<1 gives a unique stationary VAR(1) state solution.'),
    ('10', 'Normalization theorem', 'A_t = C^{-1/2} exp(X_t) C^{-1/2} is SPD and has sample mean I_R.'),
    ('11', 'Subspace-robust comparison', 'Principal angles compare leading subspaces invariantly to sign and basis rotations.'),
]

# Markdown appendix text. Keep equations ASCII-friendly so the same source renders well in plain text.
md_sections = []

def add_md(title, body):
    md_sections.append(f"## {title}\n\n{body.strip()}\n")

front_matter = f"""# Mathematical Appendix: Top-Five Monetary-Policy OVK Algorithm

This appendix operationalizes the algorithm used in the top-five robustness pack. It gives the notation, derivations, proof sketches, state-space construction, uncertainty-band logic, and robustness diagnostics underlying the empirical outputs. The formulas are written in implementation notation, so every object maps directly to the CSV tables and Python scripts included in the ZIP pack.

This appendix is intentionally self-contained. It does not prove monetary-policy identification. It proves that, conditional on the chosen shock and controls, the score-based construction is a valid finite-rank operator-valued kernel, that the dynamic A_t path stays positive definite, and that the normalization preserves the average kernel.
"""

add_md('1. Data objects and stacked local-projection responses', f"""
Let y_t be a p-dimensional monthly outcome vector. In the current pack p = {OUTCOME_COUNT}: {OUTCOME_LABELS_TEXT}. For a maximum horizon H = 24, define the stacked future response vector

    Y_t = vec_h( y_{t+h} - y_{t-1} ),  h = 0,...,H.

Thus Y_t is an M-vector with M = p(H+1). The baseline shock is m_t = MP_median with MP_pm fallback when MP_median is missing. The control shock is CBI_used, built analogously. The control vector w_t contains a constant, a linear trend, current CBI_used, 12 lags of MP_used and CBI_used, and 12 lags of macro levels and first differences.

Stacking observations over usable months gives Y as a T by M matrix, m as a T-vector, and W as a T by k control matrix. Let

    M_W = I_T - W (W'W)^+ W'

be the residual-maker, where + denotes the Moore-Penrose inverse. Define

    m_tilde = M_W m,
    Y_tilde = M_W Y,
    sigma_m^2 = T^{-1} m_tilde' m_tilde.
""")

add_md('2. Frisch-Waugh-Lovell derivation of the LP score surface', r"""
The multivariate local-projection coefficient surface after partialling out W is

    beta_hat = (m_tilde' m_tilde)^{-1} m_tilde' Y_tilde.

Define the one-observation score surface

    q_t = m_tilde_t * Y_tilde_t / sigma_m^2,

where q_t is an M-vector. Then

    T^{-1} sum_t q_t
      = T^{-1} sum_t [m_tilde_t Y_tilde_t / sigma_m^2]
      = (m_tilde' Y_tilde) / (T sigma_m^2)
      = (m_tilde' Y_tilde) / (m_tilde' m_tilde)
      = beta_hat.

So the average LP surface is the sample mean of the score surfaces. This is the key operational bridge from local projections to a covariance kernel: once q_t is built, all later objects are functions of the centered score surface

    e_t = q_t - beta_hat.

Proof. The first equality is the definition of the sample mean. The second is the definition of q_t. The third uses sigma_m^2 = T^{-1}m_tilde'm_tilde. The last equality is the FWL coefficient formula.
""")

add_md('3. Constructing the operator-valued kernel', r"""
For each centered score vector e_t, write e_t(h) for the p-vector corresponding to horizon h. Define the horizon-by-horizon block kernel

    K_bar(h,h') = T^{-1} sum_t e_t(h) e_t(h')',

where K_bar(h,h') is p by p. In stacked matrix form,

    K_bar = T^{-1} E' E,

where E is the T by M matrix with rows e_t'.

This is an operator-valued kernel because the input index is a horizon h and the output space is R^p. It maps coefficient vectors attached to horizon h' into coefficient vectors attached to horizon h. Economically, it is the average covariance geometry of complete monetary-policy response-score surfaces.
""")

add_md('4. Proof that K_bar is a positive semidefinite OVK', r"""
Take any finite set of horizon coefficients a_0,...,a_H with a_h in R^p. Then

    sum_{h,h'} a_h' K_bar(h,h') a_{h'}
      = T^{-1} sum_t sum_{h,h'} a_h' e_t(h) e_t(h')' a_{h'}
      = T^{-1} sum_t [ sum_h a_h' e_t(h) ]^2
      >= 0.

Hence K_bar is positive semidefinite as an operator-valued kernel. Symmetry follows because

    K_bar(h,h')' = K_bar(h',h).

This proof is finite-sample and does not require asymptotics. It only uses the fact that K_bar is built from centered outer products of observed score surfaces.
""")

add_md('5. Trace, variance, and economic meaning of the kernel', r"""
The trace of K_bar is

    tr(K_bar) = T^{-1} sum_t tr(e_t e_t')
              = T^{-1} sum_t ||e_t||^2.

Therefore the kernel trace equals average squared response-score variation. The top-five trace share is

    rho_5 = [sum_{r=1}^5 lambda_r] / [sum_j lambda_j].

It is the fraction of total response-score variance explained by the leading five kernel directions. This is why the trace share is the central low-rank diagnostic. It is not an IRF magnitude. It is a variance-geometry measure over complete response surfaces.
""")

add_md('6. Spectral decomposition and finite-rank optimality', r"""
Because K_bar is symmetric positive semidefinite, it has the spectral decomposition

    K_bar = V Lambda V',

where V has orthonormal columns and Lambda has nonnegative eigenvalues. Let V_R contain the first R eigenvectors and Lambda_R the first R eigenvalues. The retained rank-R average kernel is

    K_bar,R = V_R Lambda_R V_R'.

By the Eckart-Young-Mirsky theorem for symmetric matrices, K_bar,R is the best rank-R approximation to K_bar in Frobenius norm:

    K_bar,R = argmin_{rank(B)<=R} ||K_bar - B||_F.

For a PSD covariance matrix, this also means the retained eigenvalues maximize explained trace among all R-dimensional orthonormal subspaces. Operationally, using R = 5 chooses the five-dimensional response-geometry subspace with maximum sample covariance energy.
""")

add_md('7. Whitened finite-rank factors', r"""
Define the retained factor score

    z_t = Lambda_R^{-1/2} V_R' e_t.

Then the sample covariance of z_t is exactly identity inside the retained subspace:

    T^{-1} sum_t z_t z_t'
      = Lambda_R^{-1/2} V_R' [T^{-1} sum_t e_t e_t'] V_R Lambda_R^{-1/2}
      = Lambda_R^{-1/2} V_R' K_bar V_R Lambda_R^{-1/2}
      = Lambda_R^{-1/2} Lambda_R Lambda_R^{-1/2}
      = I_R.

This whitening is why A_t can be interpreted as local amplification relative to the average retained covariance geometry. In average units, z_t has identity covariance; deviations of A_t from I_R are deviations from the average kernel geometry.
""")

add_md('8. Dynamic finite-rank kernel and mean preservation', r"""
The time-varying finite-rank kernel is modeled as

    K_t = V_R Lambda_R^{1/2} A_t Lambda_R^{1/2} V_R'.

For any b in R^M,

    b' K_t b
      = [Lambda_R^{1/2} V_R' b]' A_t [Lambda_R^{1/2} V_R' b].

Therefore K_t is positive semidefinite whenever A_t is positive semidefinite. If A_t is positive definite, K_t is positive definite on the retained subspace.

If the dynamic drift is normalized so that

    T^{-1} sum_t A_t = I_R,

then

    T^{-1} sum_t K_t
      = V_R Lambda_R^{1/2} [T^{-1} sum_t A_t] Lambda_R^{1/2} V_R'
      = V_R Lambda_R V_R'
      = K_bar,R.

Thus the dynamic model preserves the average retained kernel exactly.
""")

add_md('9. Why the raw proxy must be made full rank', r"""
The most direct local covariance proxy is z_t z_t'. But z_t z_t' has rank one, so it is singular when R > 1. The matrix logarithm is not defined on singular positive semidefinite matrices. The state-space model therefore uses the shrinkage proxy

    G_t = alpha I_R + (1-alpha) z_t z_t',  0 < alpha < 1.

For any nonzero u,

    u' G_t u = alpha ||u||^2 + (1-alpha)(u'z_t)^2 >= alpha ||u||^2 > 0.

So G_t is symmetric positive definite. This guarantees that log(G_t) exists. The shrinkage is not merely numerical convenience; it is the mathematical step that moves the rank-one score proxy into the SPD cone.
""")

add_md('10. Normalizing the proxy before taking logs', r"""
Let

    G_bar = T^{-1} sum_t G_t.

Since each G_t is SPD, G_bar is SPD. Define

    H_t = G_bar^{-1/2} G_t G_bar^{-1/2}.

Then H_t is SPD and

    T^{-1} sum_t H_t
      = G_bar^{-1/2} [T^{-1} sum_t G_t] G_bar^{-1/2}
      = G_bar^{-1/2} G_bar G_bar^{-1/2}
      = I_R.

The observed log proxy is

    y_t^log = svec( log(H_t) ).

The matrix logarithm maps the SPD cone into the vector space of symmetric matrices. This makes a linear Gaussian state model possible while preserving positive definiteness after mapping back with the matrix exponential.
""")

add_md('11. Log-Euclidean state model for A_t', r"""
Let d = R(R+1)/2. For R = 5, d = 15. The observation is a d-vector

    y_t^log = svec(log(H_t)).

The latent state is

    x_t = mu + F(x_{t-1} - mu) + eta_t,
    y_t^log = x_t + eps_t,

with eta_t ~ N(0,Q) and eps_t ~ N(0,R_e). The observation matrix is the identity because the observed log proxy is a noisy version of the latent log covariance state.

The state equation is estimated as a stationary VAR(1) in log-covariance space. The fitted spectral radius of F is checked. If all eigenvalues of F lie inside the unit circle, the log-state evolution is stable and mean reverting.
""")

add_md('12. Stationarity proof for the log-state model', r"""
Assume rho(F) < 1, where rho(F) is the spectral radius of F. Define x_tilde_t = x_t - mu. Then

    x_tilde_t = F x_tilde_{t-1} + eta_t.

Iterating backward gives

    x_tilde_t = F^J x_tilde_{t-J} + sum_{j=0}^{J-1} F^j eta_{t-j}.

Because rho(F) < 1, F^J -> 0 as J -> infinity. Therefore the unique stationary solution is

    x_t - mu = sum_{j=0}^{infinity} F^j eta_{t-j}.

Its covariance Sigma_x satisfies the discrete Lyapunov equation

    Sigma_x = F Sigma_x F' + Q.

This establishes stationary drift in the latent log-kernel state. Mapping through the matrix exponential creates a stationary positive-definite A_t process, subject to the sample normalization used in the estimator.
""")

add_md('13. Kalman filter and smoother recursions', r"""
With identity observation matrix, the Kalman prediction step is

    x_{t|t-1} = mu + F(x_{t-1|t-1} - mu),
    P_{t|t-1} = F P_{t-1|t-1} F' + Q.

The update step is

    S_t = P_{t|t-1} + R_e,
    K_t^KF = P_{t|t-1} S_t^{-1},
    x_{t|t} = x_{t|t-1} + K_t^KF (y_t^log - x_{t|t-1}),
    P_{t|t} = (I - K_t^KF) P_{t|t-1} (I - K_t^KF)' + K_t^KF R_e (K_t^KF)'.

The final covariance update is written in Joseph form to preserve numerical symmetry and positive semidefiniteness.

The Rauch-Tung-Striebel smoother is

    J_t = P_{t|t} F' (P_{t+1|t})^{-1},
    x_{t|T} = x_{t|t} + J_t (x_{t+1|T} - x_{t+1|t}),
    P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'.

The smoothed x_{t|T} and P_{t|T} are the inputs for A_t point estimates and state-smoothing uncertainty bands.
""")

add_md('14. Mapping the smoothed state back to A_t', r"""
Let smat be the inverse of svec, turning a d-vector into an R by R symmetric matrix. The unnormalized positive-definite state is

    B_t = exp( smat(x_{t|T}) ).

The matrix exponential of a symmetric matrix is SPD. Define

    C = T^{-1} sum_t B_t,
    A_t = C^{-1/2} B_t C^{-1/2}.

Then A_t is SPD and

    T^{-1} sum_t A_t
      = C^{-1/2} [T^{-1} sum_t B_t] C^{-1/2}
      = C^{-1/2} C C^{-1/2}
      = I_R.

This proves both positive definiteness and mean preservation. It is the exact step that keeps the dynamic kernel centered around the average retained OVK.
""")

add_md('15. Total and basis-specific amplification', r"""
The scalar total amplification is

    tau_t = tr(A_t) / R.

Because mean(A_t) = I_R, the sample mean of tau_t is one. Values above one mean that the retained finite-rank response-score geometry is amplified relative to its average state. Values below one mean attenuation.

The diagonal entries A_{rr,t} are basis-specific amplifications. They are not separate structural multipliers. They indicate how strongly the latent response-score covariance loads on retained basis r at date t. Off-diagonal entries measure coupling between bases. In robustness work, diagonal paths should be compared only after matching bases across shock definitions, because eigenvectors can change sign or rotate within near-degenerate subspaces.
""")

add_md('16. Uncertainty bands for average IRFs, eigenvalues, and bases', r"""
The score bootstrap resamples rows of q_t in circular moving blocks. For bootstrap draw b:

    q_t^{*(b)} -> beta_hat^{*(b)} -> K_bar^{*(b)} -> V_R^{*(b)}, Lambda_R^{*(b)}.

The block length is chosen to preserve serial dependence in overlapping horizon responses. Percentile bands are then computed from the bootstrap distribution.

For eigenvectors, signs are arbitrary. Each bootstrap basis vector is sign-aligned to the baseline by multiplying by sign(v_r^{*'} v_r). For lower-ranked bases, pointwise loading bands should be interpreted cautiously because eigenvectors can rotate when eigenvalues are close. Subspace comparisons are more reliable than individual basis labels.
""")

add_md('17. State-smoothing bands for A_t', r"""
The Kalman smoother gives the conditional Gaussian approximation

    x_t | y_1,...,y_T ~ N(x_{t|T}, P_{t|T}).

For draw b, sample

    x_t^{(b)} = x_{t|T} + L_t u_t^{(b)},  u_t^{(b)} ~ N(0,I),

where L_t L_t' = P_{t|T}. Map each draw through the same exponential and normalization:

    B_t^{(b)} = exp(smat(x_t^{(b)})),
    A_t^{(b)} = C_b^{-1/2} B_t^{(b)} C_b^{-1/2},
    C_b = T^{-1} sum_t B_t^{(b)}.

Pointwise quantiles of tau_t^{(b)} and A_{rr,t}^{(b)} give the plotted state-smoothing bands. These bands are conditional on the fitted state-space model and do not include full parameter uncertainty or shock-definition uncertainty.
""")

add_md('18. Robustness diagnostics over shock definitions', r"""
For each shock definition v, compute V_v, Lambda_v, A_{t,v}, and tau_{t,v}. The pack compares three definitions: MP_median with fallback, MP_pm only, and event-level shocks aggregated manually.

The leading five-dimensional subspace is compared through principal angles. Let V_0 be the baseline basis and V_v the alternative basis. Compute singular values

    sigma_j = singular_values( V_0' V_v ),  j = 1,...,R.

The principal angles are

    theta_j = arccos( sigma_j ).

This comparison is invariant to sign flips and rotations inside the retained subspace. It is the correct diagnostic for whether the leading five-dimensional response geometry survives a shock-definition change.

The pack also compares:

    rho_5,v = top-five trace share,
    corr(tau_0, tau_v),
    top-month overlap,
    corr(A_{rr,0}, A_{rr,v}) after basis matching.

A stable result shows similar top-five subspace geometry and repeated crisis amplification across definitions. A fragile result shows that basis 4, basis 5, or the March 2020 spike depends on the fallback rule.
""")

add_md('19. Subspace perturbation logic', r"""
Let K be the population score-kernel and K_hat the empirical one. If the eigengap

    delta = lambda_R(K) - lambda_{R+1}(K)

is positive, perturbation theory implies that the estimated R-dimensional subspace is stable when ||K_hat - K|| is small relative to delta. A Davis-Kahan style bound has the form

    ||sin Theta(V_hat_R, V_R)|| <= ||K_hat - K|| / delta.

Operational implication: if basis 4 and basis 5 have small eigengaps, their individual shapes can move across robustness versions even when the top-five subspace remains stable. That is why the report treats the top-five subspace as more reliable than the economic naming of the lower-ranked bases.
""")

add_md('20. Consistency logic under weak dependence', r"""
Under stationarity, ergodicity, finite second moments of e_t, and a fixed horizon stack, the sample kernel satisfies

    K_bar_hat = T^{-1} sum_t e_t e_t' -> E[e_t e_t']

almost surely. Under stronger weak-dependence and moment assumptions, a central limit theorem applies to linear functionals of K_bar_hat. This justifies using block bootstrap methods to approximate sampling uncertainty in the average IRF, average kernel, eigenvalue shares, and basis loadings.

The state-space A_t estimate is a model-based smoother of a latent covariance process. Its validity depends additionally on the adequacy of the log-Euclidean VAR(1) approximation and the measurement-error split. The proofs above guarantee positive definiteness and mean preservation; they do not prove that the fitted latent state is the true structural monetary-transmission process.
""")

add_md('21. Algorithm pseudocode', r"""
Input: monthly macro panel, shock definition, horizon H, lags L, rank R, shrinkage alpha.

1. Build y_t, m_t, c_t, and controls w_t.
2. Build stacked responses Y_t = vec_h(y_{t+h}-y_{t-1}).
3. Residualize m and Y on W.
4. Compute q_t = m_tilde_t Y_tilde_t / sigma_m^2.
5. Compute beta_hat = mean(q_t) and e_t = q_t - beta_hat.
6. Compute K_bar = T^{-1} E'E.
7. Eigendecompose K_bar; retain V_R and Lambda_R.
8. Compute z_t = Lambda_R^{-1/2} V_R'e_t.
9. Build G_t = alpha I + (1-alpha) z_t z_t'.
10. Normalize G_t by its sample mean and take y_t^log = svec(log(G_t)).
11. Fit stationary VAR(1) state model and Kalman smooth x_t.
12. Map x_t back with matrix exponential and normalize to mean(A_t)=I.
13. Form K_t = V_R Lambda_R^{1/2} A_t Lambda_R^{1/2} V_R'.
14. Compute tau_t, A_{rr,t}, top months, basis diagnostics, and robustness comparisons.
15. Build block-bootstrap bands for average IRF, eigenspectrum, and basis loadings.
16. Build state-smoothing bands for tau_t and A_{rr,t}.

Output: average IRF, average OVK, top-five basis, dynamic A_t path, uncertainty bands, and robustness diagnostics.
""")

add_md('22. What the proof package establishes and what it does not', r"""
Established by the derivations:

- The LP score surface averages to the residualized LP coefficient surface.
- The average score covariance is a valid positive semidefinite operator-valued kernel.
- The top-five basis is the energy-maximizing finite-rank representation of that kernel.
- The whitened finite-rank score factors have identity sample covariance.
- The dynamic kernel is PSD whenever A_t is PSD.
- The log-Euclidean construction keeps A_t positive definite.
- The sample normalization mean(A_t)=I preserves the average retained OVK.
- Principal-angle diagnostics are the right way to compare leading subspaces across shock definitions.

Not established by these derivations:

- The monetary-policy shock is perfectly identified.
- A_t is a structural monetary-transmission state rather than a response-score covariance state.
- Pointwise uncertainty bands are simultaneous confidence bands.
- The exact labels of bases 4 and 5 are invariant to all reasonable specifications.

The correct empirical interpretation is therefore disciplined: the robust object is a low-dimensional, time-varying response-score covariance geometry. Structural monetary interpretation requires the robustness and identification checks included in the pack.
""")

math_md = front_matter + "\n\n" + "\n".join(md_sections)
MD_PATH = MATH_DIR / 'mathematical_algorithm_appendix.md'
TEX_PATH = MATH_DIR / 'mathematical_algorithm_appendix.tex'
MD_PATH.write_text(math_md, encoding='utf-8')

# Lightweight LaTeX source for users who want to typeset externally.
latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{hyperref}
\title{Mathematical Appendix: Top-Five Monetary-Policy OVK Algorithm}
\author{Generated analysis pack appendix}
\date{}
\begin{document}
\maketitle
\noindent This LaTeX source mirrors the Markdown/PDF mathematical appendix in the ZIP pack. Equations are written in implementation notation so every object maps to the included code and CSV tables.

\section*{Core definitions}
Let $y_t\in\mathbb R^p$, $h=0,\ldots,H$, and $M=p(H+1)$. Define $Y_t=\operatorname{vec}_h(y_{t+h}-y_{t-1})$. Let $M_W=I-W(W'W)^+W'$ and set $\tilde m=M_Wm$, $\tilde Y=M_WY$, and $\hat\sigma_m^2=T^{-1}\tilde m'\tilde m$.

\section*{LP score identity}
The residualized LP estimator is
\[
\hat\beta=(\tilde m'\tilde m)^{-1}\tilde m'\tilde Y.
\]
With $q_t=\tilde m_t\tilde Y_t/\hat\sigma_m^2$,
\[
T^{-1}\sum_t q_t = \frac{\tilde m'\tilde Y}{T\hat\sigma_m^2}=\frac{\tilde m'\tilde Y}{\tilde m'\tilde m}=\hat\beta.
\]

\section*{OVK positivity}
Define $e_t=q_t-\hat\beta$ and $\bar K(h,h')=T^{-1}\sum_t e_t(h)e_t(h')'$. For any $a_h\in\mathbb R^p$,
\[
\sum_{h,h'}a_h'\bar K(h,h')a_{h'}=T^{-1}\sum_t\left(\sum_h a_h'e_t(h)\right)^2\ge 0.
\]
Thus $\bar K$ is a positive semidefinite operator-valued kernel.

\section*{Finite rank and whitening}
Let $\bar K=V\Lambda V'$ and retain $V_R,\Lambda_R$. The rank-$R$ approximation is $\bar K_R=V_R\Lambda_RV_R'$. Define $z_t=\Lambda_R^{-1/2}V_R'e_t$. Then
\[
T^{-1}\sum_t z_tz_t'=\Lambda_R^{-1/2}V_R'\bar K V_R\Lambda_R^{-1/2}=I_R.
\]

\section*{Dynamic kernel}
Let $A_t\succeq 0$ and define
\[
K_t=V_R\Lambda_R^{1/2}A_t\Lambda_R^{1/2}V_R'.
\]
Then $K_t\succeq 0$. If $T^{-1}\sum_tA_t=I_R$, then $T^{-1}\sum_tK_t=\bar K_R$.

\section*{Log-Euclidean state model}
Use $G_t=\alpha I_R+(1-\alpha)z_tz_t'$ with $0<\alpha<1$. Then $G_t\succ0$. Normalize $H_t=\bar G^{-1/2}G_t\bar G^{-1/2}$ and set $y_t^{log}=\operatorname{svec}(\log H_t)$. Model
\[
x_t=\mu+F(x_{t-1}-\mu)+\eta_t,\qquad y_t^{log}=x_t+\varepsilon_t.
\]
If $\rho(F)<1$, then $x_t-\mu=\sum_{j=0}^{\infty}F^j\eta_{t-j}$ is the unique stationary solution.

\section*{Mapping back to $A_t$}
Let $B_t=\exp(\operatorname{smat}(\hat x_t))$, $C=T^{-1}\sum_tB_t$, and $A_t=C^{-1/2}B_tC^{-1/2}$. Then $A_t\succ0$ and $T^{-1}\sum_tA_t=I_R$.

\section*{Robustness diagnostics}
For alternative shock definition $v$, compare $V_0$ and $V_v$ through principal angles $\theta_j=\arccos\sigma_j(V_0'V_v)$. This is invariant to sign and rotations within the retained subspace. Also compare $\rho_5$, $\tau_t=\operatorname{tr}(A_t)/5$, top amplification months, and basis-specific diagonal paths after basis matching.

\end{document}
"""
TEX_PATH.write_text(latex, encoding='utf-8')

# CSV tables
with open(TABLES_DIR / 'notation_dictionary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['symbol', 'definition'])
    writer.writerows(notation_rows)
with open(TABLES_DIR / 'proposition_inventory.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['number', 'label', 'claim'])
    writer.writerows(proposition_rows)

# -----------------------------
# PDF generation for appendix
# -----------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=10))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontSize=13.5, leading=16, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontSize=11.5, leading=14, spaceBefore=6, spaceAfter=4))
styles.add(ParagraphStyle(name='BodyX', parent=styles['BodyText'], fontSize=9.2, leading=11.5, spaceAfter=5))
styles.add(ParagraphStyle(name='SmallX', parent=styles['BodyText'], fontSize=7.7, leading=9, spaceAfter=2))
styles.add(ParagraphStyle(name='ProofX', parent=styles['BodyText'], fontSize=9.0, leading=11.3, leftIndent=0.15*inch, rightIndent=0.05*inch, spaceAfter=5))
styles.add(ParagraphStyle(name='CaptionX', parent=styles['BodyText'], fontSize=7.8, leading=9, alignment=TA_CENTER, spaceAfter=5))


def para(text, style='BodyX'):
    return Paragraph(html.escape(text).replace('\n', '<br/>'), styles[style])


def codeblock(text):
    # ReportLab Preformatted does not auto-wrap; wrap long lines lightly.
    lines = []
    for line in text.strip('\n').splitlines():
        if len(line) <= 92:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=92, subsequent_indent='    '))
    return Preformatted('\n'.join(lines), ParagraphStyle(
        name='CodeBlock', fontName='Courier', fontSize=7.8, leading=9.2,
        leftIndent=0.15*inch, rightIndent=0.05*inch, spaceBefore=3, spaceAfter=6,
        backColor=colors.whitesmoke
    ))


def add_section(story, title, body):
    story.append(Paragraph(html.escape(title), styles['H1x']))
    chunks = body.strip().split('\n\n')
    for chunk in chunks:
        ch = chunk.strip('\n')
        if not ch:
            continue
        # Treat indented blocks or equation-heavy chunks as preformatted.
        if ch.startswith('    ') or ('=' in ch and '\n' in ch and len(ch) < 1300 and any(line.strip().startswith(('T','K','G','A','x','P','S','J','z','q','beta','sigma','sum','rho','theta','delta','u')) for line in ch.splitlines())):
            story.append(codeblock(ch))
        else:
            story.append(para(ch))


def table_from_rows(headers, rows, col_widths=None, font=7.0):
    data = [headers] + [list(r) for r in rows]
    if col_widths is None:
        col_widths = [1.2*inch, 5.7*inch]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONT', (0,0), (-1,0), 'Helvetica-Bold', font),
        ('FONT', (0,1), (-1,-1), 'Helvetica', font),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    return tbl

APPENDIX_BUILD = MATH_DIR / 'mathematical_algorithm_appendix.pdf'
doc = SimpleDocTemplate(str(APPENDIX_BUILD), pagesize=letter, rightMargin=0.55*inch, leftMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
story = []
story.append(Paragraph('Mathematical Appendix', styles['TitleCenter']))
story.append(Paragraph('Top-five monetary-policy OVK algorithm: notation, derivations, proofs, and operational formulas', styles['TitleCenter']))
story.append(para('This appendix is part of the full appended results pack. It formalizes the algorithm used to construct the LP score surfaces, the average operator-valued kernel, the top-five basis, the log-Euclidean state-space A_t process, uncertainty bands, and robustness diagnostics.'))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
story.append(Spacer(1, 0.08*inch))
story.append(Paragraph('Notation dictionary', styles['H1x']))
story.append(table_from_rows(['Symbol', 'Definition'], notation_rows, col_widths=[1.25*inch, 5.65*inch], font=6.8))
story.append(PageBreak())
story.append(Paragraph('Proposition inventory', styles['H1x']))
story.append(table_from_rows(['No.', 'Label', 'Claim'], proposition_rows, col_widths=[0.35*inch, 1.65*inch, 4.9*inch], font=6.9))
story.append(PageBreak())

# Add sections from md_sections by extracting titles and bodies.
for idx, sec in enumerate(md_sections, start=1):
    title_line, body = sec.split('\n\n', 1)
    title = title_line.replace('## ', '')
    add_section(story, title, body)
    # Page breaks after major clusters.
    if idx in {4, 8, 12, 15, 18, 22}:
        story.append(PageBreak())

story.append(Paragraph('Appendix source files', styles['H1x']))
story.append(para('The ZIP pack includes this appendix as PDF, Markdown, and LaTeX source, plus CSV files for the notation dictionary and proposition inventory. It also includes all data and code used to generate the empirical pack.'))
story.append(codeblock('math/mathematical_algorithm_appendix.pdf\nmath/mathematical_algorithm_appendix.md\nmath/mathematical_algorithm_appendix.tex\nmath/tables/notation_dictionary.csv\nmath/tables/proposition_inventory.csv\ncode/create_full_math_appendix_pack.py'))

doc.build(story)
shutil.copy2(APPENDIX_BUILD, APPENDIX_PDF)
shutil.copy2(APPENDIX_BUILD, REPORTS_DIR / 'mathematical_algorithm_appendix.pdf')

# Copy this script into code directory for auditability.
SCRIPT_SRC = Path(os.environ.get('OVK_MATH_SCRIPT', '/mnt/data/create_full_math_appendix_pack.py'))
if SCRIPT_SRC.exists():
    shutil.copy2(SCRIPT_SRC, CODE_DIR / 'create_full_math_appendix_pack.py')

# -----------------------------
# Combine existing report + appendix
# -----------------------------
writer = PdfWriter()
for pdf in [OLD_REPORT, APPENDIX_BUILD]:
    reader = PdfReader(str(pdf))
    for page in reader.pages:
        writer.add_page(page)
with open(COMBINED_PDF, 'wb') as f:
    writer.write(f)
shutil.copy2(COMBINED_PDF, REPORTS_DIR / 'monthly_ovk_top5_full_appended_with_math_report.pdf')

# -----------------------------
# Update README and manifest
# -----------------------------
readme_path = NEW_PACK / 'README.md'
old_readme = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''
append = f"""

## Mathematical appendix added

This version appends a full mathematical appendix to the prior top-five robustness results pack. The appendix operationalizes the algorithm with notation, derivations, propositions, proof sketches, and formulas for:

- residualized local-projection score surfaces;
- construction and positive-definiteness of the operator-valued kernel;
- top-five finite-rank eigendecomposition and score whitening;
- dynamic positive-definite kernel path K_t = V_R Lambda_R^{1/2} A_t Lambda_R^{1/2} V_R';
- shrinkage SPD proxy G_t and log-Euclidean state-space model;
- stationarity of the VAR(1) log-state evolution;
- Kalman filter and Rauch-Tung-Striebel smoother recursions;
- mapping from latent log state back to positive-definite A_t;
- uncertainty bands for IRFs, eigenvalues, bases, tau_t, and A_t diagonals;
- robustness diagnostics across MP_median with fallback, MP_pm only, and manually aggregated event-level shocks.

New files:

```text
reports/monthly_ovk_top5_full_appended_with_math_report.pdf
reports/mathematical_algorithm_appendix.pdf
math/mathematical_algorithm_appendix.pdf
math/mathematical_algorithm_appendix.md
math/mathematical_algorithm_appendix.tex
math/tables/notation_dictionary.csv
math/tables/proposition_inventory.csv
code/create_full_math_appendix_pack.py
```
"""
readme_path.write_text(old_readme + append, encoding='utf-8')

# Rebuild manifest.
manifest = []
for f in sorted(NEW_PACK.rglob('*')):
    if f.is_file():
        rel = f.relative_to(NEW_PACK)
        manifest.append({
            'relative_path': str(rel),
            'bytes': f.stat().st_size,
            'category': str(rel).split(os.sep)[0],
        })
manifest_path = NEW_PACK / 'file_manifest.csv'
with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
    writer_csv = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'category'])
    writer_csv.writeheader()
    writer_csv.writerows(manifest)
shutil.copy2(manifest_path, BASE / 'monthly_ovk_top5_full_math_file_manifest.csv')

# -----------------------------
# Zip full pack
# -----------------------------
if OUT_ZIP.exists():
    OUT_ZIP.unlink()
with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(NEW_PACK.rglob('*')):
        z.write(f, arcname=f.relative_to(NEW_PACK))

# Also expose appendix md/tex/pdf top-level names for convenience.
shutil.copy2(MD_PATH, BASE / 'monthly_ovk_top5_math_appendix.md')
shutil.copy2(TEX_PATH, BASE / 'monthly_ovk_top5_math_appendix.tex')

# -----------------------------
# Render final combined PDF and create contact sheet
# -----------------------------
if RENDER_DIR.exists():
    shutil.rmtree(RENDER_DIR)
RENDER_DIR.mkdir(parents=True, exist_ok=True)
# Use existing render script when available.
render_script = Path(os.environ.get('OVK_RENDER_PDF_SCRIPT', '/home/oai/skills/pdfs/scripts/render_pdf.py'))
if render_script.exists():
    os.system(f"python {render_script} {COMBINED_PDF} --out_dir {RENDER_DIR} --dpi 110")
imgs = []
for p in sorted(RENDER_DIR.glob('page-*.png')):
    try:
        imgs.append((p, Image.open(p).convert('RGB')))
    except Exception:
        pass
if imgs:
    thumbs = []
    for i, (p, im) in enumerate(imgs, start=1):
        th = ImageOps.contain(im, (190, 250))
        canvas = Image.new('RGB', (210, 280), 'white')
        canvas.paste(th, ((210 - th.width)//2, 24))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), f'p. {i}', fill='black')
        thumbs.append(canvas)
    cols = 4
    rows = (len(thumbs) + cols - 1)//cols
    sheet = Image.new('RGB', (cols*210, rows*280), 'white')
    for i, th in enumerate(thumbs):
        sheet.paste(th, ((i % cols)*210, (i // cols)*280))
    sheet.save(CONTACT_SHEET, quality=88)
    shutil.copy2(CONTACT_SHEET, REPORTS_DIR / 'monthly_ovk_top5_full_appended_with_math_contact_sheet.jpg')
    # Rezip to include contact sheet.
    with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(NEW_PACK.rglob('*')):
            z.write(f, arcname=f.relative_to(NEW_PACK))

# Final report counts.
combined_pages = len(PdfReader(str(COMBINED_PDF)).pages)
appendix_pages = len(PdfReader(str(APPENDIX_BUILD)).pages)
print('Created combined PDF:', COMBINED_PDF, COMBINED_PDF.stat().st_size, 'pages', combined_pages)
print('Created appendix PDF:', APPENDIX_PDF, APPENDIX_PDF.stat().st_size, 'pages', appendix_pages)
print('Created zip:', OUT_ZIP, OUT_ZIP.stat().st_size)
print('Created contact sheet:', CONTACT_SHEET, CONTACT_SHEET.stat().st_size if CONTACT_SHEET.exists() else 'missing')
