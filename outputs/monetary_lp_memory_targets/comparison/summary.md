# Monetary LP Memory-Target OVK Summary

Empirical monetary-policy data were available.

Score source: `C:\Users\ahanson8\Documents\OVK_Volterra\OVK_Volterra\data_processed\processed_panel_three_shock_definitions.csv`.

## 1. What changed from the old target

- Old: `K_old(s) = sum_t w(s,t) psi_t psi_t'`.
- HAC: `K_HAC(s) = sum_t w(s,t) Z_t Z_t'`, with Bartlett-filtered scores and `L=12`.
- Nonlinear: `K_VSig(s) = sum_t w_NL(s,t) Z_t Z_t'`, where Volterra history features alter the state weights.

## 2. Why HAC_filtered is HAC-aware

With `Z_t = (1/sqrt(L+1)) sum_ell psi_{t-ell}`, expanding `Z_t Z_t'` adds all cross-period products `psi_{t-ell} psi_{t-m}'`. Grouping by lag gives Bartlett weights `1 - h/(L+1)` without forming a giant lag-stack covariance.

## 3. What nonlinear Volterra features add

The Volterra path `Phi_t` summarizes ordered nonlinear interactions in a rank-5 projected score history. The nonlinear target compares months using both old calendar proximity and similarity of `Phi_t`, while the final operator remains `p x p`.

## 4. Empirical comparison

Top old diagonal months:
- 2019-06-01: tau_soft=2.191
- 2019-07-01: tau_soft=2.184
- 2019-09-01: tau_soft=2.167
- 2019-10-01: tau_soft=2.157
- 2019-05-01: tau_soft=2.148

Top HAC-filtered months:
- 2019-10-01: tau_soft=1.514
- 2020-01-01: tau_soft=1.507
- 2019-09-01: tau_soft=1.505
- 2019-11-01: tau_soft=1.499
- 2020-02-01: tau_soft=1.497

Top nonlinear Volterra months:
- 2020-03-01: tau_soft=1.776
- 2020-02-01: tau_soft=1.761
- 2020-01-01: tau_soft=1.659
- 2000-12-01: tau_soft=1.644
- 2019-10-01: tau_soft=1.637

Tau-path correlations:

|  | tau_soft_diagonal_old | tau_soft_hac_filtered | tau_soft_volterra_nonlinear |
| --- | --- | --- | --- |
| tau_soft_diagonal_old | 1.000 | 0.736 | 0.742 |
| tau_soft_hac_filtered | 0.736 | 1.000 | 0.988 |
| tau_soft_volterra_nonlinear | 0.742 | 0.988 | 1.000 |

Months newly highlighted by HAC or Volterra top-10 lists:

| target | rank | date | tau_soft |
| --- | --- | --- | --- |
| hac_filtered | 2 | 2020-01-01 | 1.507 |
| hac_filtered | 5 | 2020-02-01 | 1.497 |
| hac_filtered | 6 | 2019-12-01 | 1.496 |
| volterra_nonlinear | 1 | 2020-03-01 | 1.776 |
| volterra_nonlinear | 2 | 2020-02-01 | 1.761 |
| volterra_nonlinear | 3 | 2020-01-01 | 1.659 |
| volterra_nonlinear | 4 | 2000-12-01 | 1.644 |
| volterra_nonlinear | 7 | 2000-11-01 | 1.560 |
| volterra_nonlinear | 9 | 2001-01-01 | 1.487 |
| volterra_nonlinear | 10 | 2019-12-01 | 1.482 |

Block-probe CSVs in each target directory report outcome-group and horizon-bucket differences.

## 5. Diagnostics

- diagonal_old: min eig -2.669e-14, mean tau error 3.386e-14, ESS median 14.0
- hac_filtered: min eig -2.659e-14, mean tau error 8.282e-14, ESS median 14.0
- volterra_nonlinear: min eig -2.062e-13, mean tau error 6.728e-14, ESS median 13.2

## 6. Caveats

These are diagnostic moment fields, not new time-varying causal effects. The HAC target is a filtered long-run exposure target, not a conventional Newey-West standard error. The nonlinear Volterra target depends on projection rank, half-lives, feature bandwidth, and truncation level.
