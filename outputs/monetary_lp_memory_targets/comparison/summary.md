# Monetary LP Memory-Target OVK Summary

Empirical monetary-policy data were available.

Score source: `C:\Users\ahanson8\Documents\OVK_Volterra\OVK_Volterra\data_processed\processed_panel_three_shock_definitions.csv`.

## 1. What changed from the old target

- Old: `K_old(s) = sum_t w(s,t) psi_t psi_t'`.
- HAC: `K_HAC(s) = sum_t w_time(s,t) Z_t Z_t'`, with Bartlett-filtered scores and `L=12`.
- Hilbert-Volterra: `K_HV(s) = sum_t w_HV(s,t) Z_t Z_t'`, where `w_HV` uses a normalized infinite-level Fock Gram matrix.
- The old finite `Phi_t` prototype is not used by the main empirical target: no rank `r`, no level `M`, and no PCA feature truncation.

## 2. Mathematical target

`w_HV(s,t)` is proportional to the old calendar-time kernel times `exp(-d_Fock(s,t)^2/(2 h_Fock^2))`. The distance is computed from the normalized Fock Gram matrix, not from explicit tensor features.

## 3. Why the target is Hilbert-space consistent

The score vector `psi_t` lives in the original coefficient/influence space `H` represented here by `R^125`. The nonlinear history object lives in the weighted Fock space `F_beta(H)`, but only its Gram matrix is required. The final `K_HV(s)` remains a positive `p x p` operator on `H`, so the old ridge-soft relative-moment machinery applies unchanged.

## 4. Why HAC_filtered is HAC-aware

With `Z_t = (1/sqrt(L+1)) sum_ell psi_{t-ell}`, expanding `Z_t Z_t'` adds all cross-period products `psi_{t-ell} psi_{t-m}'`. Grouping by lag gives Bartlett weights `1 - h/(L+1)` without forming a giant lag-stack covariance.

## 5. What nonlinear Volterra/Fock state similarity adds

The Hilbert-Volterra target borrows from months with similar ordered score-history geometry. The recursion includes all finite-sample tensor orders; `gamma=0.05` controls high-order weighting, not truncation. Memory half-lives are `[3.0, 12.0, 36.0]` with `equal` weights.

## 6. Empirical comparison

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

Top Hilbert-Volterra months:
- 2019-10-01: tau_soft=1.533
- 2020-01-01: tau_soft=1.527
- 2019-09-01: tau_soft=1.518
- 2020-02-01: tau_soft=1.517
- 2019-11-01: tau_soft=1.510

Tau-path correlations:

|  | tau_soft_diagonal_old | tau_soft_hac_filtered | tau_soft_hilbert_volterra |
| --- | --- | --- | --- |
| tau_soft_diagonal_old | 1.000 | 0.736 | 0.743 |
| tau_soft_hac_filtered | 0.736 | 1.000 | 1.000 |
| tau_soft_hilbert_volterra | 0.743 | 1.000 | 1.000 |

Months newly highlighted by HAC or Hilbert-Volterra top-10 lists:

| target | rank | date | tau_soft |
| --- | --- | --- | --- |
| hac_filtered | 2 | 2020-01-01 | 1.507 |
| hac_filtered | 5 | 2020-02-01 | 1.497 |
| hac_filtered | 6 | 2019-12-01 | 1.496 |
| hilbert_volterra | 2 | 2020-01-01 | 1.527 |
| hilbert_volterra | 4 | 2020-02-01 | 1.517 |
| hilbert_volterra | 6 | 2019-12-01 | 1.508 |
| hilbert_volterra | 10 | 2020-03-01 | 1.482 |

Block-probe CSVs in each target directory report outcome-group and horizon-bucket differences.

## 7. Diagnostics

- diagonal_old: min eig -2.669e-14, mean tau error 3.386e-14, ESS median 14.0
- hac_filtered: min eig -2.659e-14, mean tau error 8.282e-14, ESS median 14.0
- hilbert_volterra: min eig -2.108e-14, mean tau error 2.776e-14, ESS median 13.1, kappa_norm min eig 6.084e-07

## 8. Caveats

These are diagnostic moment fields, not new time-varying causal effects. The HAC target is a filtered long-run exposure target, not a conventional Newey-West standard error. Hilbert-Volterra similarity depends on memory kernel, gamma, base inner product, and smoothing bandwidth; conventional HAC inference remains separate unless that inferential covariance target is explicitly defined.
## Route rotation diagnostics

The `tau_soft` comparison reports route scale. These diagnostics compare route-specific relative-moment shapes after placing all three routes in a common ridge-soft geometry and removing each route's common soft scale.

The probed Yosida alignment uses

`Omega = <Q^(1/2) Y_r Q^(1/2), Q^(1/2) Y_r' Q^(1/2)>_HS / (norms)`

and reports `1 - average_lambda Omega`. The full soft probe is the full-coordinate soft comparison; the macro, financial, and horizon probes show whether route rotation is concentrated in economically meaningful blocks.

Mean Yosida rotation:

| pair_label | Full soft | Macro | Financial | Short horizons | Medium horizons | Long horizons |
| --- | --- | --- | --- | --- | --- | --- |
| Diagonal vs HAC | 0.332 | 0.306 | 0.278 | 0.212 | 0.264 | 0.294 |
| HAC vs Hilbert-Volterra | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 |
| Diagonal vs Hilbert-Volterra | 0.334 | 0.308 | 0.281 | 0.214 | 0.267 | 0.295 |

P90 Yosida rotation:

| pair_label | Full soft | Macro | Financial | Short horizons | Medium horizons | Long horizons |
| --- | --- | --- | --- | --- | --- | --- |
| Diagonal vs HAC | 0.365 | 0.340 | 0.315 | 0.269 | 0.302 | 0.339 |
| HAC vs Hilbert-Volterra | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| Diagonal vs Hilbert-Volterra | 0.369 | 0.343 | 0.319 | 0.272 | 0.305 | 0.342 |

The commutator diagnostic uses `[Abar_r, Abar_r'] = Abar_r Abar_r' - Abar_r' Abar_r` after scale removal. It separates directional noncommutativity from pure eigenvalue/shape differences and should be interpreted only when both routes have nontrivial anisotropy relative to the common soft-reference shape.

Commutator summary:

| pair_label | p90_commutator_index | max_commutator_index | date_max_commutator | share_dates_interpretable |
| --- | --- | --- | --- | --- |
| Diagonal vs HAC | 0.209 | 0.316 | 2011-09-01 | 1.000 |
| HAC vs Hilbert-Volterra | 0.026 | 0.033 | 2020-05-01 | 1.000 |
| Diagonal vs Hilbert-Volterra | 0.213 | 0.317 | 2011-09-01 | 1.000 |

Reference mode: `pooled`; common dates: 359; `rho_star=3.431e-08`.

These are diagnostic shape/rotation comparisons, not time-varying causal effects and not conventional HAC standard errors.
