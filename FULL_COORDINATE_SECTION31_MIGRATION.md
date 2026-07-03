# Section 3.1 Full-Coordinate Migration Note

Generated: 2026-06-23.

## Scope

Figures 1-3 have been moved off the legacy retained-rank covariance-PCA path.
The new Section 3.1 production path uses the full 125-coordinate LP working
grid, soft ridge whitening, and no spectral cutoff.

The existing dense Kalman state-space code cannot support a 7,875-dimensional
log-SPD vech state without forming dense state covariance matrices. The
implemented backend is therefore the explicitly separate full-coordinate
temporal-kernel backend:

```text
K_hat[t] = sum_s w[t,s] outer(chi_s, chi_s)
```

The weights are normalized, nonnegative, symmetric, and average-preserving, so
the fitted path averages back to C_hat up to numerical tolerance. Figure bands
are moving-block bootstrap bands, not FFBS state uncertainty.

## Old Versus New Formulas

Legacy Figure 1:

```text
tau_rank5[t] = trace(A_rank5[t]) / 5
```

New Figure 1:

```text
C_hat = mean_t outer(chi_t, chi_t)
D_rho = C_hat + rho I
d_rho = trace(C_hat solve(D_rho, I))
A_hat[t] = D_rho^(-1/2) K_hat[t] D_rho^(-1/2)
tau_soft[t] = trace(K_hat[t] solve(D_rho, I)) / d_rho
```

Legacy Figure 2A:

```text
S_t = A_rank5[t] / tau_rank5[t]
C_t = V_5 S_t V_5'
log_relative = log(diag(C_t) / diag(V_5 V_5'))
```

New Figure 2A:

```text
cell_amp[t,m] = K_hat[t,m,m] / C_hat[m,m]
cell_shape[t,m] = log(cell_amp[t,m] / tau_soft[t])
```

Cells whose C_hat[m,m] is below `OVK_FULL_COORDINATE_CELL_VARIANCE_TOL` are
flagged in `publication_grade_full_coordinate_cell_shape_allocations.csv`.

Legacy Figure 2B:

```text
retained-basis raw shares and concentration diagnostics
```

New Figure 2B:

```text
block_amp[t,B] = trace(S_B K_hat[t] S_B') / trace(S_B C_hat S_B')
block_shape[t,B] = block_amp[t,B] / tau_soft[t]
```

Blocks are macro outcomes, financial outcomes, horizons 0-3, horizons 4-12,
and horizons 13-24. The benchmark is 1. The concentration panel is labeled as
a finite-working-grid display.

Legacy Figure 3:

```text
rank-R proxy-IV tau path
retained residual-response energy
```

New Figure 3:

```text
tau_soft[t] from the full-coordinate proxy-IV K_hat path
chi_proxy[r] = (M_r / kappa_hat) u_proxy[r]
kappa_hat = mean_r(M_r X_r)
E_r = (M_r / kappa_hat)^2
R_r = u_proxy[r]' solve(D_rho, u_proxy[r]) / d_rho
score_energy_soft[r] = chi_proxy[r]' solve(D_rho, chi_proxy[r]) / d_rho = E_r R_r
K_hat[t] = sum_r w_hat[t,r] chi_proxy[r] chi_proxy[r]'
tau_soft[t] = sum_r w_hat[t,r] E_r R_r
E_bar[t] = sum_r w_hat[t,r] E_r
R_tilde[t] = sum_r w_hat[t,r] E_r R_r / E_bar[t]
exposure_factor[t] = E_bar[t] / E_ref
residual_factor[t] = E_ref R_tilde[t]
tau_soft[t] = exposure_factor[t] * residual_factor[t]
```

The source score identity, temporal-weight reconstruction, and target-factor
product identity are checked numerically and reported in
`iv_tau_driver_diagnostic_summary.json`. The audit table is
`iv_tau_factor_decomposition_audit.csv`.

## Saved Outputs

Publication Figure 1 and Figure 2 outputs:

- `results/publication_grade_ovk/outputs/charts/01_full_coordinate_tau_soft_block_bootstrap_bands.png`
- `results/publication_grade_ovk/outputs/charts/02a_full_coordinate_cell_shape_heatmap_atlas.png`
- `results/publication_grade_ovk/outputs/charts/02b_full_coordinate_block_shape_paths.png`
- `results/publication_grade_ovk/outputs/tables/publication_grade_full_coordinate_covariance_components.npz`
- `results/publication_grade_ovk/outputs/tables/publication_grade_full_coordinate_cell_shape_allocations.csv`
- `results/publication_grade_ovk/outputs/tables/publication_grade_full_coordinate_block_shape_paths.csv`
- `results/publication_grade_ovk/outputs/tables/publication_grade_full_coordinate_diagnostics.csv`
- `results/reports/publication_grade_ovk_report.pdf`

Proxy-IV Figure 3 outputs:

- `results/iv_ovk/figures/iv_tau_driver_diagnostic.png`
- `results/iv_ovk/tables/iv_full_coordinate_covariance_components.npz`
- `results/iv_ovk/tables/iv_tau_driver_diagnostic_monthly.csv`
- `results/iv_ovk/tables/iv_tau_factor_decomposition_audit.csv`
- `results/iv_ovk/tables/iv_tau_driver_diagnostic_summary.json`
- `results/reports/iv_ovk_report.pdf`

Legacy top-five compatibility exports are skipped by default and require:

```text
OVK_WRITE_LEGACY_TOP5_COMPAT=1
```

## Runtime And Memory Report

Local regeneration was run with small bootstrap counts for acceptance speed:

- Publication report: 290.9 seconds with 8 Section 3.1 block-bootstrap draws,
  rank comparison bootstraps disabled, one worker, cache disabled.
- IV report: 97.2 seconds with 4 IV bootstrap draws and 8 nested bootstrap
  draws.

Full-coordinate publication arrays:

- `chi`: shape `(375, 125)`, 0.36 MiB.
- `K_hat`: shape `(375, 125, 125)`, 44.70 MiB.
- `A_hat`: shape `(375, 125, 125)`, 44.70 MiB.
- `temporal_weights`: shape `(375, 375)`, 1.07 MiB.
- compressed NPZ: 75.26 MiB.

Full-coordinate IV arrays:

- `chi_proxy`: shape `(374, 125)`, 0.36 MiB.
- `K_hat`: shape `(374, 125, 125)`, 44.58 MiB.
- `A_hat`: shape `(374, 125, 125)`, 44.58 MiB.
- compressed NPZ: 73.72 MiB.

The requested 7,875-dimensional log-SPD state is not instantiated. No dense
7,875 by 7,875 covariance or precision matrix is formed.

## Table I Audit

Table I is not silently changed. The separate nested mean-covariance workflow
still depends on the old low-dimensional evaluation machinery:

- `code/ovk_nested_workflow.py` builds a finite orthonormal evaluation basis
  `W` with `d_eval=min(10, M)`.
- Predictive comparison is performed in `W` coordinates, not on the full
  125-coordinate covariance operator.
- The moving-center survival diagnostic forms `z_m1` and `z_m3` by projecting
  through the leading covariance eigenbasis and `HEADLINE_R`.
- The M3 moving-center specification therefore still uses a covariance mode
  in the low-dimensional macro pipeline.

Those dependencies are now documented here for separate review.
