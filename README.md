# OVK_Volterra

This repository contains the monetary-policy local-projection OVK code and a
memory-target extension for the paper project.

Run the memory-target application:

```bash
python scripts/run_monetary_lp_memory_targets.py \
  --targets diagonal_old hac_filtered hilbert_volterra \
  --hac-lags 12 \
  --memory-half-lives 3 12 36 \
  --signature-gamma 0.05 \
  --base-inner reference_soft \
  --feature-bandwidth median \
  --strict-past \
  --bootstrap-draws 0 \
  --bootstrap-workers 0 \
  --add-rotation-diagnostics \
  --output-dir outputs/monetary_lp_memory_targets
```

Use `--bootstrap-workers 0` for automatic process parallelism when
`--bootstrap-draws` is positive, or `--bootstrap-workers 1` for serial
bootstrap draws.

Refresh only the route-rotation diagnostics from saved route operators:

```bash
python scripts/add_route_rotation_diagnostics.py \
  --comparison-dir outputs/monetary_lp_memory_targets/comparison \
  --targets-dir outputs/monetary_lp_memory_targets \
  --routes diagonal_old hac_filtered_L12 hilbert_volterra_L12_gamma005_memory_3_12_36 \
  --rotation-reference pooled \
  --rotation-lambda-min 1e-2 \
  --rotation-lambda-max 1e2 \
  --rotation-lambda-count 41 \
  --min-rotation-anisotropy 0.05
```

Run the focused tests:

```bash
python -m pytest tests/test_time_series_targets.py tests/test_hilbert_volterra.py tests/test_route_rotation.py
```
