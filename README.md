# OVK_Volterra

This repository contains the monetary-policy local-projection OVK code and a
memory-target extension for the paper project.

Run the memory-target application:

```bash
python scripts/run_monetary_lp_memory_targets.py \
  --targets diagonal_old hac_filtered volterra_nonlinear \
  --hac-lags 12 \
  --volterra-rank 5 \
  --volterra-level 2 \
  --volterra-half-lives 3 12 36 \
  --bootstrap-draws 0 \
  --output-dir outputs/monetary_lp_memory_targets
```

Run the focused tests:

```bash
python -m pytest tests/test_time_series_targets.py
```
