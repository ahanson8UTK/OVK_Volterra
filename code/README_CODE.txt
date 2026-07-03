Run the full pipeline from the workspace root:

python code\ovk_pipeline.py run-all --data-zip data_raw\data.zip --out-dir results --overwrite --headline-outcomes base5 --sf-fed-surprises data_raw\external\sf_fed_monetary_policy_surprises.xlsx

The headline output is `reports/publication_grade_ovk_report.pdf`, which upgrades the rank-five response-score covariance estimator with robust likelihood/EM estimation, structured transition shrinkage, simulation-smoother bands, full-pipeline bootstrap uncertainty, rank sensitivity, same-sample expectations comparisons, outcome-trace standardization, placebo shocks, policy/CBI splits, smooth-LP robustness, SF Fed/Bauer-Swanson appendix shocks, and episode-level spike uncertainty. The legacy top-five estimator is deprecated in the CLI; `publication_grade_ovk` now also writes the familiar top-five tables/charts/report names using the upgraded base-five headline estimator. The nested mean/covariance workflow is produced as the empirical-anchor robustness report and uses the same upgraded rank-five log-Euclidean A_t core for the survival diagnostic. SEC robustness now runs by default and appends the SEC appendix into `monthly_ovk_top5_with_SEC_robustness_report.pdf`.

Dependencies: numpy, pandas, matplotlib, scipy, scikit-learn, Pillow, reportlab, and pypdf.

Default draw counts: publication-grade FFBS state draws = 1000, publication-grade full-pipeline moving-block bootstrap draws = 1000, nested mean/covariance CI bootstrap draws = 2000, and SEC moving-block bootstrap draws = 1000.

Speed controls: pass `--workers N` to parallelize publication-grade refits, `--sec-bootstrap-draws N` to change the default 1000 SEC bootstrap draws, `--benchmark-workers` to calibrate a worker count when `--workers` is omitted, `--cache-dir .ovk_cache` to reuse binary model caches and complete step artifacts across runs, and `--no-cache` for a clean uncached run.
