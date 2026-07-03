# SEC Robustness Run Order

This robustness check is now included in the default `ovk_pipeline.py run-all` sequence. It reads the publication-grade OVK outputs and does not replace or overwrite the baseline estimator. Use this standalone order only when rerunning the SEC pack by itself.

1. From the repository root, install the existing requirements if needed:

   `python -m pip install -r code/requirements.txt`

2. Run the SEC robustness pack:

   `python code/run_sec_robustness.py --bootstrap-draws 1000 --clean`

   If the pipeline outputs were written somewhere other than `results`, add `--results-dir path\to\outdir`.

3. Run the numerical checks:

   `python tests/test_sec_robustness.py`

Primary outputs:

- `monthly_ovk_top5_with_SEC_robustness_report.pdf`
- `monthly_ovk_top5_with_SEC_robustness_full_pack.zip`
- `monthly_ovk_top5_with_SEC_robustness_contact_sheet.jpg`
- `sec_robustness_results/reports/sec_robustness_appendix.pdf`
- `sec_robustness_results/tables/*.csv`

The SEC comparison baseline is the publication-grade `tau_t` path and publication-grade rank-five shock-robustness paths. Deprecated legacy top-five estimator outputs are not used as the benchmark.
