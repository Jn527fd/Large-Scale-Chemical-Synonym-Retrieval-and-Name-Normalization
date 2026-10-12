# Day 2 Reranker Ablation and Bootstrap Summary

Generated from `notebooks/external_lab_refmet_day2_reranker_ablations_bootstrap.ipynb`.

## Reranker Ablation Outputs

- Metrics: `results/external_lab_refmet_model_comparison_reproducible/reranker_ablation_metrics.csv`
- Per-query ranks: `results/external_lab_refmet_model_comparison_reproducible/reranker_ablation_per_query_ranks.csv`
- Plot: `results/external_lab_refmet_model_comparison_reproducible/reranker_ablation_plot.png`

## Best Test Ablation

The highest exact MRR on the `test` split was `full_hybrid` with MRR=0.7296, Hit@1=66.88%, Hit@10=83.66%, and Hit@256=88.59%.

- `embedding_only`: exact MRR=0.4668, exact Hit@1=38.86%
- `lexical_only_best_weights`: exact MRR=0.7196, exact Hit@1=66.04%
- `dense_plus_exact`: exact MRR=0.4743, exact Hit@1=39.89%
- `dense_plus_soft_lexical`: exact MRR=0.7135, exact Hit@1=64.02%
- `full_hybrid`: exact MRR=0.7296, exact Hit@1=66.88%

## Bootstrap Outputs

- Confidence intervals: `results/bootstrap_confidence_intervals.csv`
- Metric differences: `results/bootstrap_metric_differences.csv`
- Difference plot: `results/bootstrap_metric_differences.png`

## Notes

- Bootstrap intervals are computed by resampling per-query ranks with replacement.
- The external-lab task supports bootstrap because per-query ranks are saved.
- Part One internal benchmark bootstrap still requires per-query rank artifacts; aggregate metric tables alone are not sufficient for valid bootstrap resampling.
- Hit@256 is candidate-pool recall and is unchanged by reranking methods that only reorder the same top-256 candidates.
