# Step 6 Error Taxonomy Summary

This summary analyzes final held-out test predictions from the corrected external laboratory RefMet normalization task.

## Overall Counts

- Test queries analyzed: 59,163
- Exact top-1 successes: 39,568 (66.88%)
- Identity-aware top-1 successes: 41,567 (70.26%)
- Exact top-1 failures: 19,595 (33.12%)
- Gold exact target in top 10 but not rank 1: 9,926 (16.78%)
- Gold exact target absent from top 10 or unranked: 9,669 (16.34%)

## Primary Error Categories

| Category | Cases | Percent of exact failures | Identity-correct rate | Mean confidence margin |
| --- | ---: | ---: | ---: | ---: |
| candidate_generation_miss_or_low_rank | 8,890 | 45.37% | 0.00% | 0.0148 |
| reranking_error_gold_in_top10 | 8,182 | 41.76% | 0.00% | 0.0153 |
| identity_match_not_exact_label | 1,999 | 10.20% | 100.00% | 0.0141 |
| reranking_error_identity_in_top10 | 524 | 2.67% | 0.00% | 0.0096 |

## Interpretation

The taxonomy separates ranking failures from candidate-generation failures. Cases where the gold target appears in the top 10 but not at rank 1 are especially important because they indicate that the semantic retrieval stage found the right neighborhood, but the final reranking decision selected a competing label. Cases where the gold target is absent from the top 10 are more likely to require better candidate generation, more complete target dictionaries, synonym expansion, or stronger normalization rules.

The heuristic labels should be treated as a first-pass taxonomy. The manual review sheet contains sampled false positives, high-confidence true positives, and gold-in-top-10 reranking errors for human validation before publication.

## Generated Artifacts

- Full row-level taxonomy: `results/error_taxonomy.csv`
- Run-local taxonomy: `results/external_lab_refmet_model_comparison_reproducible/step6_error_taxonomy.csv`
- Manual review sheet: `results/external_lab_refmet_model_comparison_reproducible/step6_manual_review_sheet.csv`
- Category counts: `results/external_lab_refmet_model_comparison_reproducible/step6_error_taxonomy_counts.csv`
- Representative examples: `results/external_lab_refmet_model_comparison_reproducible/step6_representative_examples.csv`
- Category plot: `results/external_lab_refmet_model_comparison_reproducible/plots/step6_error_taxonomy_counts.png`
