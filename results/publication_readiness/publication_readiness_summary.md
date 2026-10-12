# Publication Readiness Audit Summary

Generated: 2026-07-27T17:22:27.830230

## Dataset
- Raw lab rows: 443540
- Rows with both names: 311553
- Deduplicated query-target pairs: 295812
- Unique RefMet target labels: 68988
- Split counts: {'train': 177487, 'test': 59163, 'val': 59162}

## Main Test Results
- Embedding-only exact MRR: 0.4668
- Embedding-only exact Hit@1: 38.8604%
- Reranker exact MRR: 0.7296
- Reranker exact Hit@1: 66.8796%
- Reranker exact Hit@10: 83.6570%

## Strict Non-Self Evaluation
- Gold RefMet equals query surface in test set: 4444
- Top-1 prediction equals query surface: 4482
- Strict non-identical test queries: 54719
- Strict exact MRR after removing query-identical candidates: 0.7058
- Strict exact Hit@1 after removing query-identical candidates: 64.23%
- Strict exact Hit@10 after removing query-identical candidates: 82.33%
- Strict identity-aware Hit@1 after removing query-identical candidates: 67.86%

## Conservative Row-Exclusion Evaluation
- Removed gold self-match rows: 4444
- Removed rows where original top-1 equals query surface: 4482
- Removed non-identical gold rows where original top-1 equals query surface: 38
- Kept conservative test queries: 54681
- Conservative reranker exact MRR: 0.7080
- Conservative reranker exact Hit@1: 64.23%
- Conservative reranker exact Hit@10: 82.34%
- Conservative reranker identity-aware Hit@1: 67.86%

## Manual Review Sheet
- Rows generated for manual review: 300
- File: `manual_review_sheet.csv`
- Fill manual columns, save as `manual_review_sheet_labeled.csv`, then rerun the final manual-summary cell.

## Key Output Files
- `publication_dataset_manifest.csv`
- `primary_metric_summary.csv`
- `self_match_leakage_summary.csv`
- `strict_no_self_surface_metrics.csv`
- `strict_no_self_surface_predictions.csv`
- `conservative_row_exclusion_summary.csv`
- `conservative_no_query_surface_metrics.csv`
- `conservative_no_query_surface_predictions.csv`
- `excluded_self_or_query_surface_rows.csv`
- `conservative_no_query_surface_failure_examples.csv`
- `near_query_surface_competitor_examples.csv`
- `confidence_threshold_table.csv`
- `refmet_cid_mapping_audit.csv`
- `error_taxonomy_summary.csv`
- `manual_review_sheet.csv`
- `representative_success_failure_examples.csv`
- `reranker_rescue_examples.csv`
- `candidate_generation_failure_examples.csv`

## Remaining Human Work
- Manually label the review sheet for chemical acceptability, same-compound correctness, and failure category.
- Use manually confirmed examples in the paper and advisor update slides.
- Treat unseen-compound generalization as a future experiment unless a lightweight diagnostic can be produced from existing cached outputs.