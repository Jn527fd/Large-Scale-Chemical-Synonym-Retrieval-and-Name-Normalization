# RefMet Harmonization Publication Pipeline Summary

## Objective
Evaluate a tuned SapBERT retrieval model plus a lightweight hybrid reranker for mapping laboratory metabolite names to existing RefMet names, while producing curation routes for cases that may require new RefMet entries or manual review.

## Data
- Lab CSV: `/content/drive/MyDrive/ChemSynonymizer/chem_synonymizer/data/allmetnames_062626.csv`
- Lab rows: 443,540
- Rows with metabolite_name: 443,540
- Rows with refmet_name: 311,553
- Evaluation rows mapped to target vocabulary: 311,553
- RefMet target vocabulary size: 68,988
- PubMed/PubChem condensed rows: 24306210

## Methods
- RefMet target names are embedded and searched exactly with FAISS to avoid compression loss in the main closed-set evaluation.
- The large PubMed/PubChem synonym database is condensed by CID before embedding, keeping a deterministic top-N set of representative names per CID.
- A second-stage reranker combines SapBERT score, rank prior, compact exact match, sequence similarity, character n-gram overlap, token overlap, lipid shorthand consistency, and stereochemistry penalties.
- Final results include full and strict non-identical/no-query-surface views so exact surface-form matches do not overstate model behavior.
- Curation routing uses validation-calibrated confidence tiers: high-confidence RefMet auto-accept targeting 90% validation precision, medium-confidence RefMet review, and PubMed/PubChem fallback for queries whose RefMet candidate set falls below the validation Hit@10 target of 80%.

## Main Metrics
- **embedding_only_full_test**: MRR=0.1450, Hit@1=9.62%, Hit@10=24.25%, Hit@256=59.76%, n=62,311
- **embedding_only_nonidentical_test**: MRR=0.1302, Hit@1=8.34%, Hit@10=22.17%, Hit@256=59.91%, n=56,710
- **hybrid_rerank_full_test**: MRR=0.5012, Hit@1=45.34%, Hit@10=58.43%, Hit@256=59.76%, n=62,311
- **hybrid_rerank_nonidentical_test**: MRR=0.4932, Hit@1=44.07%, Hit@10=58.45%, Hit@256=59.91%, n=56,710
- **hybrid_rerank_nonidentical_no_query_surface_test**: MRR=0.4934, Hit@1=44.11%, Hit@10=58.45%, Hit@256=59.91%, n=56,710

## Interpretation
The closed-set results measure how well the pipeline maps a submitted metabolite name to an existing RefMet common name. The strict results should be treated as the most defensible publication estimate because they remove exact metabolite/refmet label matches and exclude returned candidates that reproduce the query surface. PubMed/PubChem fallback results should be interpreted as curation evidence rather than final labels.

## Key Output Files
- `evaluation_metrics.csv`
- `bootstrap_confidence_intervals.csv`
- `confidence_tier_summary.csv`
- `confidence_tier_diagnostics.csv`
- `refmet_top10_candidates_wide.csv`
- `refmet_predictions_with_decisions.csv`
- `refmet_plus_pubmed_curation_routes.csv`
- `human_review_case_samples.csv`
- `human_review_case_samples_strict_no_query_surface.csv`

## Next Steps
1. Have RefMet/domain experts review routed examples, especially high-confidence wrong predictions and possible-new-entry candidates.
2. Calibrate acceptance thresholds with expert review labels.
3. Report both full and strict metrics in the paper.
4. Treat PubMed/PubChem fallback as evidence for curation, not an automatic RefMet assignment.