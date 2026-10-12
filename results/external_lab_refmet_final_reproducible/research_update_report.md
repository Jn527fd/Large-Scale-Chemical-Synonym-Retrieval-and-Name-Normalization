# External Lab RefMet Final Reproducible Evaluation Report

## Summary

This run evaluates the corrected lab-name task: predict `refmet_name` from `metabolite_name` using the existing tuned SapBERT contrastive model plus a lightweight hybrid reranker. The notebook also performs deterministic row-level CID mapping against the PubMed/PubChem synonym reference and saves all outputs into one run directory.

## Data Preparation

- Raw lab rows: 443,540
- Rows with both `metabolite_name` and `refmet_name`: 311,553
- Deduplicated `metabolite_name -> refmet_name` pairs: 295,812
- Unique RefMet target labels: 68,988
- Split counts: {'train': 177487, 'test': 59163, 'val': 59162}

Rows were removed from the closed-set evaluation if either side of the pair normalized to an empty string. Duplicate normalized query-target pairs were collapsed before splitting. Identical normalized `metabolite_name` and `refmet_name` pairs are retained in the full evaluation but are also reported separately through non-identical metrics.

## Deterministic CID Mapping

The deterministic bridge normalizes lab names from both configured columns and scans the large reference synonym CSV in chunks. A lab row is labeled `exact_unique_cid` when all matched normalized names collapse to one CID, `exact_ambiguous_cid` when matched names point to multiple CIDs, and `unmatched` when no normalized reference name is found.

- Unique-CID lab rows: 216,792
- Ambiguous-CID lab rows: 16,210
- Unmatched lab rows: 210,538
- Unique assigned CIDs: 81,957

## Model Evaluation

Input: `metabolite_name`.
Expected output: the correct closed-set `refmet_name`. Identity-aware scoring additionally counts predictions that map to the same PubChem CID or a conservative strong alias when mapping evidence exists.

Held-out test results from this run:

- Embedding-only exact MRR: 0.4668
- Embedding-only exact Hit@1: 38.86%
- Reranker exact MRR: 0.7297
- Reranker exact Hit@1: 66.89%
- Reranker identity-aware MRR: 0.7554
- Reranker identity-aware Hit@1: 70.26%

## Success and Failure Examples

Representative examples are saved in `final_success_failure_examples.csv`. The table includes exact top-1 successes, identity-only successes, reranker-rescued cases, high-confidence wrong predictions, cases where the correct label is present in top-10 but not ranked first, and cases where the exact label is not found in the top-256 candidate pool.

## Interpretation

The embedding model functions as a candidate generator: it often places the correct target somewhere in the top-k set. The reranker improves the order of candidates by combining embedding similarity with lexical and chemical-name signals. It cannot recover a correct answer that was never retrieved into the candidate pool, so top-k recovery remains an upper bound on reranking performance.

## Publication-Readiness Checklist

1. Freeze this notebook, run configuration, split seed, and result folder as the canonical lab-name evaluation.
2. Repeat the run or bootstrap the held-out test predictions to add uncertainty intervals.
3. Manually audit high-confidence disagreements, especially lipid shorthand, stereochemistry, and abbreviation cases.
4. Decide whether exact RefMet label prediction or identity-aware same-compound scoring should be the primary endpoint.
5. If possible, validate the RefMet-to-PubChem mapping rules against an external/official mapping table.