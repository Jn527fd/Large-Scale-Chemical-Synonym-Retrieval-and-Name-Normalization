# Broad Embedding-Model Benchmark Addendum

## Task

All models are evaluated on the corrected closed-set task: predict the provided refmet_name target from the metabolite_name query. The target vocabulary, split, exact scoring, identity-aware scoring, and hybrid reranker feature set are shared with the final reproducible evaluation above.

## Main Findings

- Best embedding-only exact MRR: sapbert_contrastive = 0.4668.
- Best hybrid-reranked exact MRR: sapbert_contrastive = 0.7296.
- Best hybrid-reranked exact Hit@1: sapbert_contrastive = 66.88%.

## Interpretation

Embedding-only metrics measure representation quality. Hybrid reranked metrics measure how much performance can be recovered by adding deterministic lexical and chemistry-aware features over the same top-k candidate pool.

## Publication Solidification Steps

- Report exact and identity-aware metrics separately.
- Include non-identical-name metrics as the harder normalization setting.
- Use the saved success/failure examples and per-model predictions for qualitative error analysis.
- Freeze the model list, seed, data snapshot, and dependency versions before the final publication rerun.