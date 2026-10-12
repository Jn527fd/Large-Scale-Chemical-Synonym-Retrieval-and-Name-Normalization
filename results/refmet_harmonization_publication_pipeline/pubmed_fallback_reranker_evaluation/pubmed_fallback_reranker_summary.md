# PubMed/PubChem Fallback Reranker Evaluation

This evaluation reuses the saved PubMed/PubChem fallback top-k neighbor arrays. It does not re-embed strings and does not rerun FAISS.

## Evaluation Subset
- Source run: `refmet_harmonization_publication_pipeline`
- Cases evaluated: fallback-routed rows where the gold RefMet target was not in the RefMet top 10 and the gold RefMet target had at least one PubMed/PubChem CID.
- Held-out test cases: 10,284
- Candidate pool per query: top 256

## Test Metrics
- **embedding_original_order**: MRR=0.0614, Hit@1=4.16%, Hit@5=8.24%, Hit@10=9.94%, Hit@50=14.52%, Hit@256=19.37%
- **refmet_weights_rerank**: MRR=0.1284, Hit@1=11.28%, Hit@5=14.61%, Hit@10=15.78%, Hit@50=18.04%, Hit@256=19.37%
- **pubmed_tuned_rerank**: MRR=0.1288, Hit@1=11.36%, Hit@5=14.59%, Hit@10=15.67%, Hit@50=18.01%, Hit@256=19.37%

## Interpretation
If Hit@256 is low, the saved PubMed/PubChem fallback search did not retrieve the true same-CID candidate often enough, and no reranker can recover those missing cases. If Hit@256 is high but Hit@10 is low, reranking is the main bottleneck.