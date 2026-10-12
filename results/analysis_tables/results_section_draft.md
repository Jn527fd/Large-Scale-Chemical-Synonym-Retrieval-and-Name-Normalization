## Results Draft

We evaluated chemical synonym retrieval on the non-singleton subset of the CSV, using each synonym surface form as a query and all other names as retrieval candidates. The main metrics were MRR, HitRate@k, Precision@k, and RecallFrac@k, with self-matches removed before scoring.



Among the evaluated systems, the best overall run was `sapbert_contrastive` with MRR=0.708, HitRate@1=0.630, HitRate@5=0.814, and HitRate@10=0.853 over 3720979 queries.

Bucketed diagnostics by synonym-set size should be used to determine whether gains are concentrated in compounds with many aliases or persist for compounds with only two or three known names. Robustness curves should be used to describe how retrieval quality degrades under corrupted query names.
