# Lab CID Mapping Review

## Summary

The exact normalized-name bridge successfully mapped a substantial external evaluation subset, but it should be treated as a high-confidence partial labeling step rather than complete coverage of the lab file.

## Key Results

- Lab rows processed: 443,540
- Lab name values used: 755,091
- Unique normalized lab names targeted: 467,549
- Reference rows scanned: 114,674,081
- Matched rows: 233,002 (52.532%)
- Unique CID rows: 216,792 (48.878%)
- Ambiguous CID rows: 16,210 (3.655%)
- Unmatched rows: 210,538 (47.468%)
- Unique assigned CIDs in clean labeled set: 81,957

## Source Column Behavior

- `refmet_name` values matched: 154,603 / 311,553 (49.62%)
- `metabolite_name` values matched: 119,600 / 443,538 (26.96%)
- Matched rows from `refmet_name` only: 113,402
- Matched rows from `metabolite_name` only: 78,399
- Matched rows from both columns: 41,201

The `refmet_name` column is the stronger exact-match source, which makes sense because it appears to contain more standardized/common names. The `metabolite_name` column still adds many labels, but exact matching struggles more with long systematic names and formatting variants.

## Ambiguity

Most ambiguous rows are only mildly ambiguous:

- 2 candidate CIDs: 12,881 rows
- 3 candidate CIDs: 1,960 rows
- 4-5 candidate CIDs: 938 rows
- More than 5 candidate CIDs: 431 rows

These rows should not be used as gold labels without disambiguation, but they are small enough to review manually or handle with a conservative secondary rule.

## Conclusion

The external lab file can now produce a clean, high-confidence evaluation set of 216,792 rows covering 81,957 unique CIDs. This is large enough to evaluate cross-source synonym retrieval, but the exact-matched labels should be used carefully to avoid trivial string-match leakage. For model evaluation, query with the lab name while excluding identical normalized surface forms from the retrieval corpus, so success means retrieving another synonym for the same CID rather than the exact same string.

The 210,538 unmatched rows are not failures of the retrieval approach. They are mostly cases where exact string overlap is insufficient, especially long chemical names, formatting variants, stereochemical variants, and names missing `refmet_name`. These are the right candidates for semantic retrieval, robustness analysis, or manual labeling.
