from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cid_bucket_label(size: int) -> str:
    if size == 2:
        return "2"
    if 3 <= size <= 5:
        return "3-5"
    if 6 <= size <= 10:
        return "6-10"
    if 11 <= size <= 20:
        return "11-20"
    return "21+"


def bucket_metrics_to_frame(
    bucket_metrics: dict[str, dict[str, Any]],
    baseline: str,
    model_key: str | None,
    run_name: str,
) -> pd.DataFrame:
    rows = []
    order = {"2": 0, "3-5": 1, "6-10": 2, "11-20": 3, "21+": 4}
    for bucket, metrics in bucket_metrics.items():
        rows.append(
            {
                "run_name": run_name,
                "baseline": baseline,
                "model_key": model_key or baseline,
                "bucket": bucket,
                "bucket_order": order.get(bucket, 99),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["bucket_order", "bucket"]).reset_index(drop=True)


def metric_row(
    metrics: dict[str, Any],
    baseline: str,
    model_key: str | None,
    run_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_name": run_name,
        "baseline": baseline,
        "model_key": model_key or baseline,
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def make_examples(
    q_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    names: list[str],
    ids: np.ndarray,
    n_examples: int = 10,
    k: int = 10,
    seed: int = 0,
    query_names: list[str] | None = None,
) -> pd.DataFrame:
    if len(q_idx) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    sample_pos = rng.choice(len(q_idx), size=min(n_examples, len(q_idx)), replace=False)
    records = []

    for pos in sample_pos:
        qi = int(q_idx[pos])
        q_name = query_names[pos] if query_names is not None else names[qi]
        q_cid = ids[qi]

        kept = 0
        for raw_j in neighbor_indices[pos]:
            j = int(raw_j)
            if j < 0 or j == qi:
                continue
            kept += 1
            records.append(
                {
                    "query_row": qi,
                    "query_name": q_name,
                    "query_cid": q_cid,
                    "neighbor_rank": kept,
                    "neighbor_row": j,
                    "neighbor_name": names[j],
                    "neighbor_cid": ids[j],
                    "is_relevant": int(ids[j] == q_cid),
                }
            )
            if kept >= k:
                break

    return pd.DataFrame(records)
