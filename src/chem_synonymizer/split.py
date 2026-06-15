from __future__ import annotations

import numpy as np
import pandas as pd


def sample_query_indices(n_rows: int, eval_frac: float, seed: int) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=np.int64)
    frac = min(max(float(eval_frac), 0.0), 1.0)
    n_queries = min(max(int(round(n_rows * frac)), 1), n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n_queries, replace=False).astype(np.int64))


def split_surface_forms(df: pd.DataFrame, eval_frac: float, seed: int) -> dict[str, np.ndarray]:
    eval_idx = sample_query_indices(len(df), eval_frac, seed)
    mask = np.ones(len(df), dtype=bool)
    mask[eval_idx] = False
    return {"train_idx": np.where(mask)[0].astype(np.int64), "eval_idx": eval_idx}


def split_unseen_compounds(df: pd.DataFrame, eval_frac: float, seed: int) -> dict[str, np.ndarray]:
    cids = df["cid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    n_eval_cids = min(max(int(round(len(cids) * eval_frac)), 1), len(cids))
    eval_cids = set(rng.choice(cids, size=n_eval_cids, replace=False).tolist())
    eval_mask = df["cid"].isin(eval_cids).to_numpy()
    return {
        "train_idx": np.where(~eval_mask)[0].astype(np.int64),
        "eval_idx": np.where(eval_mask)[0].astype(np.int64),
    }
