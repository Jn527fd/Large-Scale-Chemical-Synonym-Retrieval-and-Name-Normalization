from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
from tqdm.auto import tqdm

from .diagnostics import cid_bucket_label
from .utils import ensure_contiguous_f32


def init_metric_accumulators(k_values: Iterable[int]) -> dict:
    k_values = [int(k) for k in k_values]
    return {
        "n_queries": 0,
        "mrr_sum": 0.0,
        "hitrate": {k: 0 for k in k_values},
        "precision_sum": {k: 0.0 for k in k_values},
        "recall_frac_sum": {k: 0.0 for k in k_values},
    }


def update_metric_accumulators(
    acc: dict,
    filtered_neighbors: list[int],
    qid: object,
    available_relevant: int,
    ids: np.ndarray,
    k_values: Iterable[int],
) -> None:
    acc["n_queries"] += 1
    relevant_flags = np.array([1 if ids[j] == qid else 0 for j in filtered_neighbors], dtype=np.int32)

    first_rel = np.where(relevant_flags == 1)[0]
    if len(first_rel) > 0:
        acc["mrr_sum"] += 1.0 / (int(first_rel[0]) + 1)

    cumsum_rel = np.cumsum(relevant_flags)
    for k in k_values:
        k = int(k)
        if len(cumsum_rel) >= k:
            rel_k = int(cumsum_rel[k - 1])
        elif len(cumsum_rel) > 0:
            rel_k = int(cumsum_rel[-1])
        else:
            rel_k = 0

        acc["hitrate"][k] += int(rel_k > 0)
        acc["precision_sum"][k] += rel_k / k
        acc["recall_frac_sum"][k] += (rel_k / available_relevant) if available_relevant > 0 else 0.0


def finalize_metric_accumulators(acc: dict, k_values: Iterable[int]) -> dict:
    n = max(int(acc["n_queries"]), 1)
    out = {
        "n_queries": int(acc["n_queries"]),
        "MRR": float(acc["mrr_sum"] / n),
    }
    for k in k_values:
        k = int(k)
        out[f"HitRate@{k}"] = float(acc["hitrate"][k] / n)
        out[f"Precision@{k}"] = float(acc["precision_sum"][k] / n)
        out[f"RecallFrac@{k}"] = float(acc["recall_frac_sum"][k] / n)
    return out


def filter_self_neighbors(neighbors: Iterable[int], query_row: int, max_k: int) -> list[int]:
    filtered: list[int] = []
    for raw_j in neighbors:
        j = int(raw_j)
        if j < 0 or j == query_row:
            continue
        filtered.append(j)
        if len(filtered) >= max_k:
            break
    return filtered


def filtered_neighbor_matrix(q_idx: np.ndarray, neighbor_indices: np.ndarray, max_k: int) -> np.ndarray:
    """Remove each query's self row and keep the first max_k neighbors, vectorized by rank."""
    q_idx = np.asarray(q_idx, dtype=np.int64)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    out = np.full((len(q_idx), int(max_k)), -1, dtype=np.int64)
    counts = np.zeros(len(q_idx), dtype=np.int16)

    for rank in range(neighbors.shape[1]):
        candidates = neighbors[:, rank]
        keep = (candidates >= 0) & (candidates != q_idx) & (counts < max_k)
        rows = np.where(keep)[0]
        if len(rows) == 0:
            continue
        out[rows, counts[rows]] = candidates[rows]
        counts[rows] += 1
        if np.all(counts >= max_k):
            break

    return out


def evaluate_neighbors(
    q_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    ids: np.ndarray,
    cid_size_arr: np.ndarray,
    k_values: Iterable[int],
) -> dict:
    k_values = [int(k) for k in k_values]
    max_k = max(k_values)
    filtered = filtered_neighbor_matrix(q_idx, neighbor_indices, max_k)
    valid = filtered >= 0
    safe_filtered = np.where(valid, filtered, 0)
    qids = ids[q_idx]
    relevant = (ids[safe_filtered] == qids[:, None]) & valid

    n = max(len(q_idx), 1)
    any_rel = relevant.any(axis=1)
    first_rel = np.argmax(relevant, axis=1)
    reciprocal_ranks = np.where(any_rel, 1.0 / (first_rel + 1), 0.0)
    cumsum_rel = np.cumsum(relevant, axis=1)
    available = np.maximum(cid_size_arr[q_idx].astype(np.float64) - 1.0, 1.0)

    out = {
        "n_queries": int(len(q_idx)),
        "MRR": float(reciprocal_ranks.sum() / n),
    }
    for k in k_values:
        rel_k = cumsum_rel[:, k - 1] if cumsum_rel.shape[1] >= k else cumsum_rel[:, -1]
        out[f"HitRate@{k}"] = float((rel_k > 0).sum() / n)
        out[f"Precision@{k}"] = float((rel_k / k).sum() / n)
        out[f"RecallFrac@{k}"] = float((rel_k / available).sum() / n)
    return out


def evaluate_neighbors_by_bucket(
    q_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    ids: np.ndarray,
    cid_size_arr: np.ndarray,
    k_values: Iterable[int],
    labeler: Callable[[int], str] = cid_bucket_label,
) -> dict[str, dict]:
    k_values = [int(k) for k in k_values]
    buckets: dict[str, dict] = {}
    bucket_labels = np.array([labeler(int(size)) for size in cid_size_arr[q_idx]], dtype=object)
    for bucket in ["2", "3-5", "6-10", "11-20", "21+"]:
        mask = bucket_labels == bucket
        if not np.any(mask):
            continue
        buckets[bucket] = evaluate_neighbors(
            q_idx=q_idx[mask],
            neighbor_indices=neighbor_indices[mask],
            ids=ids,
            cid_size_arr=cid_size_arr,
            k_values=k_values,
        )
    return buckets


def search_faiss_neighbors(
    index: object,
    query_embeddings: np.ndarray,
    k_retrieve: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    batches = []
    t0 = time.perf_counter()
    for start in tqdm(range(0, len(query_embeddings), batch_size), desc="FAISS search"):
        end = min(start + batch_size, len(query_embeddings))
        q_batch = ensure_contiguous_f32(query_embeddings[start:end])
        _, indices = index.search(q_batch, int(k_retrieve))
        batches.append(indices.astype(np.int64, copy=False))
    elapsed = time.perf_counter() - t0
    if not batches:
        return np.empty((0, k_retrieve), dtype=np.int64), elapsed
    return np.vstack(batches), elapsed


def pretty_metrics(metrics: dict, k_values: Iterable[int]) -> str:
    lines = [f"n_queries: {metrics['n_queries']}", f"MRR: {metrics['MRR']:.6f}"]
    for k in k_values:
        lines.append(
            f"k={int(k):>2} | "
            f"HitRate={metrics[f'HitRate@{int(k)}']:.6f} | "
            f"Precision={metrics[f'Precision@{int(k)}']:.6f} | "
            f"RecallFrac={metrics[f'RecallFrac@{int(k)}']:.6f}"
        )
    return "\n".join(lines)
