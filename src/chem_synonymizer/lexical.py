from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def fit_char_tfidf(names: list[str], config: dict[str, Any]) -> tuple[TfidfVectorizer, Any]:
    ngram_min = int(config.get("ngram_min", 3))
    ngram_max = int(config.get("ngram_max", 5))
    vectorizer = TfidfVectorizer(
        analyzer=config.get("analyzer", "char_wb"),
        ngram_range=(ngram_min, ngram_max),
        min_df=config.get("min_df", 1),
        lowercase=False,
        dtype=np.float32,
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        norm="l2",
    )
    matrix = vectorizer.fit_transform(names)
    return vectorizer, matrix


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + int(ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def edit_similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - (levenshtein_distance(a, b) / denom)


def retrieve_tfidf_neighbors(
    names: list[str],
    vectorizer: TfidfVectorizer,
    matrix: Any,
    k_retrieve: int,
    batch_size: int = 512,
    query_indices: np.ndarray | None = None,
    query_texts: list[str] | None = None,
    edit_distance_rerank: bool = False,
    candidate_multiplier: int = 3,
    tfidf_weight: float = 0.8,
    n_jobs: int = -1,
) -> np.ndarray:
    if query_indices is None and query_texts is None:
        raise ValueError("Pass query_indices or query_texts.")

    n_corpus = matrix.shape[0]
    if n_corpus == 0:
        return np.empty((0, 0), dtype=np.int64)

    candidate_count = int(k_retrieve)
    if edit_distance_rerank:
        candidate_count = max(candidate_count, int(k_retrieve) * max(int(candidate_multiplier), 1))
    candidate_count = min(n_corpus, max(candidate_count, 1))

    nn = NearestNeighbors(
        n_neighbors=candidate_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    nn.fit(matrix)

    if query_texts is not None:
        n_queries = len(query_texts)
    else:
        n_queries = len(query_indices)

    all_indices = []
    for start in tqdm(range(0, n_queries, batch_size), desc="TF-IDF search"):
        end = min(start + batch_size, n_queries)
        if query_texts is not None:
            q_texts = query_texts[start:end]
            q_matrix = vectorizer.transform(q_texts)
        else:
            batch_indices = query_indices[start:end]
            q_texts = [names[int(i)] for i in batch_indices]
            q_matrix = matrix[batch_indices]

        distances, indices = nn.kneighbors(q_matrix, return_distance=True)
        if edit_distance_rerank:
            indices = _rerank_with_edit_distance(
                names=names,
                query_texts=q_texts,
                indices=indices,
                tfidf_similarities=1.0 - distances,
                k_retrieve=k_retrieve,
                tfidf_weight=float(tfidf_weight),
            )
        else:
            indices = indices[:, :k_retrieve]
        all_indices.append(indices.astype(np.int64, copy=False))

    return np.vstack(all_indices)


def _rerank_with_edit_distance(
    names: list[str],
    query_texts: list[str],
    indices: np.ndarray,
    tfidf_similarities: np.ndarray,
    k_retrieve: int,
    tfidf_weight: float,
) -> np.ndarray:
    tfidf_weight = min(max(tfidf_weight, 0.0), 1.0)
    edit_weight = 1.0 - tfidf_weight
    reranked = np.full((indices.shape[0], min(k_retrieve, indices.shape[1])), -1, dtype=np.int64)

    for i, q_text in enumerate(query_texts):
        candidate_rows = indices[i]
        candidate_scores = []
        for j, tfidf_sim in zip(candidate_rows, tfidf_similarities[i]):
            candidate_name = names[int(j)]
            score = (tfidf_weight * float(tfidf_sim)) + (edit_weight * edit_similarity(q_text, candidate_name))
            candidate_scores.append(score)
        order = np.argsort(-np.asarray(candidate_scores, dtype=np.float32))
        chosen = candidate_rows[order][: reranked.shape[1]]
        reranked[i, : len(chosen)] = chosen

    return reranked
