"""Run Day 2 reranker ablations and bootstrap confidence intervals.

This script is intended to be executed from
notebooks/external_lab_refmet_day2_reranker_ablations_bootstrap.ipynb.
It does not re-encode embeddings. It loads the canonical external-lab RefMet
artifacts and evaluates reranker ablations over the saved top-256 candidate pool.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
    print("matplotlib is not installed; CSV outputs will still be generated, but plots will be skipped.")

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    print("tqdm is not installed; progress bars will be disabled.")


def env_int(name: str, default: int | None) -> int | None:
    value = globals().get(name, os.getenv(name))
    if value in (None, "", "None"):
        return default
    return int(value)


def resolve_project_root() -> Path:
    root = Path.cwd().resolve()
    if root.name == "notebooks":
        return root.parent
    if (root / "results").exists() and (root / "scripts").exists():
        return root
    if (root.parent / "results").exists() and (root.parent / "scripts").exists():
        return root.parent
    return root


PROJECT_ROOT = resolve_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
RUN_DIR = RESULTS_DIR / "external_lab_refmet_model_comparison_reproducible"
ARTIFACTS_DIR = RUN_DIR / "artifacts"
MODEL_BENCHMARK_DIR = RUN_DIR / "model_benchmarks"

POOL_K = env_int("DAY2_POOL_K", 256)
RERANK_BATCH_SIZE = env_int("DAY2_RERANK_BATCH_SIZE", 384)
MAX_QUERIES_PER_SPLIT = env_int("DAY2_MAX_QUERIES_PER_SPLIT", None)
BOOTSTRAP_N = env_int("DAY2_BOOTSTRAP_N", 1000)
BOOTSTRAP_SEED = env_int("DAY2_BOOTSTRAP_SEED", 17)
BOOTSTRAP_SPLIT = globals().get("DAY2_BOOTSTRAP_SPLIT", os.getenv("DAY2_BOOTSTRAP_SPLIT", "test"))
EVAL_SPLITS = ["val", "test"]
METRIC_KS = [1, 5, 10, 25, 50, 100, 256]
DAY2_DIR = RUN_DIR if MAX_QUERIES_PER_SPLIT is None else RUN_DIR / f"day2_smoke_test_{MAX_QUERIES_PER_SPLIT}"
DAY2_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Canonical run dir:", RUN_DIR)
print("POOL_K:", POOL_K)
print("RERANK_BATCH_SIZE:", RERANK_BATCH_SIZE)
print("MAX_QUERIES_PER_SPLIT:", MAX_QUERIES_PER_SPLIT)
print("BOOTSTRAP_N:", BOOTSTRAP_N)

required = {
    "query_pairs": ARTIFACTS_DIR / "v4_query_pairs.csv",
    "target_dictionary": ARTIFACTS_DIR / "v4_refmet_label_dictionary.csv",
    "neighbor_indices": ARTIFACTS_DIR / "closed_refmet_neighbor_indices.npy",
    "neighbor_scores": ARTIFACTS_DIR / "closed_refmet_neighbor_scores.npy",
    "best_weights": RUN_DIR / "v4_best_reranker_weights.json",
    "final_predictions": RUN_DIR / "v4_final_test_predictions.csv",
    "model_summary": MODEL_BENCHMARK_DIR / "model_benchmark_summary.csv",
}
missing = [name for name, path in required.items() if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required canonical artifacts: " + ", ".join(missing))

with required["best_weights"].open("r", encoding="utf-8") as f:
    best_weights = json.load(f)

lab_pairs = pd.read_csv(required["query_pairs"]).sort_values("query_id").reset_index(drop=True)
targets = pd.read_csv(required["target_dictionary"]).sort_values("target_id").reset_index(drop=True)

if not np.array_equal(lab_pairs["query_id"].to_numpy(), np.arange(len(lab_pairs))):
    raise ValueError("Expected query_id to align with row order in v4_query_pairs.csv")
if not np.array_equal(targets["target_id"].to_numpy(), np.arange(len(targets))):
    raise ValueError("Expected target_id to align with row order in v4_refmet_label_dictionary.csv")

neighbor_indices = np.load(required["neighbor_indices"], mmap_mode="r")
neighbor_scores = np.load(required["neighbor_scores"], mmap_mode="r")

if neighbor_indices.shape[0] != len(lab_pairs):
    raise ValueError(
        f"Neighbor row count {neighbor_indices.shape[0]} does not match query pairs {len(lab_pairs)}"
    )
if neighbor_indices.shape[1] < POOL_K:
    raise ValueError(f"POOL_K={POOL_K} exceeds saved candidate count {neighbor_indices.shape[1]}")

split_indices = {
    split: np.flatnonzero(lab_pairs["split"].astype(str).to_numpy() == split).astype(np.int64)
    for split in sorted(lab_pairs["split"].astype(str).unique())
}
if MAX_QUERIES_PER_SPLIT is not None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    split_indices = {
        split: np.sort(rng.choice(idx, size=min(len(idx), int(MAX_QUERIES_PER_SPLIT)), replace=False)).astype(
            np.int64
        )
        for split, idx in split_indices.items()
    }

print("Query pairs:", lab_pairs.shape)
print("Targets:", targets.shape)
print("Neighbor arrays:", neighbor_indices.shape, neighbor_scores.shape)
print("Split sizes:", {k: int(len(v)) for k, v in split_indices.items()})
print("Best weights:", json.dumps(best_weights, indent=2))

GREEK_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ω": "omega",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    for src, dst in GREEK_MAP.items():
        text = text.replace(src, dst)
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def compact_key(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def pipe_split(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    out = set()
    for part in str(value).split("|"):
        part = part.strip()
        if not part or part.lower() == "nan":
            continue
        if part.endswith(".0") and part[:-2].isdigit():
            part = part[:-2]
        out.add(part)
    return out


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = compact_key(text)
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def token_set(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", normalize_text(text)) if tok}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def sequence_ratio(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if len(a) > 120 or len(b) > 120:
        return jaccard(char_ngrams(a, 3), char_ngrams(b, 3))
    return SequenceMatcher(None, a, b).ratio()


def lipid_signature(text: str) -> tuple[str, str, str] | None:
    s = normalize_text(text)
    m = re.search(r"\b([a-z]{1,6})\s*[\( ]?\s*(\d{1,2})\s*:\s*(\d{1,2})\b", s)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3))


def stereo_signature(text: str) -> tuple[str, ...]:
    s = normalize_text(text)
    toks = re.findall(r"\b(?:cis|trans|alpha|beta)\b|\b[dl]-|\([+-]\)|\b[0-9]+[rs]\b|\b[0-9]+[ez]\b", s)
    return tuple(sorted(toks))


def text_profile(text: str) -> dict:
    norm = normalize_text(text)
    return {
        "norm": norm,
        "compact": compact_key(norm),
        "char3": char_ngrams(norm, 3),
        "tokens": token_set(norm),
        "lipid": lipid_signature(norm),
        "stereo": stereo_signature(norm),
    }


def lexical_feature_dict(query_profile: dict, candidate_profile: dict) -> dict:
    q = query_profile
    c = candidate_profile
    return {
        "normalized_exact": float(q["norm"] == c["norm"] and q["norm"] != ""),
        "compact_exact": float(q["compact"] == c["compact"] and q["compact"] != ""),
        "sequence_ratio": float(sequence_ratio(q["norm"], c["norm"])),
        "char3_jaccard": float(jaccard(q["char3"], c["char3"])),
        "token_jaccard": float(jaccard(q["tokens"], c["tokens"])),
        "lipid_signature_match": float(q["lipid"] is not None and q["lipid"] == c["lipid"]),
        "lipid_class_match": float(
            q["lipid"] is not None and c["lipid"] is not None and q["lipid"][0] == c["lipid"][0]
        ),
        "stereo_mismatch": float(bool(q["stereo"] and c["stereo"] and q["stereo"] != c["stereo"])),
    }


def strong_alias_match(a: str, b: str) -> bool:
    if normalize_text(a) == normalize_text(b):
        return True
    ca = compact_key(a)
    cb = compact_key(b)
    if ca and ca == cb and min(len(ca), len(cb)) >= 5:
        return True
    seq = sequence_ratio(a, b)
    tok = jaccard(token_set(a), token_set(b))
    ch = jaccard(char_ngrams(a, 3), char_ngrams(b, 3))
    return bool(seq >= 0.96 or (seq >= 0.90 and tok >= 0.80) or (ch >= 0.92 and tok >= 0.70))


def metrics_from_ranks(rank_arr: np.ndarray, prefix: str = "") -> dict:
    rank_arr = np.asarray(rank_arr, dtype=np.float64)
    finite = np.isfinite(rank_arr)
    n = len(rank_arr)
    out = {f"{prefix}MRR": float(np.sum(np.where(finite, 1.0 / rank_arr, 0.0)) / max(1, n))}
    for k in METRIC_KS:
        out[f"{prefix}HitRate@{k}"] = float(np.mean(rank_arr <= k)) if n else 0.0
    return out


def ranks_to_components(rank_arr: np.ndarray) -> dict[str, np.ndarray]:
    rank_arr = np.asarray(rank_arr, dtype=np.float64)
    finite = np.isfinite(rank_arr)
    return {
        "MRR": np.where(finite, 1.0 / rank_arr, 0.0),
        **{f"HitRate@{k}": (rank_arr <= k).astype(np.float64) for k in METRIC_KS},
    }


target_norms = targets["refmet_normalized"].fillna("").astype(str).tolist()
target_cids = targets["refmet_candidate_cids"].fillna("").astype(str).tolist()
target_cid_sets = [pipe_split(x) for x in target_cids]
gold_target_ids = lab_pairs["target_id"].to_numpy(dtype=np.int64)
query_norms = lab_pairs["metabolite_normalized"].fillna("").astype(str).tolist()

needed_rows = np.unique(np.concatenate([split_indices[s] for s in EVAL_SPLITS if s in split_indices]))
target_profiles = [text_profile(x) for x in tqdm(target_norms, desc="Precomputing target profiles")]
query_profiles = {int(i): text_profile(query_norms[int(i)]) for i in tqdm(needed_rows, desc="Precomputing query profiles")}

identity_correct_cache: dict[tuple[int, int], tuple[bool, str]] = {}


def identity_correct_pair(gold_id: int, candidate_id: int) -> tuple[bool, str]:
    if gold_id == candidate_id:
        return True, "exact_label"
    key = (int(gold_id), int(candidate_id))
    if key in identity_correct_cache:
        return identity_correct_cache[key]
    if target_cid_sets[gold_id] and target_cid_sets[candidate_id] and (target_cid_sets[gold_id] & target_cid_sets[candidate_id]):
        result = (True, "same_pubchem_cid")
    elif strong_alias_match(target_norms[gold_id], target_norms[candidate_id]):
        result = (True, "strong_alias")
    else:
        result = (False, "")
    identity_correct_cache[key] = result
    return result


def first_ranks_for_row(row_i: int, ordered_ids: np.ndarray) -> tuple[float, float]:
    gold = int(gold_target_ids[row_i])
    exact_rank = np.inf
    identity_rank = np.inf
    for rank0, target_id in enumerate(ordered_ids):
        target_id = int(target_id)
        if target_id == gold and not np.isfinite(exact_rank):
            exact_rank = rank0 + 1
        ok, _ = identity_correct_pair(gold, target_id)
        if ok and not np.isfinite(identity_rank):
            identity_rank = rank0 + 1
        if np.isfinite(exact_rank) and np.isfinite(identity_rank):
            break
    return exact_rank, identity_rank


FEATURE_NAMES = [
    "embedding_score_norm",
    "rank_prior",
    "normalized_exact",
    "compact_exact",
    "sequence_ratio",
    "char3_jaccard",
    "token_jaccard",
    "lipid_signature_match",
    "lipid_class_match",
    "stereo_mismatch",
]
DENSE = ["embedding_score_norm", "rank_prior"]
EXACT = ["normalized_exact", "compact_exact"]
SOFT_LEXICAL = ["sequence_ratio", "char3_jaccard", "token_jaccard"]
CHEM_HINTS = ["lipid_signature_match", "lipid_class_match", "stereo_mismatch"]


def clean_weight_dict(weights: dict) -> dict:
    return {name: float(weights.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}


def keep_only(base: dict, names: list[str], name: str) -> dict:
    out = {"name": name}
    for feature in FEATURE_NAMES:
        out[feature] = float(base.get(feature, 0.0)) if feature in names else 0.0
    return out


def drop_groups(base: dict, drop: list[str], name: str) -> dict:
    out = {"name": name}
    for feature in FEATURE_NAMES:
        out[feature] = 0.0 if feature in drop else float(base.get(feature, 0.0))
    return out


full_hybrid_weights = clean_weight_dict(best_weights)
full_hybrid_weights["normalized_exact"] = float(full_hybrid_weights.get("normalized_exact", 0.0))

ABLATIONS = [
    {"name": "embedding_only", **{f: 0.0 for f in FEATURE_NAMES}, "embedding_score_norm": 1.0},
    {
        "name": "exact_normalized_match_only",
        **{f: 0.0 for f in FEATURE_NAMES},
        "normalized_exact": 1.0,
        "compact_exact": 0.95,
        "rank_prior": 1e-4,
    },
    keep_only(full_hybrid_weights, EXACT + SOFT_LEXICAL + CHEM_HINTS, "lexical_only_best_weights"),
    keep_only(full_hybrid_weights, DENSE + EXACT, "dense_plus_exact"),
    keep_only(full_hybrid_weights, DENSE + SOFT_LEXICAL + ["stereo_mismatch"], "dense_plus_soft_lexical"),
    {"name": "full_hybrid", **full_hybrid_weights},
    drop_groups(full_hybrid_weights, EXACT, "full_hybrid_minus_exact"),
    drop_groups(full_hybrid_weights, SOFT_LEXICAL, "full_hybrid_minus_soft_lexical"),
    drop_groups(full_hybrid_weights, CHEM_HINTS, "full_hybrid_minus_chem_hints"),
]
ablation_weights = {row["name"]: clean_weight_dict(row) for row in ABLATIONS}
weights_table = pd.DataFrame(ABLATIONS).fillna(0.0)
weights_table.to_csv(DAY2_DIR / "reranker_ablation_weight_definitions.csv", index=False)


def build_feature_block(row_indices: np.ndarray, pool_k: int) -> tuple[np.ndarray, np.ndarray]:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    cand_ids = np.asarray(neighbor_indices[row_indices, :pool_k], dtype=np.int64)
    emb_scores = np.asarray(neighbor_scores[row_indices, :pool_k], dtype=np.float32)
    n, k = cand_ids.shape
    features = np.zeros((n, k, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, :, FEATURE_NAMES.index("embedding_score_norm")] = (emb_scores + 1.0) / 2.0
    features[:, :, FEATURE_NAMES.index("rank_prior")] = 1.0 / (np.arange(k, dtype=np.float32)[None, :] + 1.0)

    for local_i, row_i in enumerate(row_indices):
        qprof = query_profiles[int(row_i)]
        for j, target_id in enumerate(cand_ids[local_i]):
            if target_id < 0:
                continue
            feats = lexical_feature_dict(qprof, target_profiles[int(target_id)])
            for name, value in feats.items():
                features[local_i, j, FEATURE_NAMES.index(name)] = value
    return cand_ids, features


def score_features(features: np.ndarray, weights: dict) -> np.ndarray:
    w = np.array([float(weights.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)
    return np.tensordot(features, w, axes=([2], [0]))


def ranks_from_scores(cand_ids: np.ndarray, scores: np.ndarray, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exact_ranks = np.full(len(row_indices), np.inf, dtype=np.float64)
    identity_ranks = np.full(len(row_indices), np.inf, dtype=np.float64)
    order = np.argsort(-scores, axis=1, kind="mergesort")
    for local_i, row_i in enumerate(row_indices):
        ordered_ids = cand_ids[local_i, order[local_i]]
        exact_rank, identity_rank = first_ranks_for_row(int(row_i), ordered_ids)
        exact_ranks[local_i] = exact_rank
        identity_ranks[local_i] = identity_rank
    return exact_ranks, identity_ranks


def evaluate_ablation_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    rank_rows = []
    t0 = time.perf_counter()
    for split in EVAL_SPLITS:
        row_indices = split_indices[split]
        exact_blocks = {name: [] for name in ablation_weights}
        identity_blocks = {name: [] for name in ablation_weights}
        for start in tqdm(range(0, len(row_indices), int(RERANK_BATCH_SIZE)), desc=f"Evaluating ablations: {split}"):
            batch_indices = row_indices[start : start + int(RERANK_BATCH_SIZE)]
            cand_ids, features = build_feature_block(batch_indices, POOL_K)
            for system_name, weights in ablation_weights.items():
                scores = score_features(features, weights)
                exact_ranks, identity_ranks = ranks_from_scores(cand_ids, scores, batch_indices)
                exact_blocks[system_name].append(exact_ranks)
                identity_blocks[system_name].append(identity_ranks)

        non_identical = ~lab_pairs.iloc[row_indices]["metabolite_normalized"].eq(
            lab_pairs.iloc[row_indices]["refmet_normalized"]
        ).to_numpy()
        query_ids = lab_pairs.iloc[row_indices]["query_id"].to_numpy(dtype=np.int64)
        for system_name in ablation_weights:
            exact = np.concatenate(exact_blocks[system_name])
            identity = np.concatenate(identity_blocks[system_name])
            metrics = {
                **metrics_from_ranks(exact, prefix="exact_"),
                **metrics_from_ranks(identity, prefix="identity_"),
                **metrics_from_ranks(exact[non_identical], prefix="nonidentical_exact_"),
                **metrics_from_ranks(identity[non_identical], prefix="nonidentical_identity_"),
            }
            metrics_rows.append(
                {
                    "split": split,
                    "system": system_name,
                    "n_queries": int(len(row_indices)),
                    "n_nonidentical_queries": int(non_identical.sum()),
                    **metrics,
                }
            )
            rank_rows.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "system": system_name,
                        "query_id": query_ids,
                        "exact_rank": exact,
                        "identity_rank": identity,
                        "nonidentical": non_identical,
                    }
                )
            )
    metrics_df = pd.DataFrame(metrics_rows)
    ranks_df = pd.concat(rank_rows, ignore_index=True)
    metrics_df["elapsed_total_sec"] = float(time.perf_counter() - t0)
    return metrics_df, ranks_df


metrics_path = DAY2_DIR / "reranker_ablation_metrics.csv"
ranks_path = DAY2_DIR / "reranker_ablation_per_query_ranks.csv"
if metrics_path.exists() and ranks_path.exists() and MAX_QUERIES_PER_SPLIT is None:
    print("Loading cached full ablation outputs. Delete them to force recomputation.")
    ablation_metrics = pd.read_csv(metrics_path)
    ablation_ranks = pd.read_csv(ranks_path)
else:
    ablation_metrics, ablation_ranks = evaluate_ablation_splits()
    ablation_metrics.to_csv(metrics_path, index=False)
    ablation_ranks.to_csv(ranks_path, index=False)

print(ablation_metrics.sort_values(["split", "exact_MRR"], ascending=[True, False]).to_string(index=False))


def plot_ablation_metrics(metrics_df: pd.DataFrame) -> None:
    if plt is None:
        print("Skipping reranker_ablation_plot.png because matplotlib is unavailable.")
        return
    plot_df = metrics_df[metrics_df["split"].eq(BOOTSTRAP_SPLIT)].copy()
    if plot_df.empty:
        plot_df = metrics_df.copy()
    plot_df = plot_df.sort_values("exact_MRR", ascending=False)
    systems = plot_df["system"].tolist()
    x = np.arange(len(systems))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(12, len(systems) * 1.25), 6))
    ax.bar(x - width / 2, plot_df["exact_MRR"], width, label="Exact MRR")
    ax.bar(x + width / 2, plot_df["exact_HitRate@1"], width, label="Exact Hit@1")
    ax.set_title(f"Reranker ablation performance ({BOOTSTRAP_SPLIT} split)")
    ax.set_ylabel("Score")
    ymax = max(plot_df["exact_MRR"].max(), plot_df["exact_HitRate@1"].max())
    ax.set_ylim(0, min(1.0, ymax + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = DAY2_DIR / "reranker_ablation_plot.png"
    fig.savefig(out, dpi=180)
    print("Saved:", out)


plot_ablation_metrics(ablation_metrics)


def finite_rank_array(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.inf
    return arr


def bootstrap_ci_for_components(components: dict[str, np.ndarray], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(next(iter(components.values()))) if components else 0
    rows = []
    for metric, vals in components.items():
        vals = np.asarray(vals, dtype=np.float64)
        estimate = float(vals.mean()) if n else float("nan")
        boot = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = vals[idx].mean()
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n_queries": int(n),
                "n_bootstrap": int(n_boot),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_difference(comp_a: dict[str, np.ndarray], comp_b: dict[str, np.ndarray], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(next(iter(comp_a.values()))) if comp_a else 0
    for metric in [m for m in comp_a if m in comp_b]:
        a = np.asarray(comp_a[metric], dtype=np.float64)
        b = np.asarray(comp_b[metric], dtype=np.float64)
        if len(a) != len(b):
            raise ValueError(f"Cannot bootstrap paired difference for {metric}: lengths differ")
        diff_vals = a - b
        boot = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[i] = diff_vals[idx].mean()
        rows.append(
            {
                "metric": metric,
                "difference": float(diff_vals.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n_queries": int(n),
                "n_bootstrap": int(n_boot),
            }
        )
    return pd.DataFrame(rows)


def append_ci_rows(rows: list[dict], source: str, split: str, system: str, scoring: str, rank_arr: np.ndarray, seed_offset: int = 0):
    ci = bootstrap_ci_for_components(ranks_to_components(rank_arr), BOOTSTRAP_N, BOOTSTRAP_SEED + seed_offset)
    for _, row in ci.iterrows():
        rows.append({"source": source, "split": split, "system": system, "scoring": scoring, **row.to_dict()})


def append_diff_rows(
    rows: list[dict],
    source: str,
    split: str,
    system_a: str,
    system_b: str,
    scoring: str,
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    seed_offset: int = 0,
):
    diff = bootstrap_difference(ranks_to_components(ranks_a), ranks_to_components(ranks_b), BOOTSTRAP_N, BOOTSTRAP_SEED + seed_offset)
    comp_a = ranks_to_components(ranks_a)
    comp_b = ranks_to_components(ranks_b)
    for _, row in diff.iterrows():
        metric = row["metric"]
        rows.append(
            {
                "source": source,
                "split": split,
                "system_a": system_a,
                "system_b": system_b,
                "scoring": scoring,
                "estimate_a": float(comp_a[metric].mean()),
                "estimate_b": float(comp_b[metric].mean()),
                **row.to_dict(),
            }
        )


ci_rows: list[dict] = []
diff_rows: list[dict] = []
boot_ranks = ablation_ranks[ablation_ranks["split"].eq(BOOTSTRAP_SPLIT)].copy()

for offset, (system, group) in enumerate(boot_ranks.groupby("system")):
    group = group.sort_values("query_id")
    append_ci_rows(ci_rows, "reranker_ablation", BOOTSTRAP_SPLIT, system, "exact", finite_rank_array(group["exact_rank"]), 10 + offset)
    append_ci_rows(ci_rows, "reranker_ablation", BOOTSTRAP_SPLIT, system, "identity", finite_rank_array(group["identity_rank"]), 100 + offset)


def ranks_for_system(system: str, col: str) -> np.ndarray:
    group = boot_ranks[boot_ranks["system"].eq(system)].sort_values("query_id")
    if group.empty:
        raise KeyError(system)
    return finite_rank_array(group[col])


for i, (a, b) in enumerate(
    [
        ("full_hybrid", "embedding_only"),
        ("full_hybrid", "lexical_only_best_weights"),
        ("full_hybrid", "dense_plus_exact"),
        ("full_hybrid", "dense_plus_soft_lexical"),
    ]
):
    append_diff_rows(diff_rows, "reranker_ablation", BOOTSTRAP_SPLIT, a, b, "exact", ranks_for_system(a, "exact_rank"), ranks_for_system(b, "exact_rank"), 200 + i)
    append_diff_rows(
        diff_rows,
        "reranker_ablation",
        BOOTSTRAP_SPLIT,
        a,
        b,
        "identity",
        ranks_for_system(a, "identity_rank"),
        ranks_for_system(b, "identity_rank"),
        250 + i,
    )

final_pred = pd.read_csv(required["final_predictions"]).sort_values("query_id")
append_ci_rows(ci_rows, "canonical_final_predictions", "test", "embedding_only_sapbert_contrastive", "exact", finite_rank_array(final_pred["embedding_exact_rank"]), 300)
append_ci_rows(ci_rows, "canonical_final_predictions", "test", "final_reranker_sapbert_contrastive", "exact", finite_rank_array(final_pred["reranker_exact_rank"]), 301)
append_ci_rows(ci_rows, "canonical_final_predictions", "test", "embedding_only_sapbert_contrastive", "identity", finite_rank_array(final_pred["embedding_identity_rank"]), 302)
append_ci_rows(ci_rows, "canonical_final_predictions", "test", "final_reranker_sapbert_contrastive", "identity", finite_rank_array(final_pred["reranker_identity_rank"]), 303)
append_diff_rows(
    diff_rows,
    "canonical_final_predictions",
    "test",
    "final_reranker_sapbert_contrastive",
    "embedding_only_sapbert_contrastive",
    "exact",
    finite_rank_array(final_pred["reranker_exact_rank"]),
    finite_rank_array(final_pred["embedding_exact_rank"]),
    304,
)
append_diff_rows(
    diff_rows,
    "canonical_final_predictions",
    "test",
    "final_reranker_sapbert_contrastive",
    "embedding_only_sapbert_contrastive",
    "identity",
    finite_rank_array(final_pred["reranker_identity_rank"]),
    finite_rank_array(final_pred["embedding_identity_rank"]),
    305,
)

summary = pd.read_csv(required["model_summary"])
completed = summary[summary["status"].eq("complete")].copy()
non_sapbert = completed[~completed["model_key"].astype(str).str.contains("sapbert", case=False, na=False)].copy()
best_non_sapbert = None if non_sapbert.empty else non_sapbert.sort_values("embedding_exact_MRR", ascending=False).iloc[0]["model_key"]
model_pairs = [("sapbert_contrastive", "sapbert", "external_tuned_vs_untuned_sapbert")]
if best_non_sapbert:
    model_pairs.append(("sapbert_contrastive", str(best_non_sapbert), "external_tuned_vs_best_non_sapbert"))

for i, (a_model, b_model, source_name) in enumerate(model_pairs):
    a_path = MODEL_BENCHMARK_DIR / a_model / "test_predictions.csv"
    b_path = MODEL_BENCHMARK_DIR / b_model / "test_predictions.csv"
    if not (a_path.exists() and b_path.exists()):
        print("Skipping missing model benchmark pair:", a_model, b_model)
        continue
    a_df = pd.read_csv(a_path).sort_values("query_id")
    b_df = pd.read_csv(b_path).sort_values("query_id")
    if not np.array_equal(a_df["query_id"].to_numpy(), b_df["query_id"].to_numpy()):
        raise ValueError(f"Model benchmark query IDs do not align for {a_model} vs {b_model}")
    for scoring, col in [("exact", "embedding_exact_rank"), ("identity", "embedding_identity_rank")]:
        a_ranks = finite_rank_array(a_df[col])
        b_ranks = finite_rank_array(b_df[col])
        append_ci_rows(ci_rows, source_name, "test", f"{a_model}_embedding", scoring, a_ranks, 400 + i)
        append_ci_rows(ci_rows, source_name, "test", f"{b_model}_embedding", scoring, b_ranks, 450 + i)
        append_diff_rows(diff_rows, source_name, "test", f"{a_model}_embedding", f"{b_model}_embedding", scoring, a_ranks, b_ranks, 500 + i)

ci_df = pd.DataFrame(ci_rows)
diff_df = pd.DataFrame(diff_rows)

ci_root_path = (
    RESULTS_DIR / "bootstrap_confidence_intervals.csv"
    if MAX_QUERIES_PER_SPLIT is None
    else DAY2_DIR / "bootstrap_confidence_intervals.csv"
)
diff_root_path = (
    RESULTS_DIR / "bootstrap_metric_differences.csv"
    if MAX_QUERIES_PER_SPLIT is None
    else DAY2_DIR / "bootstrap_metric_differences.csv"
)
ci_run_path = DAY2_DIR / "bootstrap_confidence_intervals.csv"
diff_run_path = DAY2_DIR / "bootstrap_metric_differences.csv"
ci_df.to_csv(ci_root_path, index=False)
diff_df.to_csv(diff_root_path, index=False)
ci_df.to_csv(ci_run_path, index=False)
diff_df.to_csv(diff_run_path, index=False)


def plot_key_differences(diff_df: pd.DataFrame) -> None:
    if plt is None:
        print("Skipping bootstrap_metric_differences.png because matplotlib is unavailable.")
        return
    keep_metrics = ["MRR", "HitRate@1", "HitRate@10"]
    plot_df = diff_df[
        diff_df["metric"].isin(keep_metrics)
        & diff_df["scoring"].eq("exact")
        & diff_df["source"].isin(
            [
                "reranker_ablation",
                "canonical_final_predictions",
                "external_tuned_vs_untuned_sapbert",
                "external_tuned_vs_best_non_sapbert",
            ]
        )
    ].copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["source"] + "\n" + plot_df["system_a"] + " - " + plot_df["system_b"] + "\n" + plot_df["metric"]
    plot_df = plot_df.sort_values(["source", "metric"]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    y = plot_df["difference"].to_numpy(dtype=float)
    yerr = np.vstack([y - plot_df["ci_low"].to_numpy(dtype=float), plot_df["ci_high"].to_numpy(dtype=float) - y])
    fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 0.8), 7))
    ax.bar(x, y, yerr=yerr, capsize=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Bootstrap paired difference")
    ax.set_title("Bootstrap metric differences, exact scoring")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=65, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_root = RESULTS_DIR / "bootstrap_metric_differences.png" if MAX_QUERIES_PER_SPLIT is None else DAY2_DIR / "bootstrap_metric_differences.png"
    fig.savefig(figure_root, dpi=180)
    fig.savefig(DAY2_DIR / "bootstrap_metric_differences.png", dpi=180)


plot_key_differences(diff_df)


def fmt_pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def metric_lookup(df: pd.DataFrame, split: str, system: str, col: str) -> float:
    row = df[df["split"].eq(split) & df["system"].eq(system)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


summary_lines = [
    "# Day 2 Reranker Ablation and Bootstrap Summary",
    "",
    "Generated from `notebooks/external_lab_refmet_day2_reranker_ablations_bootstrap.ipynb`.",
    "",
    "## Reranker Ablation Outputs",
    "",
    f"- Metrics: `{metrics_path.relative_to(PROJECT_ROOT)}`",
    f"- Per-query ranks: `{ranks_path.relative_to(PROJECT_ROOT)}`",
    (
        f"- Plot: `{(DAY2_DIR / 'reranker_ablation_plot.png').relative_to(PROJECT_ROOT)}`"
        if (DAY2_DIR / "reranker_ablation_plot.png").exists()
        else "- Plot: not generated because `matplotlib` was unavailable in this runtime."
    ),
    "",
]

best_test = ablation_metrics[ablation_metrics["split"].eq(BOOTSTRAP_SPLIT)].sort_values("exact_MRR", ascending=False).head(1)
if not best_test.empty:
    row = best_test.iloc[0]
    summary_lines.extend(
        [
            "## Best Test Ablation",
            "",
            (
                f"The highest exact MRR on the `{BOOTSTRAP_SPLIT}` split was `{row['system']}` "
                f"with MRR={row['exact_MRR']:.4f}, Hit@1={fmt_pct(row['exact_HitRate@1'])}, "
                f"Hit@10={fmt_pct(row['exact_HitRate@10'])}, and Hit@256={fmt_pct(row['exact_HitRate@256'])}."
            ),
            "",
        ]
    )

for system in ["embedding_only", "lexical_only_best_weights", "dense_plus_exact", "dense_plus_soft_lexical", "full_hybrid"]:
    mrr = metric_lookup(ablation_metrics, BOOTSTRAP_SPLIT, system, "exact_MRR")
    hit1 = metric_lookup(ablation_metrics, BOOTSTRAP_SPLIT, system, "exact_HitRate@1")
    if np.isfinite(mrr):
        summary_lines.append(f"- `{system}`: exact MRR={mrr:.4f}, exact Hit@1={fmt_pct(hit1)}")

summary_lines.extend(
    [
        "",
        "## Bootstrap Outputs",
        "",
        f"- Confidence intervals: `{ci_root_path.relative_to(PROJECT_ROOT)}`",
        f"- Metric differences: `{diff_root_path.relative_to(PROJECT_ROOT)}`",
        (
            f"- Difference plot: `{((RESULTS_DIR / 'bootstrap_metric_differences.png') if MAX_QUERIES_PER_SPLIT is None else (DAY2_DIR / 'bootstrap_metric_differences.png')).relative_to(PROJECT_ROOT)}`"
            if (((RESULTS_DIR / "bootstrap_metric_differences.png") if MAX_QUERIES_PER_SPLIT is None else (DAY2_DIR / "bootstrap_metric_differences.png")).exists())
            else "- Difference plot: not generated because `matplotlib` was unavailable in this runtime."
        ),
        "",
        "## Notes",
        "",
        "- Bootstrap intervals are computed by resampling per-query ranks with replacement.",
        "- The external-lab task supports bootstrap because per-query ranks are saved.",
        "- Part One internal benchmark bootstrap still requires per-query rank artifacts; aggregate metric tables alone are not sufficient for valid bootstrap resampling.",
        "- Hit@256 is candidate-pool recall and is unchanged by reranking methods that only reorder the same top-256 candidates.",
    ]
)

summary_path = DAY2_DIR / "day2_ablation_bootstrap_summary.md"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("Saved:")
for path in [
    metrics_path,
    ranks_path,
    DAY2_DIR / "reranker_ablation_plot.png",
    ci_root_path,
    diff_root_path,
    (RESULTS_DIR / "bootstrap_metric_differences.png") if MAX_QUERIES_PER_SPLIT is None else (DAY2_DIR / "bootstrap_metric_differences.png"),
    summary_path,
]:
    if path.exists():
        print(" -", path)
