"""Evaluate a lexical/chemical reranker for saved PubMed fallback candidates.

This script intentionally does not embed strings and does not run FAISS. It
reuses the PubMed/PubChem fallback neighbor arrays produced by
refmet_harmonization_publication_pipeline.ipynb and asks whether reranking those
saved candidates improves same-CID recovery when RefMet retrieval missed the
gold RefMet target in the top 10.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9]+")
CID_SPLIT_RE = re.compile(r"[|;,]\s*")
LIPID_RE = re.compile(
    r"([a-z]{1,8})\s*\(?\s*(\d{1,2})\s*[:/]\s*(\d{1,2})"
    r"(?:[;_/,-](\d{1,2})\s*[:/]\s*(\d{1,2}))*",
    re.IGNORECASE,
)
STEREO_RE = re.compile(r"\b(?:cis|trans|alpha|beta|d|l|r|s|e|z|rac|ent)\b|[()\[\]@+-]")

FEATURE_COLUMNS = [
    "embedding_score_norm",
    "rank_prior",
    "compact_exact",
    "sequence_ratio",
    "char3_jaccard",
    "token_jaccard",
    "lipid_signature_match",
    "lipid_class_match",
    "stereo_mismatch",
]

DEFAULT_REFMET_WEIGHTS = {
    "embedding_score_norm": 0.5223138794606293,
    "rank_prior": 0.010028118150807442,
    "compact_exact": 0.10347796607940273,
    "sequence_ratio": 0.20662133328771864,
    "char3_jaccard": 0.10506264532298014,
    "token_jaccard": 0.20224043840953188,
    "lipid_signature_match": 0.027791846646039415,
    "lipid_class_match": 0.04006538444337206,
    "stereo_mismatch": -0.009761344640321495,
}


@dataclass
class Config:
    project_dir: Path
    source_run_name: str = "refmet_harmonization_publication_pipeline"
    output_name: str = "pubmed_fallback_reranker_evaluation"
    pool_k: int = 256
    top_x: int = 10
    random_search_trials: int = 120
    tuning_max_queries: int = 10_000
    random_seed: int = 7
    chunk_size: int = 1024
    copy_inputs_to_scratch: bool = True
    force: bool = False

    @property
    def source_results_dir(self) -> Path:
        return self.project_dir / "results" / self.source_run_name

    @property
    def source_artifact_dir(self) -> Path:
        return self.source_results_dir / "artifacts"

    @property
    def output_dir(self) -> Path:
        return self.source_results_dir / self.output_name

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def scratch_dir(self) -> Path:
        if Path("/mnt/local-scratch").exists():
            root = Path("/mnt/local-scratch")
        elif Path("/content").exists():
            root = Path("/content")
        else:
            root = self.project_dir / ".scratch"
        return root / "chem_synonymizer" / self.source_run_name / self.output_name


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def find_project_dir(cli_project_dir: str | None = None) -> Path:
    candidates = []
    if cli_project_dir:
        candidates.append(Path(cli_project_dir))
    if env("CHEMSYN_PROJECT_DIR"):
        candidates.append(Path(env("CHEMSYN_PROJECT_DIR", "")))
    candidates.extend(
        [
            Path("/content/drive/MyDrive/ChemSynonymizer/chem_synonymizer"),
            Path("/content/drive/MyDrive/ChemSynonymizer"),
            Path.cwd(),
            Path.cwd().parent,
        ]
    )
    for candidate in candidates:
        if (candidate / "data").exists() or (candidate / "src").exists() or (candidate / "results").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        project_dir=find_project_dir(args.project_dir),
        source_run_name=env("CHEMSYN_SOURCE_RUN_NAME", args.source_run_name),
        output_name=env("CHEMSYN_OUTPUT_NAME", args.output_name),
        pool_k=env_int("CHEMSYN_POOL_K", args.pool_k),
        top_x=env_int("CHEMSYN_TOP_X", args.top_x),
        random_search_trials=env_int("CHEMSYN_RANDOM_SEARCH_TRIALS", args.random_search_trials),
        tuning_max_queries=env_int("CHEMSYN_TUNING_MAX_QUERIES", args.tuning_max_queries),
        random_seed=env_int("CHEMSYN_RANDOM_SEED", args.random_seed),
        chunk_size=env_int("CHEMSYN_CHUNK_SIZE", args.chunk_size),
        copy_inputs_to_scratch=env_bool("CHEMSYN_COPY_INPUTS_TO_SCRATCH", args.copy_inputs_to_scratch),
        force=env_bool("CHEMSYN_FORCE", args.force),
    )


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = text.replace("\u2212", "-").replace("Âµ", "u")
    return SPACE_RE.sub(" ", text)


def compact_text(value) -> str:
    return PUNCT_RE.sub("", normalize_name(value))


@lru_cache(maxsize=500_000)
def cached_token_set(value: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", normalize_name(value)) if t)


@lru_cache(maxsize=500_000)
def cached_char3(value: str) -> frozenset[str]:
    text = compact_text(value)
    if not text:
        return frozenset()
    if len(text) <= 3:
        return frozenset([text])
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


@lru_cache(maxsize=500_000)
def cached_lipid_signature(value: str) -> str:
    matches = LIPID_RE.findall(normalize_name(value))
    if not matches:
        return ""
    return "|".join(f"{m[1]}:{m[2]}" for m in matches if m[1] and m[2])


@lru_cache(maxsize=500_000)
def cached_lipid_class(value: str) -> str:
    match = re.match(r"^([a-z]{1,8})\b", normalize_name(value))
    return match.group(1) if match else ""


@lru_cache(maxsize=500_000)
def cached_has_stereo(value: str) -> bool:
    return bool(STEREO_RE.search(normalize_name(value)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def fast_sequence_ratio(a: str, b: str) -> float:
    ac = compact_text(a)
    bc = compact_text(b)
    if not ac or not bc:
        return 0.0
    if ac == bc:
        return 1.0
    if len(ac) > 96 or len(bc) > 96:
        return min(len(ac), len(bc)) / max(len(ac), len(bc))
    return float(SequenceMatcher(None, ac, bc).ratio())


def parse_cids(value) -> set[str]:
    if pd.isna(value) or value is None:
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    out = set()
    for part in CID_SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(str(int(float(part))))
        except Exception:
            out.add(part)
    return out


def cid_string(cids: Iterable[str]) -> str:
    values = sorted((str(c) for c in cids if str(c)), key=lambda x: int(x) if str(x).isdigit() else str(x))
    return "|".join(values)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def restore_file_from_parts(parts_dir: Path, dest_path: Path, force: bool = False) -> Path | None:
    manifest_path = parts_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest["source_size_bytes"])
    if dest_path.exists() and dest_path.stat().st_size == expected and not force:
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with tmp_path.open("wb") as out_f:
        for i in range(int(manifest["n_parts"])):
            part_path = parts_dir / f"part_{i:05d}.bin"
            print(f"Restoring {dest_path.name} part {i + 1}/{manifest['n_parts']}")
            with part_path.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=64 * 1024 * 1024)
    tmp_path.replace(dest_path)
    return dest_path


def find_required_file(cfg: Config, name: str) -> Path:
    candidates = [
        cfg.source_artifact_dir / name,
        cfg.source_results_dir / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
        parts_dir = candidate.with_suffix(candidate.suffix + ".parts")
        if parts_dir.exists():
            restored = restore_file_from_parts(parts_dir, cfg.scratch_dir / name, cfg.force)
            if restored is not None:
                return restored
    raise FileNotFoundError(f"Could not find required input `{name}` in {cfg.source_artifact_dir} or {cfg.source_results_dir}")


def materialize(path: Path, cfg: Config) -> Path:
    if not cfg.copy_inputs_to_scratch:
        return path
    if path.suffix.lower() not in {".npy", ".csv"}:
        return path
    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.scratch_dir / path.name
    if dest.exists() and dest.stat().st_size == path.stat().st_size and not cfg.force:
        return dest
    print(f"Copying {path.name} to scratch for faster local reads...")
    shutil.copyfile(path, dest)
    return dest


def rank_metrics(ranks: np.ndarray) -> dict:
    ranks = np.asarray(ranks, dtype=np.float64)
    finite = np.isfinite(ranks)
    return {
        "n": int(len(ranks)),
        "MRR": float(np.where(finite, 1.0 / ranks, 0.0).mean()) if len(ranks) else math.nan,
        "Hit@1": float((ranks <= 1).mean()) if len(ranks) else math.nan,
        "Hit@5": float((ranks <= 5).mean()) if len(ranks) else math.nan,
        "Hit@10": float((ranks <= 10).mean()) if len(ranks) else math.nan,
        "Hit@50": float((ranks <= 50).mean()) if len(ranks) else math.nan,
        "Hit@256": float((ranks <= 256).mean()) if len(ranks) else math.nan,
        "median_rank": float(np.median(ranks[finite])) if finite.any() else math.inf,
    }


def feature_values(q_norm: str, q_compact: str, t_norm: str, t_compact: str, emb_score: float, rank0: int) -> list[float]:
    return [
        float((emb_score + 1.0) / 2.0),
        float(1.0 / (rank0 + 1.0)),
        float(q_compact == t_compact and q_compact != ""),
        fast_sequence_ratio(q_norm, t_norm),
        jaccard(cached_char3(q_norm), cached_char3(t_norm)),
        jaccard(cached_token_set(q_norm), cached_token_set(t_norm)),
        float(cached_lipid_signature(q_norm) != "" and cached_lipid_signature(q_norm) == cached_lipid_signature(t_norm)),
        float(cached_lipid_class(q_norm) != "" and cached_lipid_class(q_norm) == cached_lipid_class(t_norm)),
        float(cached_has_stereo(q_norm) != cached_has_stereo(t_norm)),
    ]


def load_refmet_weights(cfg: Config) -> dict:
    path = cfg.source_results_dir / "best_reranker_weights.json"
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if all(col in loaded for col in FEATURE_COLUMNS):
            return {col: float(loaded[col]) for col in FEATURE_COLUMNS}
    return dict(DEFAULT_REFMET_WEIGHTS)


def random_weights(rng: np.random.Generator) -> dict:
    return {
        "embedding_score_norm": float(rng.uniform(0.20, 0.75)),
        "rank_prior": float(rng.uniform(0.00, 0.08)),
        "compact_exact": float(rng.uniform(0.00, 0.30)),
        "sequence_ratio": float(rng.uniform(0.00, 0.35)),
        "char3_jaccard": float(rng.uniform(0.00, 0.25)),
        "token_jaccard": float(rng.uniform(0.00, 0.35)),
        "lipid_signature_match": float(rng.uniform(0.00, 0.12)),
        "lipid_class_match": float(rng.uniform(0.00, 0.10)),
        "stereo_mismatch": float(rng.uniform(-0.18, 0.00)),
    }


def weight_vector(weights: dict) -> np.ndarray:
    return np.array([weights[col] for col in FEATURE_COLUMNS], dtype=np.float32)


def first_same_cid_rank_from_order(candidate_cids: list[set[str]], gold_cids: set[str], order: np.ndarray) -> float:
    if not gold_cids:
        return math.inf
    for rank0, j in enumerate(order):
        if int(j) >= len(candidate_cids):
            continue
        if gold_cids & candidate_cids[int(j)]:
            return float(rank0 + 1)
    return math.inf


def make_blocks(
    qids: np.ndarray,
    predictions: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    surface_names: np.ndarray,
    surface_norm: np.ndarray,
    surface_compact: np.ndarray,
    surface_cids: np.ndarray,
    gold_cid_by_qid: dict[int, set[str]],
    pool_k: int,
) -> tuple[np.ndarray, np.ndarray, list[list[set[str]]], np.ndarray]:
    qids = np.asarray(qids, dtype=np.int64)
    pool_k = min(pool_k, indices.shape[1])
    feats = np.zeros((len(qids), pool_k, len(FEATURE_COLUMNS)), dtype=np.float32)
    labels = np.zeros((len(qids), pool_k), dtype=bool)
    invalid = np.zeros((len(qids), pool_k), dtype=bool)
    cid_blocks: list[list[set[str]]] = []

    pred_lookup = predictions.set_index("query_i")
    for row_i, qid in enumerate(tqdm(qids, desc="Building PubMed reranker features")):
        q = pred_lookup.at[int(qid), "metabolite_name"]
        q_norm = normalize_name(q)
        q_compact = compact_text(q_norm)
        gold_cids = gold_cid_by_qid.get(int(qid), set())
        row_cids = []
        for j in range(pool_k):
            pos = int(indices[int(qid), j])
            if pos < 0:
                invalid[row_i, j] = True
                row_cids.append(set())
                continue
            t_norm = str(surface_norm[pos]) if pd.notna(surface_norm[pos]) else normalize_name(surface_names[pos])
            t_compact = str(surface_compact[pos]) if pd.notna(surface_compact[pos]) else compact_text(t_norm)
            cids = parse_cids(surface_cids[pos])
            row_cids.append(cids)
            labels[row_i, j] = bool(gold_cids & cids)
            feats[row_i, j] = feature_values(q_norm, q_compact, t_norm, t_compact, float(scores[int(qid), j]), j)
        cid_blocks.append(row_cids)
    return feats, labels, cid_blocks, invalid


def ranks_from_scores(score_matrix: np.ndarray, labels: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    score_matrix = np.where(invalid, -np.inf, score_matrix)
    order = np.argsort(-score_matrix, axis=1)
    ranks = np.full(labels.shape[0], np.inf, dtype=np.float32)
    for row_i in range(labels.shape[0]):
        found = np.flatnonzero(labels[row_i, order[row_i]])
        if len(found):
            ranks[row_i] = float(found[0] + 1)
    return ranks


def tune_weights(
    feats: np.ndarray,
    labels: np.ndarray,
    invalid: np.ndarray,
    refmet_weights: dict,
    cfg: Config,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)
    candidates = [
        ("embedding_only", {"embedding_score_norm": 1.0, **{c: 0.0 for c in FEATURE_COLUMNS if c != "embedding_score_norm"}}),
        ("refmet_weights", refmet_weights),
    ]
    for trial in range(cfg.random_search_trials):
        candidates.append((f"random_{trial:04d}", random_weights(rng)))

    rows = []
    best_name = ""
    best_weights = candidates[0][1]
    best_mrr = -1.0
    for name, weights in tqdm(candidates, desc="Tuning PubMed fallback reranker"):
        scores = np.tensordot(feats, weight_vector(weights), axes=([2], [0]))
        ranks = ranks_from_scores(scores, labels, invalid)
        metrics = rank_metrics(ranks)
        row = {"trial": name, **metrics, **{col: float(weights[col]) for col in FEATURE_COLUMNS}}
        rows.append(row)
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            best_name = name
            best_weights = dict(weights)
    out = pd.DataFrame(rows).sort_values("MRR", ascending=False)
    print(f"Best validation PubMed fallback reranker: {best_name}, MRR={best_mrr:.4f}")
    return best_weights, out


def evaluate_method(
    method_name: str,
    weights: dict | None,
    qids: np.ndarray,
    predictions: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    surface_names: np.ndarray,
    surface_norm: np.ndarray,
    surface_compact: np.ndarray,
    surface_cids: np.ndarray,
    gold_cid_by_qid: dict[int, set[str]],
    cfg: Config,
) -> tuple[dict, pd.DataFrame]:
    all_ranks = []
    top_rows = []
    for start in range(0, len(qids), cfg.chunk_size):
        batch_qids = qids[start : start + cfg.chunk_size]
        feats, labels, _, invalid = make_blocks(
            batch_qids,
            predictions,
            scores,
            indices,
            surface_names,
            surface_norm,
            surface_compact,
            surface_cids,
            gold_cid_by_qid,
            cfg.pool_k,
        )
        if weights is None:
            rank_scores = np.where(invalid, -np.inf, scores[batch_qids, : feats.shape[1]])
        else:
            rank_scores = np.tensordot(feats, weight_vector(weights), axes=([2], [0]))
            rank_scores = np.where(invalid, -np.inf, rank_scores)
        ranks = ranks_from_scores(rank_scores, labels, invalid)
        all_ranks.append(ranks)

        order = np.argsort(-rank_scores, axis=1)
        pred_lookup = predictions.set_index("query_i")
        for row_i, qid in enumerate(batch_qids):
            qid_int = int(qid)
            row = {
                "query_i": qid_int,
                "split": pred_lookup.at[qid_int, "split"],
                "metabolite_name": pred_lookup.at[qid_int, "metabolite_name"],
                "gold_refmet_name": pred_lookup.at[qid_int, "gold_refmet_name"],
                "gold_refmet_cids": cid_string(gold_cid_by_qid.get(qid_int, set())),
                f"{method_name}_same_cid_rank": float(ranks[row_i]) if np.isfinite(ranks[row_i]) else np.inf,
            }
            for rank0, j in enumerate(order[row_i, : cfg.top_x]):
                pos = int(indices[qid_int, int(j)])
                if pos < 0:
                    continue
                row[f"{method_name}_top{rank0 + 1}_name"] = surface_names[pos]
                row[f"{method_name}_top{rank0 + 1}_cids"] = surface_cids[pos]
                row[f"{method_name}_top{rank0 + 1}_embedding_score"] = float(scores[qid_int, int(j)])
                row[f"{method_name}_top{rank0 + 1}_rerank_score"] = float(rank_scores[row_i, int(j)])
            top_rows.append(row)
        del feats, labels, invalid, rank_scores
        gc.collect()

    ranks = np.concatenate(all_ranks) if all_ranks else np.array([], dtype=np.float32)
    metrics = {"method": method_name, **rank_metrics(ranks)}
    return metrics, pd.DataFrame(top_rows)


def plot_metrics(metrics: pd.DataFrame, cfg: Config) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc!r}")
        return
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["MRR", "Hit@1", "Hit@5", "Hit@10", "Hit@50", "Hit@256"]:
        plt.figure(figsize=(9, 4))
        plt.bar(metrics["method"], metrics[metric])
        plt.ylabel(metric)
        plt.title(f"PubMed fallback same-CID recovery: {metric}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / f"pubmed_fallback_{metric.replace('@', 'at')}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    save_json(cfg.output_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()})

    print(f"Project: {cfg.project_dir}")
    print(f"Source results: {cfg.source_results_dir}")
    print(f"Output: {cfg.output_dir}")
    print("This run reuses saved PubMed fallback neighbors. No embeddings or FAISS search are run.")

    predictions_path = find_required_file(cfg, "refmet_predictions_with_decisions.csv")
    targets_path = find_required_file(cfg, "refmet_target_dictionary.csv")
    surfaces_path = materialize(find_required_file(cfg, "pubmed_fallback_surfaces.csv"), cfg)
    scores_path = materialize(find_required_file(cfg, "pubmed_fallback_neighbor_scores.npy"), cfg)
    indices_path = materialize(find_required_file(cfg, "pubmed_fallback_neighbor_indices.npy"), cfg)

    print("Loading predictions...")
    pred_cols = [
        "query_i",
        "split",
        "metabolite_name",
        "gold_refmet_name",
        "gold_target_id",
        "predicted_refmet_name",
        "gold_rank",
        "confidence_tier",
    ]
    predictions = pd.read_csv(predictions_path, usecols=pred_cols)
    predictions["query_i"] = predictions["query_i"].astype(np.int64)
    predictions["gold_target_id"] = pd.to_numeric(predictions["gold_target_id"], errors="coerce").fillna(-1).astype(np.int64)
    predictions["gold_rank"] = pd.to_numeric(predictions["gold_rank"], errors="coerce").fillna(np.inf)

    print("Loading RefMet target CID map...")
    targets = pd.read_csv(targets_path, dtype=str)
    if "target_id" not in targets.columns:
        raise ValueError("refmet_target_dictionary.csv must contain target_id")
    cid_col = "refmet_cid" if "refmet_cid" in targets.columns else None
    if cid_col is None:
        raise ValueError("refmet_target_dictionary.csv must contain refmet_cid for same-CID fallback evaluation")
    target_cids = {
        int(float(row["target_id"])): parse_cids(row[cid_col])
        for _, row in targets[["target_id", cid_col]].iterrows()
    }
    predictions["gold_refmet_cids"] = predictions["gold_target_id"].map(lambda tid: cid_string(target_cids.get(int(tid), set())))
    predictions["gold_has_cid"] = predictions["gold_refmet_cids"].str.len() > 0

    eval_mask = (
        predictions["gold_has_cid"]
        & predictions["confidence_tier"].eq("low_confidence_pubmed_fallback")
        & (predictions["gold_rank"] > 10)
    )
    eval_cases = predictions[eval_mask].copy()
    gold_cid_by_qid = {
        int(row.query_i): parse_cids(row.gold_refmet_cids)
        for row in eval_cases[["query_i", "gold_refmet_cids"]].itertuples(index=False)
    }
    summary = {
        "n_total_predictions": int(len(predictions)),
        "n_gold_with_cid": int(predictions["gold_has_cid"].sum()),
        "n_fallback_gold_not_refmet_top10_with_gold_cid": int(len(eval_cases)),
        "split_counts": eval_cases["split"].value_counts().to_dict(),
    }
    save_json(cfg.output_dir / "input_subset_summary.json", summary)
    print(json.dumps(summary, indent=2))

    print("Loading saved PubMed/PubChem neighbor arrays...")
    neighbor_scores = np.load(scores_path, mmap_mode="r")
    neighbor_indices = np.load(indices_path, mmap_mode="r")
    cfg.pool_k = min(cfg.pool_k, neighbor_indices.shape[1])
    print(f"Neighbor scores: {neighbor_scores.shape}; indices: {neighbor_indices.shape}; pool_k={cfg.pool_k}")

    print("Loading PubMed/PubChem fallback surface table. This is the largest CPU-only read.")
    surface_cols = ["display_name", "normalized_name", "compact_name", "candidate_cids"]
    surfaces = pd.read_csv(surfaces_path, usecols=lambda c: c in set(surface_cols), dtype=str, low_memory=False)
    if "normalized_name" not in surfaces.columns:
        surfaces["normalized_name"] = surfaces["display_name"].map(normalize_name)
    if "compact_name" not in surfaces.columns:
        surfaces["compact_name"] = surfaces["normalized_name"].map(compact_text)
    for col in surface_cols:
        if col not in surfaces.columns:
            surfaces[col] = ""
    surface_names = surfaces["display_name"].astype(str).to_numpy()
    surface_norm = surfaces["normalized_name"].astype(str).to_numpy()
    surface_compact = surfaces["compact_name"].astype(str).to_numpy()
    surface_cids = surfaces["candidate_cids"].astype(str).to_numpy()
    print(f"Loaded {len(surfaces):,} PubMed/PubChem fallback surfaces.")

    refmet_weights = load_refmet_weights(cfg)
    save_json(cfg.output_dir / "refmet_reranker_weights_used.json", refmet_weights)

    validation_qids = eval_cases.loc[eval_cases["split"].eq("validation"), "query_i"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(cfg.random_seed)
    if cfg.tuning_max_queries and len(validation_qids) > cfg.tuning_max_queries:
        validation_qids = rng.choice(validation_qids, size=cfg.tuning_max_queries, replace=False)
        validation_qids = np.sort(validation_qids)
    if len(validation_qids) == 0:
        raise ValueError("No validation fallback cases with gold CIDs were found for tuning.")

    print(f"Tuning on {len(validation_qids):,} validation fallback cases.")
    tune_feats, tune_labels, _, tune_invalid = make_blocks(
        validation_qids,
        predictions,
        neighbor_scores,
        neighbor_indices,
        surface_names,
        surface_norm,
        surface_compact,
        surface_cids,
        gold_cid_by_qid,
        cfg.pool_k,
    )
    tuned_weights, search_rows = tune_weights(tune_feats, tune_labels, tune_invalid, refmet_weights, cfg)
    save_json(cfg.output_dir / "pubmed_fallback_reranker_weights.json", tuned_weights)
    search_rows.to_csv(cfg.output_dir / "pubmed_fallback_reranker_random_search.csv", index=False)
    del tune_feats, tune_labels, tune_invalid
    gc.collect()

    test_qids = eval_cases.loc[eval_cases["split"].eq("test"), "query_i"].to_numpy(dtype=np.int64)
    if len(test_qids) == 0:
        raise ValueError("No test fallback cases with gold CIDs were found for evaluation.")
    print(f"Evaluating on {len(test_qids):,} held-out test fallback cases.")

    methods = [
        ("embedding_original_order", None),
        ("refmet_weights_rerank", refmet_weights),
        ("pubmed_tuned_rerank", tuned_weights),
    ]
    metric_rows = []
    top_tables = []
    for method_name, weights in methods:
        print(f"Evaluating {method_name}...")
        metrics, top_df = evaluate_method(
            method_name,
            weights,
            test_qids,
            predictions,
            neighbor_scores,
            neighbor_indices,
            surface_names,
            surface_norm,
            surface_compact,
            surface_cids,
            gold_cid_by_qid,
            cfg,
        )
        metric_rows.append(metrics)
        top_tables.append(top_df.set_index("query_i"))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(cfg.output_dir / "pubmed_fallback_reranker_metrics.csv", index=False)
    save_json(cfg.output_dir / "pubmed_fallback_reranker_metrics.json", {"metrics": metric_rows})
    plot_metrics(metrics_df, cfg)

    combined = pd.concat(top_tables, axis=1)
    # Remove duplicate metadata columns introduced by concatenating method-specific tables.
    combined = combined.loc[:, ~combined.columns.duplicated()].reset_index()
    combined.to_csv(cfg.output_dir / "pubmed_fallback_rerank_top10_wide.csv", index=False)

    best_rank_col = "pubmed_tuned_rerank_same_cid_rank"
    combined["case_group"] = np.select(
        [
            combined[best_rank_col] <= 1,
            combined[best_rank_col] <= 10,
            combined["embedding_original_order_same_cid_rank"] <= 10,
        ],
        [
            "tuned_top1_same_cid",
            "tuned_top10_same_cid",
            "embedding_top10_only",
        ],
        default="same_cid_not_found_top256",
    )
    combined.to_csv(cfg.output_dir / "pubmed_fallback_rerank_gold_not_refmet_top10_cid_recovery.csv", index=False)

    samples = []
    for group, n in {
        "tuned_top1_same_cid": 50,
        "tuned_top10_same_cid": 75,
        "embedding_top10_only": 75,
        "same_cid_not_found_top256": 100,
    }.items():
        subset = combined[combined["case_group"].eq(group)]
        if not subset.empty:
            samples.append(subset.sample(min(n, len(subset)), random_state=cfg.random_seed))
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(cfg.output_dir / "pubmed_fallback_rerank_examples.csv", index=False)

    summary_lines = [
        "# PubMed/PubChem Fallback Reranker Evaluation",
        "",
        "This evaluation reuses the saved PubMed/PubChem fallback top-k neighbor arrays. It does not re-embed strings and does not rerun FAISS.",
        "",
        "## Evaluation Subset",
        f"- Source run: `{cfg.source_run_name}`",
        f"- Cases evaluated: fallback-routed rows where the gold RefMet target was not in the RefMet top 10 and the gold RefMet target had at least one PubMed/PubChem CID.",
        f"- Held-out test cases: {len(test_qids):,}",
        f"- Candidate pool per query: top {cfg.pool_k}",
        "",
        "## Test Metrics",
    ]
    for _, row in metrics_df.iterrows():
        summary_lines.append(
            f"- **{row['method']}**: MRR={row['MRR']:.4f}, "
            f"Hit@1={row['Hit@1']:.2%}, Hit@5={row['Hit@5']:.2%}, "
            f"Hit@10={row['Hit@10']:.2%}, Hit@50={row['Hit@50']:.2%}, "
            f"Hit@256={row['Hit@256']:.2%}"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "If Hit@256 is low, the saved PubMed/PubChem fallback search did not retrieve the true same-CID candidate often enough, and no reranker can recover those missing cases. If Hit@256 is high but Hit@10 is low, reranking is the main bottleneck.",
        ]
    )
    (cfg.output_dir / "pubmed_fallback_reranker_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"Outputs written to {cfg.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir")
    parser.add_argument("--source-run-name", default="refmet_harmonization_publication_pipeline")
    parser.add_argument("--output-name", default="pubmed_fallback_reranker_evaluation")
    parser.add_argument("--pool-k", type=int, default=256)
    parser.add_argument("--top-x", type=int, default=10)
    parser.add_argument("--random-search-trials", type=int, default=120)
    parser.add_argument("--tuning-max-queries", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--copy-inputs-to-scratch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
