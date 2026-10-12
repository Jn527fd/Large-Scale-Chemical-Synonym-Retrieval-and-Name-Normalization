"""Publication pipeline for RefMet harmonization experiments.

This runner is designed for Colab/A100 execution through
notebooks/refmet_harmonization_publication_pipeline.ipynb, but it can also be
run from a terminal. It keeps large embedding artifacts in local scratch during
execution and checkpoints them back to the project results folder.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


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

DEFAULT_WEIGHTS = {
    "embedding_score_norm": 0.6146199629957105,
    "rank_prior": 0.001058188851763783,
    "compact_exact": 0.0019459908649009634,
    "sequence_ratio": 0.15473446059955223,
    "char3_jaccard": 0.04851935434311621,
    "token_jaccard": 0.15534640024585533,
    "lipid_signature_match": 0.023706233573146503,
    "lipid_class_match": 0.0000694085259543694,
    "stereo_mismatch": -0.06,
}


@dataclass
class Config:
    project_dir: Path
    run_name: str = "refmet_harmonization_publication_pipeline"
    lab_csv: Path | None = None
    refmet_csv: Path | None = None
    pubmed_csv: Path | None = None
    model_name_or_path: str | None = None
    device: str = "cuda"
    max_length: int = 96
    pooling: str = "cls"
    embed_dtype: str = "float16"
    refmet_batch_size: int = 8192
    query_batch_size: int = 8192
    pubmed_batch_size: int = 8192
    top_x: int = 10
    k_retrieve_refmet: int = 256
    rerank_pool_k: int = 128
    k_retrieve_pubmed: int = 100
    search_batch_size: int = 8192
    exact_gpu_max_vectors: int = 2_500_000
    pubmed_search_mode: str = "stream_gpu"
    pubmed_gpu_chunk_vectors: int = 6_000_000
    pubmed_fallback_query_mode: str = "calibrated"
    auto_accept_target_precision: float = 0.90
    fallback_target_hit10: float = 0.80
    run_pubmed_fallback: bool = True
    rebuild_pubmed_condensed: bool = False
    pubmed_max_names_per_cid: int = 20
    pubmed_embed_row_limit: int | None = 2_000_000
    pubmed_deduplicate_global_surfaces: bool = True
    random_seed: int = 7
    train_frac: float = 0.60
    val_frac: float = 0.20
    rerank_random_search_trials: int = 240
    rerank_tuning_max_queries: int = 30_000
    bootstrap_samples: int = 1000
    auto_accept_margin: float = 0.0283
    high_conf_margin: float = 0.0638
    low_conf_margin: float = 0.0031
    min_top1_score: float = 0.45
    pubmed_fallback_min_score: float = 0.70
    run_synthetic_open_set: bool = True
    open_set_holdout_target_frac: float = 0.10
    open_set_max_queries: int = 25_000
    large_artifact_threshold_gb: float = 4.0
    part_size_gb: float = 2.0
    pipeline_stage: str = "all"
    n_jobs: int = 4
    prediction_chunk_size: int = 4096
    force_rescore: bool = False

    @property
    def results_dir(self) -> Path:
        return self.project_dir / "results" / self.run_name

    @property
    def artifact_dir(self) -> Path:
        return self.results_dir / "artifacts"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    @property
    def scratch_dir(self) -> Path:
        if Path("/mnt/local-scratch").exists():
            root = Path("/mnt/local-scratch")
        elif Path("/content").exists():
            root = Path("/content")
        else:
            root = self.project_dir / ".scratch"
        return root / "chem_synonymizer" / self.run_name


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


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
        if (candidate / "data").exists() or (candidate / "src").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def build_config(args: argparse.Namespace) -> Config:
    project_dir = find_project_dir(args.project_dir)
    default_model = project_dir / "results" / "sapbert_contrastive" / "artifacts" / "model"
    model_name = env("CHEMSYN_MODEL_NAME_OR_PATH") or args.model_name_or_path
    if not model_name:
        model_name = str(default_model if default_model.exists() else "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    pubmed_candidates = [
        project_dir / "data" / "CID-Synonym-full.csv",
        project_dir / "data" / "good.csv",
    ]
    pubmed_default = next((p for p in pubmed_candidates if p.exists()), pubmed_candidates[0])

    refmet_env = env("CHEMSYN_REFMET_CSV") or args.refmet_csv
    refmet_csv = Path(refmet_env).expanduser() if refmet_env else None
    if refmet_csv is not None and not refmet_csv.exists():
        print(f"Configured RefMet CSV does not exist; falling back to lab refmet_name values: {refmet_csv}")
        refmet_csv = None

    cfg = Config(
        project_dir=project_dir,
        run_name=env("CHEMSYN_RUN_NAME", args.run_name),
        lab_csv=Path(env("CHEMSYN_LAB_CSV") or args.lab_csv or project_dir / "data" / "allmetnames_062626.csv"),
        refmet_csv=refmet_csv,
        pubmed_csv=Path(env("CHEMSYN_PUBMED_CSV") or args.pubmed_csv or pubmed_default),
        model_name_or_path=model_name,
        device=env("CHEMSYN_DEVICE", args.device),
        max_length=env_int("CHEMSYN_MAX_LENGTH", args.max_length),
        pooling=env("CHEMSYN_POOLING", args.pooling),
        refmet_batch_size=env_int("CHEMSYN_REFMET_BATCH_SIZE", args.refmet_batch_size),
        query_batch_size=env_int("CHEMSYN_QUERY_BATCH_SIZE", args.query_batch_size),
        pubmed_batch_size=env_int("CHEMSYN_PUBMED_BATCH_SIZE", args.pubmed_batch_size),
        top_x=env_int("CHEMSYN_TOP_X", args.top_x),
        k_retrieve_refmet=env_int("CHEMSYN_K_RETRIEVE_REFMET", args.k_retrieve_refmet),
        rerank_pool_k=env_int("CHEMSYN_RERANK_POOL_K", args.rerank_pool_k),
        k_retrieve_pubmed=env_int("CHEMSYN_K_RETRIEVE_PUBMED", args.k_retrieve_pubmed),
        search_batch_size=env_int("CHEMSYN_SEARCH_BATCH_SIZE", args.search_batch_size),
        exact_gpu_max_vectors=env_int("CHEMSYN_EXACT_GPU_MAX_VECTORS", args.exact_gpu_max_vectors),
        pubmed_search_mode=env("CHEMSYN_PUBMED_SEARCH_MODE", args.pubmed_search_mode),
        pubmed_gpu_chunk_vectors=env_int("CHEMSYN_PUBMED_GPU_CHUNK_VECTORS", args.pubmed_gpu_chunk_vectors),
        pubmed_fallback_query_mode=env("CHEMSYN_PUBMED_FALLBACK_QUERY_MODE", args.pubmed_fallback_query_mode),
        auto_accept_target_precision=env_float("CHEMSYN_AUTO_ACCEPT_TARGET_PRECISION", args.auto_accept_target_precision),
        fallback_target_hit10=env_float("CHEMSYN_FALLBACK_TARGET_HIT10", args.fallback_target_hit10),
        run_pubmed_fallback=env_bool("CHEMSYN_RUN_PUBMED_FALLBACK", not args.no_pubmed_fallback),
        rebuild_pubmed_condensed=env_bool("CHEMSYN_REBUILD_PUBMED_CONDENSED", args.rebuild_pubmed_condensed),
        pubmed_max_names_per_cid=env_int("CHEMSYN_PUBMED_MAX_NAMES_PER_CID", args.pubmed_max_names_per_cid),
        pubmed_embed_row_limit=env_int("CHEMSYN_PUBMED_EMBED_ROW_LIMIT", args.pubmed_embed_row_limit),
        rerank_random_search_trials=env_int("CHEMSYN_RERANK_RANDOM_SEARCH_TRIALS", args.rerank_random_search_trials),
        rerank_tuning_max_queries=env_int("CHEMSYN_RERANK_TUNING_MAX_QUERIES", args.rerank_tuning_max_queries),
        bootstrap_samples=env_int("CHEMSYN_BOOTSTRAP_SAMPLES", args.bootstrap_samples),
        pipeline_stage=env("CHEMSYN_PIPELINE_STAGE", args.pipeline_stage),
        n_jobs=env_int("CHEMSYN_N_JOBS", args.n_jobs),
        prediction_chunk_size=env_int("CHEMSYN_PREDICTION_CHUNK_SIZE", args.prediction_chunk_size),
        force_rescore=env_bool("CHEMSYN_FORCE_RESCORE", args.force_rescore),
    )
    cfg.rerank_pool_k = min(cfg.rerank_pool_k, cfg.k_retrieve_refmet)
    return cfg


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = text.replace("\u2212", "-").replace("µ", "u")
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
    return {part.strip() for part in CID_SPLIT_RE.split(text) if part.strip()}


def cid_string(cids: Iterable[str]) -> str:
    values = sorted(str(c) for c in cids if str(c))
    return "|".join(values)


def detect_column(columns: Iterable[str], candidates: list[str], required: bool = True, label: str = "column") -> str | None:
    lower = {str(c).lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower().strip() in lower:
            return lower[candidate.lower().strip()]
    if required:
        raise ValueError(f"Could not detect {label}. Tried {candidates}. Available columns: {list(columns)}")
    return None


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def embedding_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def embedding_meta_matches(path: Path, expected_meta: dict | None) -> bool:
    if not expected_meta:
        return True
    meta_path = embedding_meta_path(path)
    if not meta_path.exists():
        return True
    try:
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, expected in expected_meta.items():
        if str(actual.get(key)) != str(expected):
            return False
    return True


def write_embedding_meta(path: Path, arr: np.ndarray, meta: dict | None) -> None:
    payload = dict(meta or {})
    payload.update(
        {
            "embedding_path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_bytes": int(Path(path).stat().st_size) if Path(path).exists() else int(arr.nbytes),
        }
    )
    save_json(embedding_meta_path(path), payload)


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def backup_file_in_parts(src: Path, dest_parts_dir: Path, part_size_gb: float = 2.0, force: bool = False) -> Path:
    src = Path(src)
    dest_parts_dir = Path(dest_parts_dir)
    dest_parts_dir.mkdir(parents=True, exist_ok=True)
    part_size = int(part_size_gb * 1024**3)
    size = src.stat().st_size
    n_parts = math.ceil(size / part_size)
    manifest = {
        "source_name": src.name,
        "source_size_bytes": size,
        "part_size_bytes": part_size,
        "n_parts": n_parts,
    }
    manifest_path = dest_parts_dir / "manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_size_bytes") == size and existing.get("n_parts") == n_parts:
            print(f"Parts already checkpointed for {src.name}")
            return manifest_path
    with src.open("rb") as in_f:
        for i in range(n_parts):
            part_path = dest_parts_dir / f"part_{i:05d}.bin"
            remaining = min(part_size, size - i * part_size)
            if part_path.exists() and part_path.stat().st_size == remaining and not force:
                print(f"SKIP part {i + 1}/{n_parts}")
                in_f.seek(min((i + 1) * part_size, size))
                continue
            print(f"Checkpointing {src.name} part {i + 1}/{n_parts}: {bytes_to_gb(remaining):.2f} GB")
            with part_path.open("wb") as out_f:
                copied = 0
                while copied < remaining:
                    chunk = in_f.read(min(64 * 1024 * 1024, remaining - copied))
                    if not chunk:
                        break
                    out_f.write(chunk)
                    copied += len(chunk)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def restore_file_from_parts(parts_dir: Path, dest_path: Path, force: bool = False) -> Path | None:
    parts_dir = Path(parts_dir)
    dest_path = Path(dest_path)
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


def restore_or_none(local_path: Path, drive_path: Path, cfg: Config) -> Path | None:
    if local_path.exists():
        return local_path
    if drive_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Restoring {drive_path.name} to scratch...")
        shutil.copyfile(drive_path, local_path)
        return local_path
    parts_dir = drive_path.with_suffix(drive_path.suffix + ".parts")
    return restore_file_from_parts(parts_dir, local_path)


def checkpoint_artifact(local_path: Path, drive_path: Path, cfg: Config) -> None:
    local_path = Path(local_path)
    drive_path = Path(drive_path)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    size_gb = bytes_to_gb(local_path.stat().st_size)
    if size_gb >= cfg.large_artifact_threshold_gb:
        backup_file_in_parts(local_path, drive_path.with_suffix(drive_path.suffix + ".parts"), cfg.part_size_gb)
    else:
        if not drive_path.exists() or drive_path.stat().st_size != local_path.stat().st_size:
            print(f"Checkpointing {local_path.name} ({size_gb:.2f} GB) to results...")
            shutil.copyfile(local_path, drive_path)


def checkpoint_embedding_artifact(local_path: Path, drive_path: Path, cfg: Config, meta: dict | None = None) -> None:
    print(f"Checkpointing completed embedding cache before downstream search: {local_path.name}")
    checkpoint_artifact(local_path, drive_path, cfg)
    arr = np.load(local_path, mmap_mode="r")
    write_embedding_meta(local_path, arr, meta)
    write_embedding_meta(drive_path, arr, meta)
    print(f"Embedding checkpoint complete; safe to resume from: {drive_path}")


def artifact_or_parts_exists(path: Path) -> bool:
    return path.exists() or (path.with_suffix(path.suffix + ".parts") / "manifest.json").exists()


def load_lab_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Lab CSV not found: {path}")
    lab = pd.read_csv(path, dtype="string", keep_default_na=False, on_bad_lines="skip")
    metabolite_col = detect_column(lab.columns, ["metabolite_name", "metabolite", "name", "compound_name"], label="metabolite_name")
    refmet_col = detect_column(lab.columns, ["refmet_name", "refmet", "refmet name", "common_name"], required=False, label="refmet_name")
    lab = lab.rename(columns={metabolite_col: "metabolite_name"})
    if refmet_col and refmet_col != "refmet_name":
        lab = lab.rename(columns={refmet_col: "refmet_name"})
    if "refmet_name" not in lab.columns:
        lab["refmet_name"] = ""
    lab = lab.reset_index(drop=True)
    lab["query_i"] = np.arange(len(lab), dtype=np.int64)
    lab["source_row_id"] = lab["query_i"]
    lab["metabolite_name"] = lab["metabolite_name"].astype("string").fillna("").str.strip()
    lab["refmet_name"] = lab["refmet_name"].astype("string").fillna("").str.strip()
    lab["metabolite_normalized"] = lab["metabolite_name"].map(normalize_name)
    lab["refmet_normalized"] = lab["refmet_name"].map(normalize_name)
    lab["metabolite_compact"] = lab["metabolite_name"].map(compact_text)
    lab["refmet_compact"] = lab["refmet_name"].map(compact_text)
    lab["has_metabolite_name"] = lab["metabolite_normalized"].str.len() > 0
    lab["has_refmet_name"] = lab["refmet_normalized"].str.len() > 0
    return lab


def load_refmet_targets(cfg: Config, lab: pd.DataFrame) -> pd.DataFrame:
    if cfg.refmet_csv and cfg.refmet_csv.exists():
        raw = pd.read_csv(cfg.refmet_csv, dtype="string", keep_default_na=False, on_bad_lines="skip")
        name_col = detect_column(raw.columns, ["refmet_name", "name", "display_name", "common_name"], label="RefMet name")
        id_col = detect_column(raw.columns, ["refmet_id", "id", "refmetid"], required=False, label="RefMet ID")
        cid_col = detect_column(raw.columns, ["pubchem_cid", "cid", "pubchem id", "pubchem"], required=False, label="PubChem CID")
        targets = pd.DataFrame(
            {
                "refmet_name": raw[name_col].astype("string").fillna("").str.strip(),
                "refmet_id": raw[id_col].astype("string").fillna("").str.strip() if id_col else "",
                "refmet_cid": raw[cid_col].astype("string").fillna("").str.strip() if cid_col else "",
                "target_source": str(cfg.refmet_csv),
            }
        )
    else:
        values = lab.loc[lab["has_refmet_name"], "refmet_name"].drop_duplicates().reset_index(drop=True)
        targets = pd.DataFrame({"refmet_name": values, "refmet_id": "", "refmet_cid": "", "target_source": "lab_unique_refmet_names"})
    targets["refmet_normalized"] = targets["refmet_name"].map(normalize_name)
    targets["refmet_compact"] = targets["refmet_name"].map(compact_text)
    targets = targets[targets["refmet_normalized"].str.len() > 0].copy()
    targets = targets.drop_duplicates("refmet_normalized").reset_index(drop=True)
    targets["target_id"] = np.arange(len(targets), dtype=np.int64)
    targets["refmet_cid_set"] = targets["refmet_cid"].map(parse_cids)
    targets["refmet_cid"] = targets["refmet_cid_set"].map(cid_string)
    return targets


def split_lab(lab: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    valid = lab[lab["has_metabolite_name"] & lab["has_refmet_name"] & lab["gold_mapped_to_target"]].copy()
    lab["split"] = "unlabeled_or_unmapped"
    if len(valid) < 10:
        return lab
    test_size = max(0.0, 1.0 - cfg.train_frac - cfg.val_frac)
    train_val, test = train_test_split(valid, test_size=test_size, random_state=cfg.random_seed, shuffle=True)
    val_size = cfg.val_frac / (cfg.train_frac + cfg.val_frac)
    train, val = train_test_split(train_val, test_size=val_size, random_state=cfg.random_seed, shuffle=True)
    lab.loc[train.index, "split"] = "train"
    lab.loc[val.index, "split"] = "validation"
    lab.loc[test.index, "split"] = "test"
    return lab


def build_pubmed_condensed(cfg: Config) -> tuple[Path | None, dict]:
    if not cfg.run_pubmed_fallback:
        return None, {"run_pubmed_fallback": False}
    if not cfg.pubmed_csv or not cfg.pubmed_csv.exists():
        print(f"PubMed/PubChem synonym CSV not found; skipping fallback: {cfg.pubmed_csv}")
        return None, {"run_pubmed_fallback": False, "missing_pubmed_csv": str(cfg.pubmed_csv)}
    out_parquet = cfg.artifact_dir / f"pubmed_condensed_top{cfg.pubmed_max_names_per_cid}_per_cid.parquet"
    con = duckdb.connect()
    if out_parquet.exists() and not cfg.rebuild_pubmed_condensed:
        print(f"Using cached condensed PubMed/PubChem parquet: {out_parquet}")
    else:
        print("Condensing PubMed/PubChem synonyms by CID...")
        con.execute(
            f"""
            COPY (
              WITH raw AS (
                SELECT CAST(cid AS VARCHAR) AS cid, CAST(name AS VARCHAR) AS display_name
                FROM read_csv_auto('{sql_path(cfg.pubmed_csv)}', all_varchar=true, ignore_errors=true)
                WHERE cid IS NOT NULL AND name IS NOT NULL
              ), normalized AS (
                SELECT cid, trim(display_name) AS display_name,
                       regexp_replace(lower(trim(display_name)), '\\s+', ' ', 'g') AS normalized_name
                FROM raw
                WHERE length(trim(display_name)) > 0
              ), deduped AS (
                SELECT cid, normalized_name, any_value(display_name) AS display_name
                FROM normalized
                WHERE length(normalized_name) > 0
                GROUP BY cid, normalized_name
              ), scored AS (
                SELECT cid, display_name, normalized_name,
                       regexp_replace(normalized_name, '[^a-z0-9]+', '', 'g') AS compact_name,
                       length(normalized_name) AS name_len,
                       (
                         CASE WHEN regexp_matches(normalized_name, '^[0-9]+$') THEN -10 ELSE 0 END +
                         CASE WHEN length(normalized_name) BETWEEN 4 AND 80 THEN 3 ELSE 0 END +
                         CASE WHEN regexp_matches(normalized_name, '[a-z]') THEN 2 ELSE 0 END +
                         CASE WHEN regexp_matches(normalized_name, '[0-9].*[a-z]|[a-z].*[0-9]') THEN 1 ELSE 0 END +
                         CASE WHEN regexp_matches(normalized_name, '(acid|ate|ine|ol|ose|ide|lipid|phosphate|carnitine|coenzyme|vitamin)') THEN 2 ELSE 0 END -
                         CASE WHEN regexp_matches(normalized_name, '(deprecated|unknown|mixture|unspecified|not available)') THEN 4 ELSE 0 END
                       ) AS quality_score
                FROM deduped
                WHERE length(regexp_replace(normalized_name, '[^a-z0-9]+', '', 'g')) > 1
              ), ranked AS (
                SELECT *, row_number() OVER (
                  PARTITION BY cid ORDER BY quality_score DESC, name_len ASC, normalized_name ASC
                ) AS cid_name_rank
                FROM scored
              )
              SELECT cid, display_name, normalized_name, compact_name, name_len, quality_score, cid_name_rank
              FROM ranked
              WHERE cid_name_rank <= {int(cfg.pubmed_max_names_per_cid)}
            ) TO '{sql_path(out_parquet)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    summary = con.execute(
        f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT cid) AS n_unique_cids,
               count(DISTINCT normalized_name) AS n_unique_normalized_surfaces,
               max(cid_name_rank) AS max_names_per_cid_after_condensing
        FROM read_parquet('{sql_path(out_parquet)}')
        """
    ).df().iloc[0].to_dict()
    con.close()
    summary = {k: int(v) if pd.notna(v) and float(v).is_integer() else float(v) if pd.notna(v) else None for k, v in summary.items()}
    summary["pubmed_csv"] = str(cfg.pubmed_csv)
    summary["condensed_parquet"] = str(out_parquet)
    summary["pubmed_max_names_per_cid"] = int(cfg.pubmed_max_names_per_cid)
    save_json(cfg.results_dir / "pubmed_condensation_summary.json", summary)
    return out_parquet, summary


def augment_refmet_cids_from_pubmed(cfg: Config, refmet_targets: pd.DataFrame, pubmed_parquet: Path | None) -> pd.DataFrame:
    """Attach PubMed/PubChem CIDs to RefMet targets by exact normalized name.

    Prefer the full synonym CSV for this mapping so CID coverage is not limited
    by the top-N-per-CID condensed fallback surface. If the full CSV is missing,
    fall back to the condensed parquet.
    """
    con = duckdb.connect()
    con.register("refmet_target_norms", refmet_targets[["refmet_normalized"]].drop_duplicates())
    if cfg.pubmed_csv and cfg.pubmed_csv.exists():
        print("Mapping RefMet targets to PubMed/PubChem CIDs using the full synonym CSV...")
        cid_map = con.execute(
            f"""
            WITH raw AS (
              SELECT CAST(cid AS VARCHAR) AS cid,
                     regexp_replace(lower(trim(CAST(name AS VARCHAR))), '\\s+', ' ', 'g') AS normalized_name
              FROM read_csv_auto('{sql_path(cfg.pubmed_csv)}', all_varchar=true, ignore_errors=true)
              WHERE cid IS NOT NULL AND name IS NOT NULL
            )
            SELECT t.refmet_normalized,
                   string_agg(DISTINCT raw.cid, '|') AS pubmed_exact_match_cids
            FROM refmet_target_norms t
            JOIN raw ON t.refmet_normalized = raw.normalized_name
            GROUP BY t.refmet_normalized
            """
        ).df()
    elif pubmed_parquet is not None and pubmed_parquet.exists():
        cid_map = con.execute(
            f"""
            SELECT normalized_name AS refmet_normalized,
                   string_agg(DISTINCT cid, '|') AS pubmed_exact_match_cids
            FROM read_parquet('{sql_path(pubmed_parquet)}')
            GROUP BY normalized_name
            """
        ).df()
    else:
        con.close()
        return refmet_targets
    con.close()
    out = refmet_targets.drop(columns=["refmet_cid_set"], errors="ignore").merge(cid_map, on="refmet_normalized", how="left")
    out["refmet_cid"] = out.apply(
        lambda r: r["refmet_cid"] if isinstance(r.get("refmet_cid"), str) and r["refmet_cid"] else (r.get("pubmed_exact_match_cids") or ""),
        axis=1,
    )
    out["refmet_cid_set"] = out["refmet_cid"].map(parse_cids)
    summary = {
        "n_refmet_targets": int(len(out)),
        "n_refmet_targets_with_pubmed_cid": int(out["refmet_cid"].astype(str).str.len().gt(0).sum()),
        "coverage_fraction": float(out["refmet_cid"].astype(str).str.len().gt(0).mean()) if len(out) else 0.0,
        "mapping_source": str(cfg.pubmed_csv if cfg.pubmed_csv and cfg.pubmed_csv.exists() else pubmed_parquet),
    }
    save_json(cfg.results_dir / "refmet_cid_mapping_summary.json", summary)
    return out


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_hf_model(model_name_or_path: str, requested_device: str):
    device = get_device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return tokenizer, model, device


def pool_outputs(outputs, attention_mask, pooling: str):
    if pooling == "cls":
        return outputs.last_hidden_state[:, 0]
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def encode_texts_to_npy(
    texts: list[str],
    local_path: Path,
    drive_path: Path,
    tokenizer,
    model,
    device: torch.device,
    cfg: Config,
    batch_size: int,
    label: str,
    cache_meta: dict | None = None,
):
    expected_meta = {
        "label": label,
        "model_name_or_path": cfg.model_name_or_path,
        "max_length": cfg.max_length,
        "pooling": cfg.pooling,
        "embed_dtype": cfg.embed_dtype,
        "n_texts": len(texts),
        **(cache_meta or {}),
    }
    restored = restore_or_none(local_path, drive_path, cfg)
    if restored is not None:
        arr = np.load(restored, mmap_mode="r")
        if arr.shape[0] == len(texts) and embedding_meta_matches(drive_path, expected_meta):
            print(f"Using cached {label} embeddings: {restored} {arr.shape} {arr.dtype}")
            write_embedding_meta(restored, arr, expected_meta)
            write_embedding_meta(drive_path, arr, expected_meta)
            return restored
        print(f"Cached {label} shape/config mismatch; rebuilding.")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sample = tokenizer(texts[:1] or [""], padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
    sample = {k: v.to(device) for k, v in sample.items()}
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        dim = int(pool_outputs(model(**sample), sample["attention_mask"], cfg.pooling).shape[1])
    dtype = np.float16 if cfg.embed_dtype == "float16" else np.float32
    mmap = np.lib.format.open_memmap(local_path, mode="w+", dtype=dtype, shape=(len(texts), dim))
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {label}"):
        batch = [str(x) if x is not None else "" for x in texts[start : start + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            emb = pool_outputs(model(**encoded), encoded["attention_mask"], cfg.pooling)
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        mmap[start : start + len(batch)] = emb.detach().cpu().numpy().astype(dtype)
    mmap.flush()
    del mmap
    checkpoint_embedding_artifact(local_path, drive_path, cfg, expected_meta)
    return local_path


def make_exact_index(xb: np.ndarray, cfg: Config):
    cpu_index = faiss.IndexFlatIP(xb.shape[1])
    cpu_index.add(xb.astype("float32", copy=False))
    if torch.cuda.is_available() and xb.shape[0] <= cfg.exact_gpu_max_vectors and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            return gpu_index, res
        except Exception as exc:
            print(f"GPU FAISS exact index failed; using CPU index. Reason: {exc!r}")
    return cpu_index, None


def search_embeddings(index, query_embeddings: np.ndarray, k: int, batch_size: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = [], []
    for start in tqdm(range(0, query_embeddings.shape[0], batch_size), desc=label):
        q = np.asarray(query_embeddings[start : start + batch_size], dtype="float32")
        s, i = index.search(q, k)
        scores.append(s)
        indices.append(i)
    return np.vstack(scores), np.vstack(indices)


def has_gpu_faiss() -> bool:
    return bool(torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "GpuIndexFlatIP"))


def make_gpu_flat_index_for_chunk(xb_chunk: np.ndarray):
    res = faiss.StandardGpuResources()
    gpu_config = faiss.GpuIndexFlatConfig()
    gpu_config.device = 0
    gpu_index = faiss.GpuIndexFlatIP(res, xb_chunk.shape[1], gpu_config)
    gpu_index.add(np.asarray(xb_chunk, dtype="float32"))
    return gpu_index, res


def merge_topk(existing_scores: np.ndarray, existing_indices: np.ndarray, new_scores: np.ndarray, new_indices: np.ndarray, k: int):
    combined_scores = np.concatenate([existing_scores, new_scores], axis=1)
    combined_indices = np.concatenate([existing_indices, new_indices], axis=1)
    keep = np.argpartition(-combined_scores, kth=min(k - 1, combined_scores.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(combined_scores.shape[0])[:, None]
    kept_scores = combined_scores[rows, keep]
    kept_indices = combined_indices[rows, keep]
    order = np.argsort(-kept_scores, axis=1)
    return kept_scores[rows, order], kept_indices[rows, order]


def search_embeddings_streaming_gpu(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_ids: np.ndarray,
    k: int,
    cfg: Config,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact search by repeatedly loading corpus chunks onto the GPU.

    This preserves exact dense retrieval over the full corpus while bounding GPU
    memory. It is intended for PubMed/PubChem fallback, where the corpus can be
    far larger than one GPU-resident FAISS flat index.
    """
    if not has_gpu_faiss():
        raise RuntimeError("GPU FAISS is not available; cannot use streaming GPU search.")
    query_ids = np.asarray(query_ids, dtype=np.int64)
    n_queries = query_embeddings.shape[0]
    full_scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
    full_indices = np.full((n_queries, k), -1, dtype=np.int64)
    if len(query_ids) == 0:
        return full_scores, full_indices

    top_scores = np.full((len(query_ids), k), -np.inf, dtype=np.float32)
    top_indices = np.full((len(query_ids), k), -1, dtype=np.int64)
    chunk_size = max(k, int(cfg.pubmed_gpu_chunk_vectors))
    n_chunks = math.ceil(corpus_embeddings.shape[0] / chunk_size)
    print(
        f"{label}: exact GPU streaming over {corpus_embeddings.shape[0]:,} corpus vectors, "
        f"{len(query_ids):,} queries, chunk_size={chunk_size:,}, chunks={n_chunks:,}"
    )

    for c_start in tqdm(range(0, corpus_embeddings.shape[0], chunk_size), desc=f"{label}: corpus chunks"):
        c_end = min(c_start + chunk_size, corpus_embeddings.shape[0])
        xb_chunk = np.asarray(corpus_embeddings[c_start:c_end], dtype="float32")
        index, resources = make_gpu_flat_index_for_chunk(xb_chunk)
        this_k = min(k, c_end - c_start)
        for q_start in range(0, len(query_ids), cfg.search_batch_size):
            q_end = min(q_start + cfg.search_batch_size, len(query_ids))
            q_batch_ids = query_ids[q_start:q_end]
            q = np.asarray(query_embeddings[q_batch_ids], dtype="float32")
            new_scores, new_indices = index.search(q, this_k)
            new_indices = new_indices.astype(np.int64) + c_start
            old_scores = top_scores[q_start:q_end]
            old_indices = top_indices[q_start:q_end]
            merged_scores, merged_indices = merge_topk(old_scores, old_indices, new_scores.astype("float32"), new_indices, k)
            top_scores[q_start:q_end] = merged_scores
            top_indices[q_start:q_end] = merged_indices
        del index, resources, xb_chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_scores[query_ids] = top_scores
    full_indices[query_ids] = top_indices
    return full_scores, full_indices


def run_refmet_search(cfg: Config, refmet_emb_path: Path, query_emb_path: Path) -> tuple[Path, Path]:
    local_scores = cfg.scratch_dir / "refmet_neighbor_scores.npy"
    local_indices = cfg.scratch_dir / "refmet_neighbor_indices.npy"
    drive_scores = cfg.artifact_dir / "refmet_neighbor_scores.npy"
    drive_indices = cfg.artifact_dir / "refmet_neighbor_indices.npy"
    if restore_or_none(local_scores, drive_scores, cfg) and restore_or_none(local_indices, drive_indices, cfg):
        return local_scores, local_indices
    target_embeddings = np.load(refmet_emb_path, mmap_mode="r")
    query_embeddings = np.load(query_emb_path, mmap_mode="r")
    index, resources = make_exact_index(np.asarray(target_embeddings, dtype="float32"), cfg)
    scores, indices = search_embeddings(index, query_embeddings, cfg.k_retrieve_refmet, cfg.search_batch_size, "Searching RefMet targets")
    np.save(local_scores, scores.astype("float32"))
    np.save(local_indices, indices.astype("int64"))
    checkpoint_artifact(local_scores, drive_scores, cfg)
    checkpoint_artifact(local_indices, drive_indices, cfg)
    del index, resources, scores, indices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return local_scores, local_indices


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


def make_feature_block(
    qids: np.ndarray,
    lab: pd.DataFrame,
    targets: pd.DataFrame,
    neighbor_scores: np.ndarray,
    neighbor_indices: np.ndarray,
    cfg: Config,
    strict_no_query_surface: bool = False,
    exclude_target_ids: set[int] | None = None,
):
    pool_k = min(cfg.rerank_pool_k, neighbor_indices.shape[1])
    qids = np.asarray(qids, dtype=np.int64)
    feats = np.zeros((len(qids), pool_k, len(FEATURE_COLUMNS)), dtype=np.float32)
    target_ids = np.full((len(qids), pool_k), -1, dtype=np.int64)
    target_pos = np.full((len(qids), pool_k), -1, dtype=np.int64)
    invalid = np.zeros((len(qids), pool_k), dtype=bool)
    target_ids_all = targets["target_id"].to_numpy(dtype=np.int64)
    target_norm = targets["refmet_normalized"].astype(str).to_numpy()
    target_compact = targets["refmet_compact"].astype(str).to_numpy()
    q_norm_all = lab["metabolite_normalized"].astype(str).to_numpy()
    q_compact_all = lab["metabolite_compact"].astype(str).to_numpy()
    exclude_target_ids = exclude_target_ids or set()
    for row_i, qid in enumerate(qids):
        q_norm = q_norm_all[qid]
        q_compact = q_compact_all[qid]
        for j in range(pool_k):
            pos = int(neighbor_indices[qid, j])
            if pos < 0:
                invalid[row_i, j] = True
                continue
            tid = int(target_ids_all[pos])
            t_norm = target_norm[pos]
            if tid in exclude_target_ids or (strict_no_query_surface and q_norm == t_norm):
                invalid[row_i, j] = True
            target_ids[row_i, j] = tid
            target_pos[row_i, j] = pos
            feats[row_i, j] = feature_values(q_norm, q_compact, t_norm, target_compact[pos], float(neighbor_scores[qid, j]), j)
    return feats, target_ids, target_pos, invalid


def base_gold_ranks(qids: np.ndarray, lab: pd.DataFrame, targets: pd.DataFrame, neighbor_indices: np.ndarray) -> np.ndarray:
    target_ids_all = targets["target_id"].to_numpy(dtype=np.int64)
    gold = lab["gold_target_id"].fillna(-1).astype(int).to_numpy()
    ranks = np.full(len(qids), np.inf, dtype=np.float32)
    for row_i, qid in enumerate(qids):
        gold_id = int(gold[qid])
        if gold_id < 0:
            continue
        tids = target_ids_all[neighbor_indices[qid]]
        found = np.flatnonzero(tids == gold_id)
        if len(found):
            ranks[row_i] = float(found[0] + 1)
    return ranks


def ranks_for_weights(feats, target_ids, gold_ids, invalid, weights, fallback_ranks=None):
    weight_vec = np.array([weights[c] for c in FEATURE_COLUMNS], dtype=np.float32)
    scores = np.tensordot(feats, weight_vec, axes=([2], [0]))
    scores = np.where(invalid, -np.inf, scores)
    order = np.argsort(-scores, axis=1)
    ordered_tids = np.take_along_axis(target_ids, order, axis=1)
    match = ordered_tids == gold_ids.reshape(-1, 1)
    ranks = np.full(len(gold_ids), np.inf, dtype=np.float32)
    for i in range(len(gold_ids)):
        found = np.flatnonzero(match[i])
        if len(found):
            ranks[i] = float(found[0] + 1)
        elif fallback_ranks is not None:
            ranks[i] = fallback_ranks[i]
    return ranks


def rank_metrics(ranks: Iterable[float], ks=(1, 5, 10, 50, 256)) -> dict:
    arr = np.asarray([r for r in ranks if np.isfinite(r)], dtype=np.float32)
    out = {"n": int(len(list(ranks))) if not isinstance(ranks, np.ndarray) else int(len(ranks))}
    if out["n"] == 0:
        out.update({"MRR": np.nan, "median_rank": np.nan})
        for k in ks:
            out[f"Hit@{k}"] = np.nan
        return out
    full = np.asarray(ranks, dtype=np.float32)
    reciprocal = np.where(np.isfinite(full), 1.0 / full, 0.0)
    out["MRR"] = float(np.mean(reciprocal))
    for k in ks:
        out[f"Hit@{k}"] = float(np.mean(full <= k))
    out["median_rank"] = float(np.median(arr)) if len(arr) else np.inf
    return out


_PREDICT_WORKER = {}


def _init_predict_worker(
    lab_payload: dict,
    target_payload: dict,
    scores_path: str,
    indices_path: str,
    weights: dict,
    cfg_payload: dict,
    strict_no_query_surface: bool,
    exclude_target_ids: list[int] | None,
):
    global _PREDICT_WORKER
    _PREDICT_WORKER = {
        "lab": lab_payload,
        "targets": target_payload,
        "scores": np.load(scores_path, mmap_mode="r"),
        "indices": np.load(indices_path, mmap_mode="r"),
        "weight_vec": np.array([weights[c] for c in FEATURE_COLUMNS], dtype=np.float32),
        "cfg": cfg_payload,
        "strict_no_query_surface": strict_no_query_surface,
        "exclude_target_ids": set(exclude_target_ids or []),
    }


def _payload_from_lab(lab: pd.DataFrame) -> dict:
    return {
        "source_row_id": lab["source_row_id"].astype(int).to_numpy(),
        "split": lab["split"].astype(str).to_numpy(),
        "metabolite_name": lab["metabolite_name"].astype(str).to_numpy(),
        "metabolite_normalized": lab["metabolite_normalized"].astype(str).to_numpy(),
        "metabolite_compact": lab["metabolite_compact"].astype(str).to_numpy(),
        "refmet_name": lab["refmet_name"].astype(str).to_numpy(),
        "gold_target_id": lab["gold_target_id"].fillna(-1).astype(int).to_numpy(),
    }


def _payload_from_targets(targets: pd.DataFrame) -> dict:
    return {
        "target_id": targets["target_id"].astype(int).to_numpy(),
        "refmet_name": targets["refmet_name"].astype(str).to_numpy(),
        "refmet_normalized": targets["refmet_normalized"].astype(str).to_numpy(),
        "refmet_compact": targets["refmet_compact"].astype(str).to_numpy(),
        "refmet_cid": targets["refmet_cid"].fillna("").astype(str).to_numpy(),
    }


def _predict_chunk_worker(qids: list[int]) -> tuple[list[dict], list[dict]]:
    state = _PREDICT_WORKER
    lab = state["lab"]
    targets = state["targets"]
    scores = state["scores"]
    indices = state["indices"]
    weight_vec = state["weight_vec"]
    cfg = state["cfg"]
    strict_no_query_surface = state["strict_no_query_surface"]
    exclude_target_ids = state["exclude_target_ids"]
    pool_k = min(int(cfg["rerank_pool_k"]), indices.shape[1])
    top_x = int(cfg["top_x"])

    target_ids_all = targets["target_id"]
    target_names = targets["refmet_name"]
    target_norm = targets["refmet_normalized"]
    target_compact = targets["refmet_compact"]
    target_cids = targets["refmet_cid"]

    pred_rows = []
    wide_rows = []
    for qid in qids:
        qid = int(qid)
        q_norm = lab["metabolite_normalized"][qid]
        q_compact = lab["metabolite_compact"][qid]
        gold_id = int(lab["gold_target_id"][qid])
        candidate_scores = []

        for j in range(pool_k):
            pos = int(indices[qid, j])
            if pos < 0:
                continue
            tid = int(target_ids_all[pos])
            t_norm = target_norm[pos]
            if tid in exclude_target_ids or (strict_no_query_surface and q_norm == t_norm):
                continue
            feats = feature_values(q_norm, q_compact, t_norm, target_compact[pos], float(scores[qid, j]), j)
            candidate_scores.append((float(np.dot(weight_vec, np.asarray(feats, dtype=np.float32))), j, pos, tid))

        candidate_scores.sort(key=lambda item: (-item[0], item[1]))
        rank = np.inf
        for new_rank, (_, _, _, tid) in enumerate(candidate_scores, start=1):
            if tid == gold_id:
                rank = float(new_rank)
                break
        if not np.isfinite(rank):
            for j in range(indices.shape[1]):
                pos = int(indices[qid, j])
                if pos >= 0 and int(target_ids_all[pos]) == gold_id:
                    rank = float(j + 1)
                    break

        top = candidate_scores[:top_x]
        if top:
            top1_score, _, top1_pos, top1_tid = top[0]
            top2_score = top[1][0] if len(top) > 1 else -np.inf
            pred_name = target_names[top1_pos]
            pred_norm = target_norm[top1_pos]
            pred_cid = target_cids[top1_pos]
        else:
            top1_score = np.nan
            top2_score = np.nan
            top1_tid = -1
            pred_name = ""
            pred_norm = ""
            pred_cid = ""

        base = {
            "query_i": qid,
            "source_row_id": int(lab["source_row_id"][qid]),
            "split": lab["split"][qid],
            "metabolite_name": lab["metabolite_name"][qid],
            "gold_refmet_name": lab["refmet_name"][qid],
            "gold_target_id": gold_id,
        }
        pred_rows.append(
            {
                **base,
                "predicted_refmet_name": pred_name,
                "predicted_refmet_normalized": pred_norm,
                "predicted_target_id": int(top1_tid),
                "predicted_refmet_cid": pred_cid,
                "prediction_score": top1_score,
                "second_score": top2_score,
                "score_margin": top1_score - top2_score if np.isfinite(top2_score) else np.nan,
                "gold_rank": rank,
                "top1_exact_correct": int(top1_tid) == gold_id,
                "top1_relation_to_query": surface_relation(lab["metabolite_name"][qid], pred_name),
            }
        )
        wide = dict(base)
        for display_rank, (candidate_score, original_rank0, pos, tid) in enumerate(top, start=1):
            name = target_names[pos]
            wide[f"top{display_rank}_refmet_name"] = name
            wide[f"top{display_rank}_target_id"] = int(tid)
            wide[f"top{display_rank}_score"] = float(candidate_score)
            wide[f"top{display_rank}_faiss_rank"] = int(original_rank0 + 1)
            wide[f"top{display_rank}_is_gold"] = int(tid) == gold_id
            wide[f"top{display_rank}_relation_to_query"] = surface_relation(lab["metabolite_name"][qid], name)
        wide_rows.append(wide)
    return pred_rows, wide_rows


def tune_reranker(cfg: Config, lab: pd.DataFrame, targets: pd.DataFrame, scores_path: Path, indices_path: Path) -> dict:
    weights_path = cfg.results_dir / "best_reranker_weights.json"
    if weights_path.exists() and not cfg.force_rescore:
        try:
            weights = json.loads(weights_path.read_text(encoding="utf-8"))
            if all(col in weights for col in FEATURE_COLUMNS):
                print(f"Using cached reranker weights: {weights_path}")
                return {col: float(weights[col]) for col in FEATURE_COLUMNS}
        except Exception as exc:
            print(f"Could not reuse cached reranker weights; retuning. Reason: {exc!r}")

    scores = np.load(scores_path, mmap_mode="r")
    indices = np.load(indices_path, mmap_mode="r")
    val_qids = lab.index[(lab["split"] == "validation") & lab["gold_mapped_to_target"]].to_numpy(dtype=np.int64)
    if len(val_qids) == 0:
        return DEFAULT_WEIGHTS.copy()
    rng = np.random.default_rng(cfg.random_seed)
    if cfg.rerank_tuning_max_queries and len(val_qids) > cfg.rerank_tuning_max_queries:
        val_qids = rng.choice(val_qids, size=cfg.rerank_tuning_max_queries, replace=False)
    feats, tids, _, invalid = make_feature_block(val_qids, lab, targets, scores, indices, cfg)
    gold = lab.loc[val_qids, "gold_target_id"].fillna(-1).astype(int).to_numpy()
    fallback = base_gold_ranks(val_qids, lab, targets, indices)
    best_weights = DEFAULT_WEIGHTS.copy()
    best_ranks = ranks_for_weights(feats, tids, gold, invalid, best_weights, fallback)
    best_mrr = rank_metrics(best_ranks)["MRR"]
    log = [{"trial": "default_prior", "validation_MRR": best_mrr, **best_weights}]
    for trial in tqdm(range(cfg.rerank_random_search_trials), desc="Tuning reranker"):
        w = {
            "embedding_score_norm": float(rng.uniform(0.35, 0.85)),
            "rank_prior": float(rng.uniform(0.0, 0.05)),
            "compact_exact": float(rng.uniform(0.0, 0.20)),
            "sequence_ratio": float(rng.uniform(0.00, 0.30)),
            "char3_jaccard": float(rng.uniform(0.00, 0.20)),
            "token_jaccard": float(rng.uniform(0.00, 0.25)),
            "lipid_signature_match": float(rng.uniform(0.00, 0.08)),
            "lipid_class_match": float(rng.uniform(0.00, 0.05)),
            "stereo_mismatch": float(rng.uniform(-0.12, 0.00)),
        }
        ranks = ranks_for_weights(feats, tids, gold, invalid, w, fallback)
        mrr = rank_metrics(ranks)["MRR"]
        log.append({"trial": trial, "validation_MRR": mrr, **w})
        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = w
    pd.DataFrame(log).sort_values("validation_MRR", ascending=False).to_csv(cfg.results_dir / "reranker_random_search.csv", index=False)
    save_json(cfg.results_dir / "best_reranker_weights.json", best_weights)
    print(f"Best validation MRR: {best_mrr:.4f}")
    return best_weights


def surface_relation(query: str, target: str) -> str:
    qn = normalize_name(query)
    tn = normalize_name(target)
    qc = compact_text(query)
    tc = compact_text(target)
    if qn and qn == tn:
        return "exact_normalized"
    if qc and qc == tc:
        return "exact_compact"
    if qn and tn and (qn in tn or tn in qn):
        return "substring"
    if cached_token_set(qn) and cached_token_set(qn) == cached_token_set(tn):
        return "same_tokens"
    return "different"


def predict_refmet(
    cfg: Config,
    lab: pd.DataFrame,
    targets: pd.DataFrame,
    scores_path: Path,
    indices_path: Path,
    weights: dict,
    strict_no_query_surface: bool = False,
    suffix: str = "",
    exclude_target_ids: set[int] | None = None,
) -> pd.DataFrame:
    pred_path = cfg.results_dir / f"refmet_predictions{suffix}.csv"
    wide_path = cfg.results_dir / f"refmet_top{cfg.top_x}_candidates_wide{suffix}.csv"
    if pred_path.exists() and wide_path.exists() and not cfg.force_rescore:
        try:
            cached = pd.read_csv(pred_path)
            if len(cached) == len(lab):
                print(f"Using cached RefMet predictions{suffix}: {pred_path}")
                return cached
        except Exception as exc:
            print(f"Could not reuse cached predictions{suffix}; rebuilding. Reason: {exc!r}")

    scores = np.load(scores_path, mmap_mode="r")
    indices = np.load(indices_path, mmap_mode="r")
    target_names = targets["refmet_name"].astype(str).to_numpy()
    target_norm = targets["refmet_normalized"].astype(str).to_numpy()
    target_cids = targets["refmet_cid"].fillna("").astype(str).to_numpy()
    target_ids_all = targets["target_id"].to_numpy(dtype=np.int64)
    gold_ids = lab["gold_target_id"].fillna(-1).astype(int).to_numpy()
    weight_vec = np.array([weights[c] for c in FEATURE_COLUMNS], dtype=np.float32)

    for path in [wide_path, pred_path]:
        if path.exists():
            path.unlink()

    if cfg.n_jobs and cfg.n_jobs > 1:
        qids_all = np.arange(len(lab), dtype=np.int64)
        chunks = [
            qids_all[start : start + cfg.prediction_chunk_size].astype(int).tolist()
            for start in range(0, len(qids_all), cfg.prediction_chunk_size)
        ]
        cfg_payload = {"top_x": cfg.top_x, "rerank_pool_k": cfg.rerank_pool_k}
        lab_payload = _payload_from_lab(lab)
        target_payload = _payload_from_targets(targets)
        exclude_list = sorted(int(x) for x in (exclude_target_ids or []))
        pred_rows = []
        print(
            f"Parallel RefMet scoring{suffix}: n_jobs={cfg.n_jobs}, "
            f"chunks={len(chunks)}, chunk_size={cfg.prediction_chunk_size}"
        )
        with ProcessPoolExecutor(
            max_workers=cfg.n_jobs,
            initializer=_init_predict_worker,
            initargs=(
                lab_payload,
                target_payload,
                str(scores_path),
                str(indices_path),
                weights,
                cfg_payload,
                strict_no_query_surface,
                exclude_list,
            ),
        ) as pool:
            futures = [pool.submit(_predict_chunk_worker, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scoring RefMet candidates{suffix}"):
                chunk_pred_rows, chunk_wide_rows = future.result()
                pred_rows.extend(chunk_pred_rows)
                pd.DataFrame(chunk_wide_rows).to_csv(wide_path, index=False, mode="a", header=not wide_path.exists())
        predictions = pd.DataFrame(pred_rows).sort_values("query_i").reset_index(drop=True)
        predictions.to_csv(pred_path, index=False)
        return predictions

    pred_rows = []
    chunk_size = max(512, min(cfg.search_batch_size, 8192))
    qids_all = np.arange(len(lab), dtype=np.int64)
    for start in tqdm(range(0, len(qids_all), chunk_size), desc=f"Scoring RefMet candidates{suffix}"):
        qids = qids_all[start : start + chunk_size]
        feats, tids, tpos, invalid = make_feature_block(
            qids,
            lab,
            targets,
            scores,
            indices,
            cfg,
            strict_no_query_surface=strict_no_query_surface,
            exclude_target_ids=exclude_target_ids,
        )
        hybrid_scores = np.tensordot(feats, weight_vec, axes=([2], [0]))
        hybrid_scores = np.where(invalid, -np.inf, hybrid_scores)
        order = np.argsort(-hybrid_scores, axis=1)
        fallback_ranks = base_gold_ranks(qids, lab, targets, indices)
        wide_rows = []
        for row_i, qid in enumerate(qids):
            ordered = order[row_i]
            top_order = [j for j in ordered if np.isfinite(hybrid_scores[row_i, j])][: cfg.top_x]
            row = {
                "query_i": int(qid),
                "source_row_id": int(lab.at[qid, "source_row_id"]),
                "split": lab.at[qid, "split"],
                "metabolite_name": lab.at[qid, "metabolite_name"],
                "gold_refmet_name": lab.at[qid, "refmet_name"],
                "gold_target_id": int(gold_ids[qid]),
            }
            rank = np.inf
            for new_rank, j in enumerate(ordered, start=1):
                if not np.isfinite(hybrid_scores[row_i, j]):
                    continue
                if int(tids[row_i, j]) == int(gold_ids[qid]):
                    rank = float(new_rank)
                    break
            if not np.isfinite(rank) and fallback_ranks[row_i] > cfg.rerank_pool_k:
                rank = float(fallback_ranks[row_i])

            top1_j = top_order[0] if top_order else -1
            top2_j = top_order[1] if len(top_order) > 1 else -1
            if top1_j >= 0:
                top1_pos = int(tpos[row_i, top1_j])
                top1_score = float(hybrid_scores[row_i, top1_j])
                top2_score = float(hybrid_scores[row_i, top2_j]) if top2_j >= 0 else -np.inf
                pred_name = target_names[top1_pos]
                pred_norm = target_norm[top1_pos]
                pred_tid = int(tids[row_i, top1_j])
                pred_cid = target_cids[top1_pos]
            else:
                top1_score = np.nan
                top2_score = np.nan
                pred_name = ""
                pred_norm = ""
                pred_tid = -1
                pred_cid = ""
            pred_rows.append(
                {
                    **row,
                    "predicted_refmet_name": pred_name,
                    "predicted_refmet_normalized": pred_norm,
                    "predicted_target_id": pred_tid,
                    "predicted_refmet_cid": pred_cid,
                    "prediction_score": top1_score,
                    "second_score": top2_score,
                    "score_margin": top1_score - top2_score if np.isfinite(top2_score) else np.nan,
                    "gold_rank": rank,
                    "top1_exact_correct": pred_tid == int(gold_ids[qid]),
                    "top1_relation_to_query": surface_relation(lab.at[qid, "metabolite_name"], pred_name),
                }
            )
            for display_rank, j in enumerate(top_order, start=1):
                pos = int(tpos[row_i, j])
                row[f"top{display_rank}_refmet_name"] = target_names[pos]
                row[f"top{display_rank}_target_id"] = int(tids[row_i, j])
                row[f"top{display_rank}_score"] = float(hybrid_scores[row_i, j])
                row[f"top{display_rank}_is_gold"] = int(tids[row_i, j]) == int(gold_ids[qid])
                row[f"top{display_rank}_relation_to_query"] = surface_relation(lab.at[qid, "metabolite_name"], target_names[pos])
            wide_rows.append(row)
        pd.DataFrame(wide_rows).to_csv(wide_path, index=False, mode="a", header=not wide_path.exists())
    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(pred_path, index=False)
    return predictions


def embedding_only_predictions(
    cfg: Config,
    lab: pd.DataFrame,
    targets: pd.DataFrame,
    scores_path: Path,
    indices_path: Path,
) -> pd.DataFrame:
    pred_path = cfg.results_dir / "refmet_embedding_only_predictions.csv"
    if pred_path.exists() and not cfg.force_rescore:
        try:
            cached = pd.read_csv(pred_path)
            if len(cached) == len(lab):
                print(f"Using cached embedding-only predictions: {pred_path}")
                return cached
        except Exception as exc:
            print(f"Could not reuse cached embedding-only predictions; rebuilding. Reason: {exc!r}")

    scores = np.load(scores_path, mmap_mode="r")
    indices = np.load(indices_path, mmap_mode="r")
    target_names = targets["refmet_name"].astype(str).to_numpy()
    target_norm = targets["refmet_normalized"].astype(str).to_numpy()
    target_cids = targets["refmet_cid"].fillna("").astype(str).to_numpy()
    target_ids_all = targets["target_id"].to_numpy(dtype=np.int64)
    gold_ids = lab["gold_target_id"].fillna(-1).astype(int).to_numpy()
    pred_rows = []
    qids_all = np.arange(len(lab), dtype=np.int64)
    for start in tqdm(range(0, len(qids_all), cfg.search_batch_size), desc="Writing embedding-only predictions"):
        qids = qids_all[start : start + cfg.search_batch_size]
        fallback_ranks = base_gold_ranks(qids, lab, targets, indices)
        for row_i, qid in enumerate(qids):
            top1_pos = int(indices[qid, 0])
            top2_score = float(scores[qid, 1]) if scores.shape[1] > 1 else -np.inf
            if top1_pos >= 0:
                pred_name = target_names[top1_pos]
                pred_norm = target_norm[top1_pos]
                pred_tid = int(target_ids_all[top1_pos])
                pred_cid = target_cids[top1_pos]
                top1_score = float(scores[qid, 0])
            else:
                pred_name = ""
                pred_norm = ""
                pred_tid = -1
                pred_cid = ""
                top1_score = np.nan
            pred_rows.append(
                {
                    "query_i": int(qid),
                    "source_row_id": int(lab.at[qid, "source_row_id"]),
                    "split": lab.at[qid, "split"],
                    "metabolite_name": lab.at[qid, "metabolite_name"],
                    "gold_refmet_name": lab.at[qid, "refmet_name"],
                    "gold_target_id": int(gold_ids[qid]),
                    "predicted_refmet_name": pred_name,
                    "predicted_refmet_normalized": pred_norm,
                    "predicted_target_id": pred_tid,
                    "predicted_refmet_cid": pred_cid,
                    "prediction_score": top1_score,
                    "second_score": top2_score,
                    "score_margin": top1_score - top2_score if np.isfinite(top2_score) else np.nan,
                    "gold_rank": float(fallback_ranks[row_i]),
                    "top1_exact_correct": pred_tid == int(gold_ids[qid]),
                    "top1_relation_to_query": surface_relation(lab.at[qid, "metabolite_name"], pred_name),
                }
            )
    out = pd.DataFrame(pred_rows)
    out.to_csv(pred_path, index=False)
    return out


def calibrate_confidence_thresholds(predictions: pd.DataFrame, cfg: Config) -> dict:
    path = cfg.results_dir / "confidence_thresholds_calibrated.json"
    if path.exists() and not cfg.force_rescore:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            required = {
                "auto_accept_margin_threshold",
                "pubmed_fallback_margin_threshold",
                "retained_refmet_review_validation_hit10",
            }
            if required.issubset(cached):
                print(f"Using cached calibrated confidence thresholds: {path}")
                return cached
        except Exception as exc:
            print(f"Could not reuse cached confidence thresholds; recalibrating. Reason: {exc!r}")

    val = predictions[predictions["split"] == "validation"].copy()
    if val.empty:
        fallback = {
            "mode": "fallback_defaults_no_validation_rows",
            "auto_accept_margin_threshold": cfg.high_conf_margin,
            "pubmed_fallback_margin_threshold": cfg.low_conf_margin,
            "auto_accept_target_precision": cfg.auto_accept_target_precision,
            "fallback_target_hit10": cfg.fallback_target_hit10,
        }
        save_json(path, fallback)
        return fallback

    val["score_margin"] = pd.to_numeric(val["score_margin"], errors="coerce").fillna(-np.inf)
    val["prediction_score"] = pd.to_numeric(val["prediction_score"], errors="coerce").fillna(-np.inf)
    val["top1_exact_correct"] = val["top1_exact_correct"].astype(bool)
    val["gold_in_top10"] = pd.to_numeric(val["gold_rank"], errors="coerce").fillna(np.inf) <= 10
    surface_supported = val["top1_relation_to_query"].isin({"exact_normalized", "exact_compact", "same_tokens", "substring"})

    thresholds = np.unique(np.quantile(val["score_margin"].replace([-np.inf, np.inf], np.nan).dropna(), np.linspace(0, 1, 101)))
    if len(thresholds) == 0:
        thresholds = np.array([cfg.low_conf_margin, cfg.high_conf_margin], dtype=float)

    auto_rows = []
    best_auto = None
    for threshold in thresholds:
        accepted = val[
            (val["prediction_score"] >= cfg.min_top1_score)
            & ((val["score_margin"] >= threshold) | ((val["score_margin"] >= cfg.auto_accept_margin) & surface_supported))
        ]
        if len(accepted) == 0:
            continue
        precision = float(accepted["top1_exact_correct"].mean())
        coverage = float(len(accepted) / len(val))
        row = {
            "threshold": float(threshold),
            "accepted_n": int(len(accepted)),
            "precision": precision,
            "coverage": coverage,
        }
        auto_rows.append(row)
        if precision >= cfg.auto_accept_target_precision:
            if best_auto is None or coverage > best_auto["coverage"]:
                best_auto = row
    if best_auto is None:
        best_auto = max(auto_rows, key=lambda r: (r["precision"], r["coverage"])) if auto_rows else {
            "threshold": cfg.high_conf_margin,
            "accepted_n": 0,
            "precision": np.nan,
            "coverage": 0.0,
        }

    fallback_rows = []
    best_fallback = None
    for threshold in thresholds:
        retained = val[
            (val["prediction_score"] >= cfg.min_top1_score)
            & (val["score_margin"] >= threshold)
        ]
        fallback = val.drop(retained.index)
        retained_hit10 = float(retained["gold_in_top10"].mean()) if len(retained) else np.nan
        fallback_hit10 = float(fallback["gold_in_top10"].mean()) if len(fallback) else np.nan
        retained_coverage = float(len(retained) / len(val))
        fallback_coverage = float(len(fallback) / len(val))
        row = {
            "threshold": float(threshold),
            "retained_refmet_review_n": int(len(retained)),
            "retained_refmet_review_hit10": retained_hit10,
            "retained_refmet_review_coverage": retained_coverage,
            "fallback_n": int(len(fallback)),
            "fallback_hit10": fallback_hit10,
            "fallback_coverage": fallback_coverage,
        }
        fallback_rows.append(row)
        if len(retained) and retained_hit10 >= cfg.fallback_target_hit10:
            if best_fallback is None or retained_coverage > best_fallback["retained_refmet_review_coverage"]:
                best_fallback = row
    if best_fallback is None:
        # If the target is not attainable, keep the threshold with the best observed RefMet-review Hit@10.
        valid_rows = [r for r in fallback_rows if pd.notna(r["retained_refmet_review_hit10"])]
        best_fallback = max(
            valid_rows,
            key=lambda r: (r["retained_refmet_review_hit10"], r["retained_refmet_review_coverage"]),
        ) if valid_rows else {
            "threshold": cfg.low_conf_margin,
            "fallback_n": 0,
            "fallback_coverage": 0.0,
            "fallback_hit10": np.nan,
            "retained_refmet_review_n": 0,
            "retained_refmet_review_hit10": np.nan,
            "retained_refmet_review_coverage": 0.0,
        }

    calibrated = {
        "mode": "validation_calibrated",
        "auto_accept_target_precision": cfg.auto_accept_target_precision,
        "fallback_target_hit10": cfg.fallback_target_hit10,
        "auto_accept_margin_threshold": float(best_auto["threshold"]),
        "auto_accept_validation_precision": float(best_auto["precision"]) if pd.notna(best_auto["precision"]) else None,
        "auto_accept_validation_coverage": float(best_auto["coverage"]),
        "pubmed_fallback_margin_threshold": float(best_fallback["threshold"]),
        "pubmed_fallback_validation_hit10": float(best_fallback["fallback_hit10"]) if pd.notna(best_fallback["fallback_hit10"]) else None,
        "pubmed_fallback_validation_coverage": float(best_fallback["fallback_coverage"]),
        "retained_refmet_review_validation_hit10": float(best_fallback["retained_refmet_review_hit10"]) if pd.notna(best_fallback["retained_refmet_review_hit10"]) else None,
        "retained_refmet_review_validation_coverage": float(best_fallback["retained_refmet_review_coverage"]),
        "validation_rows": int(len(val)),
    }
    save_json(path, calibrated)
    pd.DataFrame(auto_rows).to_csv(cfg.results_dir / "auto_accept_threshold_sweep.csv", index=False)
    pd.DataFrame(fallback_rows).to_csv(cfg.results_dir / "fallback_threshold_sweep.csv", index=False)
    print("Calibrated confidence thresholds:")
    print(json.dumps(calibrated, indent=2))
    return calibrated


def assign_decisions(predictions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = predictions.copy()
    out["prediction_score"] = pd.to_numeric(out["prediction_score"], errors="coerce")
    out["score_margin"] = pd.to_numeric(out["score_margin"], errors="coerce")
    out["refmet_confidence_score"] = out["score_margin"].fillna(-np.inf)

    surface_supported = out["top1_relation_to_query"].isin({"exact_normalized", "exact_compact", "same_tokens", "substring"})
    mode = cfg.pubmed_fallback_query_mode.lower().strip()
    thresholds = calibrate_confidence_thresholds(out, cfg) if mode in {"calibrated", "threshold", "threshold_calibrated"} else None

    if thresholds is not None:
        high_conf = (
            (out["prediction_score"].fillna(-np.inf) >= cfg.min_top1_score)
            & (
                (out["score_margin"].fillna(-np.inf) >= thresholds["auto_accept_margin_threshold"])
                | ((out["score_margin"].fillna(-np.inf) >= cfg.auto_accept_margin) & surface_supported)
            )
        )
        low_conf = (
            (out["score_margin"].fillna(-np.inf) < thresholds["pubmed_fallback_margin_threshold"])
            | (out["prediction_score"].fillna(-np.inf) < cfg.min_top1_score)
        ) & ~high_conf
        out["confidence_tier"] = "medium_confidence_refmet_review"
        out.loc[high_conf, "confidence_tier"] = "high_confidence_refmet_auto_accept"
        out.loc[low_conf, "confidence_tier"] = "low_confidence_pubmed_fallback"
    elif mode in {"low_confidence", "low_conf"}:
        high_conf = (
            (out["prediction_score"].fillna(-np.inf) >= cfg.min_top1_score)
            & (
                (out["score_margin"].fillna(-np.inf) >= cfg.high_conf_margin)
                | ((out["score_margin"].fillna(-np.inf) >= cfg.auto_accept_margin) & surface_supported)
            )
        )
        out["confidence_tier"] = "medium_confidence_refmet_review"
        out.loc[high_conf, "confidence_tier"] = "high_confidence_refmet_auto_accept"
        low_conf = (
            (out["prediction_score"].fillna(-np.inf) < cfg.min_top1_score)
            | (out["score_margin"].fillna(-np.inf) <= cfg.low_conf_margin)
        ) & (out["confidence_tier"] != "high_confidence_refmet_auto_accept")
        out.loc[low_conf, "confidence_tier"] = "low_confidence_pubmed_fallback"
    elif mode in {"review_only", "uncertain", "manual_review"}:
        high_conf = (
            (out["prediction_score"].fillna(-np.inf) >= cfg.min_top1_score)
            & (
                (out["score_margin"].fillna(-np.inf) >= cfg.high_conf_margin)
                | ((out["score_margin"].fillna(-np.inf) >= cfg.auto_accept_margin) & surface_supported)
            )
        )
        out["confidence_tier"] = "medium_confidence_refmet_review"
        out.loc[high_conf, "confidence_tier"] = "high_confidence_refmet_auto_accept"
        low_conf = out["confidence_tier"] != "high_confidence_refmet_auto_accept"
        out.loc[low_conf, "confidence_tier"] = "low_confidence_pubmed_fallback"
    elif mode == "all":
        out["confidence_tier"] = "low_confidence_pubmed_fallback"
    else:
        raise ValueError(
            "Unsupported pubmed_fallback_query_mode="
            f"{cfg.pubmed_fallback_query_mode!r}. Use calibrated, all, review_only, or low_confidence."
        )

    out["decision"] = out["confidence_tier"]
    return out


def write_confidence_tier_summary(predictions: pd.DataFrame, cfg: Config, suffix: str = "") -> pd.DataFrame:
    out = predictions.groupby(["split", "confidence_tier"]).size().reset_index(name="n")
    out["fraction_within_split"] = out["n"] / out.groupby("split")["n"].transform("sum")
    out.to_csv(cfg.results_dir / f"confidence_tier_summary{suffix}.csv", index=False)
    return out


def write_confidence_tier_diagnostics(predictions: pd.DataFrame, cfg: Config, suffix: str = "") -> pd.DataFrame:
    rows = []
    for (split, tier), group in predictions.groupby(["split", "confidence_tier"], dropna=False):
        ranks = group["gold_rank"].to_numpy(dtype=np.float32)
        metrics = rank_metrics(ranks)
        rows.append(
            {
                "split": split,
                "confidence_tier": tier,
                "n": int(len(group)),
                "fraction_within_split": float(len(group) / len(predictions[predictions["split"] == split])),
                "MRR": metrics["MRR"],
                "Hit@1": metrics["Hit@1"],
                "Hit@10": metrics["Hit@10"],
                "Hit@256": metrics["Hit@256"],
                "median_rank": metrics["median_rank"],
                "mean_score_margin": float(group["score_margin"].mean()),
                "median_score_margin": float(group["score_margin"].median()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["split", "confidence_tier"])
    out.to_csv(cfg.results_dir / f"confidence_tier_diagnostics{suffix}.csv", index=False)
    return out


def metrics_for_predictions(pred: pd.DataFrame, label: str, split: str = "test", nonidentical: bool = False) -> dict:
    work = pred[pred["split"] == split].copy()
    if nonidentical:
        work = work[work["metabolite_name"].map(normalize_name) != work["gold_refmet_name"].map(normalize_name)]
    metrics = rank_metrics(work["gold_rank"].to_numpy(dtype=np.float32))
    metrics["evaluation"] = label
    metrics["split"] = split
    metrics["nonidentical_gold_surface_only"] = bool(nonidentical)
    return metrics


def bootstrap_ci(pred: pd.DataFrame, label: str, cfg: Config, split: str = "test", nonidentical: bool = False) -> pd.DataFrame:
    work = pred[pred["split"] == split].copy()
    if nonidentical:
        work = work[work["metabolite_name"].map(normalize_name) != work["gold_refmet_name"].map(normalize_name)]
    if work.empty:
        return pd.DataFrame()
    ranks = work["gold_rank"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for _ in tqdm(range(cfg.bootstrap_samples), desc=f"Bootstrap {label}"):
        sample = rng.choice(ranks, size=len(ranks), replace=True)
        rows.append(rank_metrics(sample))
    boot = pd.DataFrame(rows)
    out_rows = []
    for metric in ["MRR", "Hit@1", "Hit@5", "Hit@10", "Hit@50", "Hit@256", "median_rank"]:
        out_rows.append(
            {
                "evaluation": label,
                "split": split,
                "nonidentical_gold_surface_only": bool(nonidentical),
                "metric": metric,
                "mean": float(boot[metric].mean()),
                "ci_lower_95": float(boot[metric].quantile(0.025)),
                "ci_upper_95": float(boot[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(out_rows)


def confidence_curve(pred: pd.DataFrame, cfg: Config, split: str = "test") -> pd.DataFrame:
    work = pred[pred["split"] == split].copy()
    thresholds = np.unique(np.quantile(work["score_margin"].fillna(0.0), np.linspace(0.0, 1.0, 41))) if len(work) else []
    rows = []
    for threshold in thresholds:
        accepted = work[(work["score_margin"] >= threshold) & (work["prediction_score"] >= cfg.min_top1_score)]
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(len(accepted) / len(work)) if len(work) else np.nan,
                "accepted_n": int(len(accepted)),
                "top1_precision_among_accepted": float(accepted["top1_exact_correct"].mean()) if len(accepted) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_metric_plots(metrics: pd.DataFrame, route_summary: pd.DataFrame, cfg: Config) -> None:
    for metric in ["MRR", "Hit@1", "Hit@10", "Hit@256"]:
        if metric not in metrics.columns:
            continue
        plt.figure(figsize=(10, 5))
        plt.bar(metrics["evaluation"], metrics[metric])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.title(f"RefMet harmonization: {metric}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / f"{metric.replace('@', 'at')}_comparison.png", dpi=200)
        plt.close()
    test_routes = route_summary[route_summary["split"] == "test"].copy()
    if not test_routes.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(test_routes["curation_route"], test_routes["fraction_within_split"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Fraction of test queries")
        plt.title("Curation route distribution")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "curation_route_distribution.png", dpi=200)
        plt.close()


def prepare_pubmed_surfaces(cfg: Config, pubmed_parquet: Path) -> pd.DataFrame:
    out_csv = cfg.artifact_dir / "pubmed_fallback_surfaces.csv"
    con = duckdb.connect()
    limit_clause = f"LIMIT {int(cfg.pubmed_embed_row_limit)}" if cfg.pubmed_embed_row_limit is not None else ""
    if cfg.pubmed_deduplicate_global_surfaces:
        query = f"""
        WITH base AS (
          SELECT normalized_name,
                 any_value(display_name) AS display_name,
                 string_agg(DISTINCT cid, '|') AS candidate_cids,
                 min(name_len) AS name_len,
                 max(quality_score) AS quality_score
          FROM read_parquet('{sql_path(pubmed_parquet)}')
          GROUP BY normalized_name
        )
        SELECT row_number() OVER () - 1 AS pubmed_surface_id,
               display_name, normalized_name,
               regexp_replace(normalized_name, '[^a-z0-9]+', '', 'g') AS compact_name,
               candidate_cids, name_len, quality_score
        FROM base
        ORDER BY quality_score DESC, name_len ASC, normalized_name ASC
        {limit_clause}
        """
    else:
        query = f"""
        SELECT row_number() OVER () - 1 AS pubmed_surface_id,
               display_name, normalized_name, compact_name,
               cid AS candidate_cids, name_len, quality_score
        FROM read_parquet('{sql_path(pubmed_parquet)}')
        ORDER BY quality_score DESC, name_len ASC, normalized_name ASC
        {limit_clause}
        """
    surfaces = con.execute(query).df()
    con.close()
    surfaces.to_csv(out_csv, index=False)
    return surfaces


def search_pubmed_fallback(
    cfg: Config,
    lab: pd.DataFrame,
    pubmed_surfaces: pd.DataFrame,
    pubmed_emb_path: Path,
    query_emb_path: Path,
    query_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    scores_path = cfg.scratch_dir / "pubmed_fallback_neighbor_scores.npy"
    indices_path = cfg.scratch_dir / "pubmed_fallback_neighbor_indices.npy"
    drive_scores = cfg.artifact_dir / "pubmed_fallback_neighbor_scores.npy"
    drive_indices = cfg.artifact_dir / "pubmed_fallback_neighbor_indices.npy"
    if query_ids is None:
        query_ids = np.arange(len(lab), dtype=np.int64)
    else:
        query_ids = np.asarray(sorted(set(map(int, query_ids))), dtype=np.int64)

    restored_scores = restore_or_none(scores_path, drive_scores, cfg)
    restored_indices = restore_or_none(indices_path, drive_indices, cfg)
    cache_valid = False
    if restored_scores is not None and restored_indices is not None and not cfg.force_rescore:
        try:
            cached_indices = np.load(indices_path, mmap_mode="r")
            cached_query_ids = np.flatnonzero(cached_indices[:, 0] >= 0).astype(np.int64)
            cache_valid = np.array_equal(cached_query_ids, query_ids) and cached_indices.shape[1] >= cfg.k_retrieve_pubmed
            if cache_valid:
                print(f"Using cached PubMed/PubChem fallback neighbors for {len(query_ids):,} selected queries.")
            else:
                print(
                    "Cached PubMed/PubChem fallback neighbors do not match the current query selection; "
                    f"cached={len(cached_query_ids):,}, current={len(query_ids):,}. Rebuilding fallback search."
                )
        except Exception as exc:
            print(f"Could not validate cached PubMed/PubChem fallback neighbors; rebuilding. Reason: {exc!r}")

    if not cache_valid:
        drive_pubmed_embeddings = cfg.artifact_dir / "pubmed_fallback_surface_embeddings.npy"
        if Path(pubmed_emb_path).exists() and not artifact_or_parts_exists(drive_pubmed_embeddings):
            checkpoint_embedding_artifact(
                Path(pubmed_emb_path),
                drive_pubmed_embeddings,
                cfg,
                {
                    "label": "condensed PubMed/PubChem fallback surfaces",
                    "model_name_or_path": cfg.model_name_or_path,
                    "max_length": cfg.max_length,
                    "pooling": cfg.pooling,
                    "embed_dtype": cfg.embed_dtype,
                    "pubmed_max_names_per_cid": cfg.pubmed_max_names_per_cid,
                    "pubmed_embed_row_limit": cfg.pubmed_embed_row_limit,
                    "pubmed_deduplicate_global_surfaces": cfg.pubmed_deduplicate_global_surfaces,
                    "pubmed_surface_count": len(pubmed_surfaces),
                },
            )
        xb = np.load(pubmed_emb_path, mmap_mode="r")
        xq = np.load(query_emb_path, mmap_mode="r")

        if cfg.pubmed_search_mode == "stream_gpu" or (cfg.pubmed_search_mode == "auto" and xb.shape[0] > cfg.exact_gpu_max_vectors):
            s, i = search_embeddings_streaming_gpu(
                xb,
                xq,
                query_ids,
                cfg.k_retrieve_pubmed,
                cfg,
                "Searching PubMed/PubChem fallback",
            )
            index = resources = None
        else:
            if cfg.pubmed_search_mode == "flat_gpu":
                if not has_gpu_faiss():
                    raise RuntimeError("pubmed_search_mode='flat_gpu' requested, but GPU FAISS is not available.")
                index, resources = make_gpu_flat_index_for_chunk(np.asarray(xb, dtype="float32"))
            elif cfg.pubmed_search_mode == "flat_cpu":
                index = faiss.IndexFlatIP(xb.shape[1])
                index.add(np.asarray(xb, dtype="float32"))
                resources = None
            else:
                index, resources = make_exact_index(np.asarray(xb, dtype="float32"), cfg)
            if query_ids is not None and len(query_ids) != xq.shape[0]:
                sub_scores, sub_indices = search_embeddings(
                    index,
                    xq[query_ids],
                    cfg.k_retrieve_pubmed,
                    cfg.search_batch_size,
                    "Searching PubMed/PubChem fallback",
                )
                s = np.full((xq.shape[0], cfg.k_retrieve_pubmed), -np.inf, dtype=np.float32)
                i = np.full((xq.shape[0], cfg.k_retrieve_pubmed), -1, dtype=np.int64)
                s[query_ids] = sub_scores
                i[query_ids] = sub_indices
            else:
                s, i = search_embeddings(index, xq, cfg.k_retrieve_pubmed, cfg.search_batch_size, "Searching PubMed/PubChem fallback")
        np.save(scores_path, s.astype("float32"))
        np.save(indices_path, i.astype("int64"))
        checkpoint_artifact(scores_path, drive_scores, cfg)
        checkpoint_artifact(indices_path, drive_indices, cfg)
        del index, resources, s, i
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores = np.load(scores_path, mmap_mode="r")
    indices = np.load(indices_path, mmap_mode="r")
    top_rows = []
    wide_path = cfg.results_dir / f"pubmed_fallback_top{min(cfg.top_x, cfg.k_retrieve_pubmed)}_wide.csv"
    if wide_path.exists():
        wide_path.unlink()
    searched_qids = np.flatnonzero(indices[:, 0] >= 0)
    for start in tqdm(range(0, len(searched_qids), cfg.search_batch_size), desc="Writing PubMed/PubChem fallback outputs"):
        rows = []
        for qid in searched_qids[start : start + cfg.search_batch_size]:
            row = {"query_i": qid}
            for rank in range(min(cfg.top_x, cfg.k_retrieve_pubmed)):
                pos = int(indices[qid, rank])
                if pos < 0:
                    continue
                surf = pubmed_surfaces.iloc[pos]
                row[f"pubmed_top{rank + 1}_name"] = surf["display_name"]
                row[f"pubmed_top{rank + 1}_cids"] = surf["candidate_cids"]
                row[f"pubmed_top{rank + 1}_score"] = float(scores[qid, rank])
                row[f"pubmed_top{rank + 1}_relation_to_query"] = surface_relation(lab.at[qid, "metabolite_name"], surf["display_name"])
                if rank == 0:
                    top_rows.append(
                        {
                            "query_i": qid,
                            "top_pubmed_candidate_name": surf["display_name"],
                            "top_pubmed_candidate_cids": surf["candidate_cids"],
                            "top_pubmed_score": float(scores[qid, rank]),
                        }
                    )
            rows.append(row)
        pd.DataFrame(rows).to_csv(wide_path, index=False, mode="a", header=not wide_path.exists())
    return pd.DataFrame(top_rows)


def add_curation_routes(refmet_pred: pd.DataFrame, pubmed_top1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = refmet_pred.merge(pubmed_top1, on="query_i", how="left") if not pubmed_top1.empty else refmet_pred.copy()
    if "top_pubmed_score" not in out.columns:
        out["top_pubmed_score"] = np.nan
        out["top_pubmed_candidate_name"] = ""
        out["top_pubmed_candidate_cids"] = ""
    def route(row):
        tier = str(row.get("confidence_tier", row.get("decision", "")))
        if tier == "high_confidence_refmet_auto_accept":
            return "existing_refmet_auto_accept"
        if tier == "medium_confidence_refmet_review":
            return "medium_confidence_refmet_review"
        if tier == "low_confidence_pubmed_fallback" and pd.notna(row["top_pubmed_score"]) and float(row["top_pubmed_score"]) >= cfg.pubmed_fallback_min_score:
            return "manual_review_possible_new_refmet_or_pubchem_candidate"
        if tier == "low_confidence_pubmed_fallback":
            return "manual_review_low_confidence_no_pubmed_candidate"
        return "manual_review_uncategorized"
    out["curation_route"] = out.apply(route, axis=1)
    return out


def select_pubmed_fallback_query_ids(predictions: pd.DataFrame, cfg: Config) -> np.ndarray:
    mode = cfg.pubmed_fallback_query_mode.lower().strip()
    if mode == "all":
        query_ids = predictions["query_i"].to_numpy(dtype=np.int64)
    elif mode in {"calibrated", "threshold", "threshold_calibrated"}:
        query_ids = predictions.loc[predictions["confidence_tier"] == "low_confidence_pubmed_fallback", "query_i"].to_numpy(dtype=np.int64)
    elif mode in {"review_only", "uncertain", "manual_review"}:
        needs_review = predictions["confidence_tier"] != "high_confidence_refmet_auto_accept"
        query_ids = predictions.loc[needs_review, "query_i"].to_numpy(dtype=np.int64)
    elif mode in {"low_confidence", "low_conf"}:
        query_ids = predictions.loc[predictions["confidence_tier"] == "low_confidence_pubmed_fallback", "query_i"].to_numpy(dtype=np.int64)
    else:
        raise ValueError(
            "Unsupported pubmed_fallback_query_mode="
            f"{cfg.pubmed_fallback_query_mode!r}. Use calibrated, all, review_only, or low_confidence."
        )
    query_ids = np.asarray(sorted(set(map(int, query_ids))), dtype=np.int64)
    summary = {
        "pubmed_fallback_query_mode": cfg.pubmed_fallback_query_mode,
        "auto_accept_target_precision": cfg.auto_accept_target_precision,
        "fallback_target_hit10": cfg.fallback_target_hit10,
        "n_queries_total": int(len(predictions)),
        "n_queries_sent_to_pubmed_fallback": int(len(query_ids)),
        "fraction_sent_to_pubmed_fallback": float(len(query_ids) / len(predictions)) if len(predictions) else 0.0,
    }
    save_json(cfg.results_dir / "pubmed_fallback_query_selection_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return query_ids


def synthetic_open_set_audit(
    cfg: Config,
    lab: pd.DataFrame,
    targets: pd.DataFrame,
    scores_path: Path,
    indices_path: Path,
    weights: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.run_synthetic_open_set:
        return pd.DataFrame(), pd.DataFrame()
    eval_rows = lab[(lab["split"] == "test") & lab["gold_mapped_to_target"]].copy()
    if eval_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed)
    unique_targets = eval_rows["gold_target_id"].dropna().astype(int).unique()
    n_holdout = max(1, int(len(unique_targets) * cfg.open_set_holdout_target_frac))
    heldout_targets = set(rng.choice(unique_targets, size=n_holdout, replace=False).tolist())
    open_queries = eval_rows[eval_rows["gold_target_id"].astype(int).isin(heldout_targets)].copy()
    if cfg.open_set_max_queries and len(open_queries) > cfg.open_set_max_queries:
        open_queries = open_queries.sample(cfg.open_set_max_queries, random_state=cfg.random_seed)

    scores = np.load(scores_path, mmap_mode="r")
    indices = np.load(indices_path, mmap_mode="r")
    target_names = targets["refmet_name"].astype(str).to_numpy()
    target_ids_all = targets["target_id"].to_numpy(dtype=np.int64)
    weight_vec = np.array([weights[c] for c in FEATURE_COLUMNS], dtype=np.float32)
    tmp_rows = []
    qids_all = open_queries.index.to_numpy(dtype=np.int64)
    for start in tqdm(range(0, len(qids_all), cfg.search_batch_size), desc="Synthetic open-set audit"):
        qids = qids_all[start : start + cfg.search_batch_size]
        feats, tids, tpos, invalid = make_feature_block(
            qids,
            lab,
            targets,
            scores,
            indices,
            cfg,
            strict_no_query_surface=False,
            exclude_target_ids=heldout_targets,
        )
        hybrid_scores = np.tensordot(feats, weight_vec, axes=([2], [0]))
        hybrid_scores = np.where(invalid, -np.inf, hybrid_scores)
        order = np.argsort(-hybrid_scores, axis=1)
        for row_i, qid in enumerate(qids):
            valid_order = [j for j in order[row_i] if np.isfinite(hybrid_scores[row_i, j])]
            if valid_order:
                top1_j = valid_order[0]
                top2_j = valid_order[1] if len(valid_order) > 1 else -1
                top1_pos = int(tpos[row_i, top1_j])
                top1_score = float(hybrid_scores[row_i, top1_j])
                top2_score = float(hybrid_scores[row_i, top2_j]) if top2_j >= 0 else -np.inf
                top1_name = target_names[top1_pos]
                top1_tid = int(target_ids_all[top1_pos])
            else:
                top1_score = np.nan
                top2_score = np.nan
                top1_name = ""
                top1_tid = -1
            tmp_rows.append(
                {
                    "query_i": int(qid),
                    "metabolite_name": lab.at[qid, "metabolite_name"],
                    "heldout_gold_refmet_name": lab.at[qid, "refmet_name"],
                    "heldout_gold_target_id": int(lab.at[qid, "gold_target_id"]),
                    "predicted_existing_refmet_name": top1_name,
                    "predicted_existing_target_id": top1_tid,
                    "prediction_score": top1_score,
                    "second_score": top2_score,
                    "score_margin": top1_score - top2_score if np.isfinite(top2_score) else np.nan,
                    "top1_relation_to_query": surface_relation(lab.at[qid, "metabolite_name"], top1_name),
                }
            )
    tmp_pred = assign_decisions(pd.DataFrame(tmp_rows), cfg)
    rows = []
    thresholds = np.unique(np.quantile(tmp_pred["score_margin"].fillna(0.0), np.linspace(0, 1, 31))) if len(tmp_pred) else []
    for threshold in thresholds:
        accepts = (tmp_pred["score_margin"] >= threshold) & (tmp_pred["prediction_score"] >= cfg.min_top1_score)
        rows.append(
            {
                "threshold": float(threshold),
                "n_open_queries": int(len(tmp_pred)),
                "false_accept_rate": float(accepts.mean()) if len(tmp_pred) else np.nan,
                "abstain_or_review_rate": float((~accepts).mean()) if len(tmp_pred) else np.nan,
            }
        )
    return tmp_pred, pd.DataFrame(rows)


def write_review_samples(pred: pd.DataFrame, cfg: Config, suffix: str = "") -> None:
    out = pred.copy()
    out["case_group"] = np.select(
        [
            out["top1_exact_correct"],
            out["gold_rank"] <= 10,
            out["gold_rank"] <= 256,
        ],
        ["top1_correct", "gold_in_top10_not_rank1", "gold_in_top256_not_top10"],
        default="gold_not_retrieved_top256",
    )
    summary = out.groupby(["split", "case_group"]).size().reset_index(name="n")
    summary["fraction_within_split"] = summary["n"] / summary.groupby("split")["n"].transform("sum")
    summary.to_csv(cfg.results_dir / f"case_group_summary{suffix}.csv", index=False)
    samples = []
    for case_group, n in {
        "top1_correct": 50,
        "gold_in_top10_not_rank1": 75,
        "gold_in_top256_not_top10": 75,
        "gold_not_retrieved_top256": 75,
    }.items():
        group = out[(out["split"] == "test") & (out["case_group"] == case_group)]
        if not group.empty:
            samples.append(group.sample(min(n, len(group)), random_state=cfg.random_seed))
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(cfg.results_dir / f"human_review_case_samples{suffix}.csv", index=False)


def write_summary(cfg: Config, lab: pd.DataFrame, targets: pd.DataFrame, metrics: pd.DataFrame, pubmed_summary: dict) -> None:
    lines = [
        "# RefMet Harmonization Publication Pipeline Summary",
        "",
        "## Objective",
        "Evaluate a tuned SapBERT retrieval model plus a lightweight hybrid reranker for mapping laboratory metabolite names to existing RefMet names, while producing curation routes for cases that may require new RefMet entries or manual review.",
        "",
        "## Data",
        f"- Lab CSV: `{cfg.lab_csv}`",
        f"- Lab rows: {len(lab):,}",
        f"- Rows with metabolite_name: {int(lab['has_metabolite_name'].sum()):,}",
        f"- Rows with refmet_name: {int(lab['has_refmet_name'].sum()):,}",
        f"- Evaluation rows mapped to target vocabulary: {int((lab['split'] != 'unlabeled_or_unmapped').sum()):,}",
        f"- RefMet target vocabulary size: {len(targets):,}",
        f"- PubMed/PubChem condensed rows: {pubmed_summary.get('n_rows', 'not run')}",
        "",
        "## Methods",
        "- RefMet target names are embedded and searched exactly with FAISS to avoid compression loss in the main closed-set evaluation.",
        "- The large PubMed/PubChem synonym database is condensed by CID before embedding, keeping a deterministic top-N set of representative names per CID.",
        "- A second-stage reranker combines SapBERT score, rank prior, compact exact match, sequence similarity, character n-gram overlap, token overlap, lipid shorthand consistency, and stereochemistry penalties.",
        "- Final results include full and strict non-identical/no-query-surface views so exact surface-form matches do not overstate model behavior.",
        f"- Curation routing uses validation-calibrated confidence tiers: high-confidence RefMet auto-accept targeting {cfg.auto_accept_target_precision:.0%} validation precision, medium-confidence RefMet review, and PubMed/PubChem fallback for queries whose RefMet candidate set falls below the validation Hit@10 target of {cfg.fallback_target_hit10:.0%}.",
        "",
        "## Main Metrics",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"- **{row['evaluation']}**: MRR={row['MRR']:.4f}, "
            f"Hit@1={row['Hit@1']:.2%}, Hit@10={row['Hit@10']:.2%}, "
            f"Hit@256={row['Hit@256']:.2%}, n={int(row['n']):,}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "The closed-set results measure how well the pipeline maps a submitted metabolite name to an existing RefMet common name. The strict results should be treated as the most defensible publication estimate because they remove exact metabolite/refmet label matches and exclude returned candidates that reproduce the query surface. PubMed/PubChem fallback results should be interpreted as curation evidence rather than final labels.",
            "",
            "## Key Output Files",
            "- `evaluation_metrics.csv`",
            "- `bootstrap_confidence_intervals.csv`",
            "- `confidence_tier_summary.csv`",
            "- `confidence_tier_diagnostics.csv`",
            f"- `refmet_top{cfg.top_x}_candidates_wide.csv`",
            "- `refmet_predictions_with_decisions.csv`",
            "- `refmet_plus_pubmed_curation_routes.csv`",
            "- `human_review_case_samples.csv`",
            "- `human_review_case_samples_strict_no_query_surface.csv`",
            "",
            "## Next Steps",
            "1. Have RefMet/domain experts review routed examples, especially high-confidence wrong predictions and possible-new-entry candidates.",
            "2. Calibrate acceptance thresholds with expert review labels.",
            "3. Report both full and strict metrics in the paper.",
            "4. Treat PubMed/PubChem fallback as evidence for curation, not an automatic RefMet assignment.",
        ]
    )
    (cfg.results_dir / "publication_pipeline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir")
    parser.add_argument("--run-name", default="refmet_harmonization_publication_pipeline")
    parser.add_argument("--lab-csv")
    parser.add_argument("--refmet-csv")
    parser.add_argument("--pubmed-csv")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--refmet-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--pubmed-batch-size", type=int, default=8192)
    parser.add_argument("--top-x", type=int, default=10)
    parser.add_argument("--k-retrieve-refmet", type=int, default=256)
    parser.add_argument("--rerank-pool-k", type=int, default=128)
    parser.add_argument("--k-retrieve-pubmed", type=int, default=100)
    parser.add_argument("--search-batch-size", type=int, default=8192)
    parser.add_argument("--exact-gpu-max-vectors", type=int, default=2_500_000)
    parser.add_argument("--pubmed-search-mode", default="stream_gpu", choices=["auto", "stream_gpu", "flat_cpu", "flat_gpu"])
    parser.add_argument("--pubmed-gpu-chunk-vectors", type=int, default=6_000_000)
    parser.add_argument(
        "--pubmed-fallback-query-mode",
        default="calibrated",
        choices=["calibrated", "all", "review_only", "low_confidence"],
    )
    parser.add_argument("--auto-accept-target-precision", type=float, default=0.90)
    parser.add_argument("--fallback-target-hit10", type=float, default=0.80)
    parser.add_argument("--no-pubmed-fallback", action="store_true")
    parser.add_argument("--rebuild-pubmed-condensed", action="store_true")
    parser.add_argument("--pubmed-max-names-per-cid", type=int, default=20)
    parser.add_argument("--pubmed-embed-row-limit", type=int, default=2_000_000)
    parser.add_argument("--rerank-random-search-trials", type=int, default=240)
    parser.add_argument("--rerank-tuning-max-queries", type=int, default=30_000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--pipeline-stage", default="all", choices=["all", "gpu", "cpu"])
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--prediction-chunk-size", type=int, default=4096)
    parser.add_argument("--force-rescore", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    save_json(cfg.results_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()})
    print(f"Project: {cfg.project_dir}")
    print(f"Results: {cfg.results_dir}")
    print(f"Scratch: {cfg.scratch_dir}")
    print(f"Model: {cfg.model_name_or_path}")
    print(f"Pipeline stage: {cfg.pipeline_stage}")
    print(f"CPU scoring jobs: {cfg.n_jobs}")

    lab = load_lab_data(cfg.lab_csv)
    refmet_targets = load_refmet_targets(cfg, lab)
    pubmed_parquet, pubmed_summary = build_pubmed_condensed(cfg)
    refmet_targets = augment_refmet_cids_from_pubmed(cfg, refmet_targets, pubmed_parquet)

    norm_to_target_id = dict(zip(refmet_targets["refmet_normalized"], refmet_targets["target_id"]))
    lab["gold_target_id"] = lab["refmet_normalized"].map(norm_to_target_id)
    lab["gold_mapped_to_target"] = lab["gold_target_id"].notna()
    lab = split_lab(lab, cfg)

    dataset_summary = {
        "lab_csv": str(cfg.lab_csv),
        "refmet_csv": str(cfg.refmet_csv) if cfg.refmet_csv else None,
        "pubmed_csv": str(cfg.pubmed_csv) if cfg.pubmed_csv else None,
        "n_lab_rows": int(len(lab)),
        "n_rows_with_metabolite_name": int(lab["has_metabolite_name"].sum()),
        "n_rows_with_refmet_name": int(lab["has_refmet_name"].sum()),
        "n_eval_rows_mapped_to_refmet_target": int(lab["gold_mapped_to_target"].sum()),
        "n_refmet_targets": int(len(refmet_targets)),
        "n_exact_normalized_metabolite_equals_refmet": int(
            ((lab["metabolite_normalized"] == lab["refmet_normalized"]) & lab["gold_mapped_to_target"]).sum()
        ),
        "split_counts": lab["split"].value_counts().to_dict(),
    }
    save_json(cfg.results_dir / "dataset_summary.json", dataset_summary)
    lab.to_csv(cfg.artifact_dir / "lab_queries_with_splits.csv", index=False)
    refmet_targets.to_csv(cfg.artifact_dir / "refmet_target_dictionary.csv", index=False)
    print(json.dumps(dataset_summary, indent=2))

    pubmed_top1 = pd.DataFrame()
    if cfg.pipeline_stage == "cpu":
        refmet_emb = restore_or_none(
            cfg.scratch_dir / "refmet_label_embeddings.npy",
            cfg.artifact_dir / "refmet_label_embeddings.npy",
            cfg,
        )
        query_emb = restore_or_none(
            cfg.scratch_dir / "metabolite_query_embeddings.npy",
            cfg.artifact_dir / "metabolite_query_embeddings.npy",
            cfg,
        )
        if refmet_emb is None or query_emb is None:
            raise FileNotFoundError(
                "CPU stage requires cached RefMet and query embeddings. "
                "Run with PIPELINE_STAGE='gpu' or PIPELINE_STAGE='all' first."
            )
        pubmed_surfaces = prepare_pubmed_surfaces(cfg, pubmed_parquet) if cfg.run_pubmed_fallback and pubmed_parquet is not None else pd.DataFrame()
        pubmed_emb_path = cfg.scratch_dir / "pubmed_fallback_surface_embeddings.npy"
        pubmed_emb = (
            restore_or_none(pubmed_emb_path, cfg.artifact_dir / "pubmed_fallback_surface_embeddings.npy", cfg)
            or pubmed_emb_path
        ) if cfg.run_pubmed_fallback else None
    else:
        tokenizer, model, device = load_hf_model(cfg.model_name_or_path, cfg.device)
        refmet_emb = encode_texts_to_npy(
            refmet_targets["refmet_name"].astype(str).tolist(),
            cfg.scratch_dir / "refmet_label_embeddings.npy",
            cfg.artifact_dir / "refmet_label_embeddings.npy",
            tokenizer,
            model,
            device,
            cfg,
            cfg.refmet_batch_size,
            "RefMet target labels",
            cache_meta={"target_count": len(refmet_targets)},
        )
        query_emb = encode_texts_to_npy(
            lab["metabolite_name"].astype(str).tolist(),
            cfg.scratch_dir / "metabolite_query_embeddings.npy",
            cfg.artifact_dir / "metabolite_query_embeddings.npy",
            tokenizer,
            model,
            device,
            cfg,
            cfg.query_batch_size,
            "metabolite queries",
            cache_meta={"query_count": len(lab)},
        )

        if cfg.run_pubmed_fallback and pubmed_parquet is not None:
            pubmed_surfaces = prepare_pubmed_surfaces(cfg, pubmed_parquet)
            pubmed_emb = encode_texts_to_npy(
                pubmed_surfaces["display_name"].astype(str).tolist(),
                cfg.scratch_dir / "pubmed_fallback_surface_embeddings.npy",
                cfg.artifact_dir / "pubmed_fallback_surface_embeddings.npy",
                tokenizer,
                model,
                device,
                cfg,
                cfg.pubmed_batch_size,
                "condensed PubMed/PubChem fallback surfaces",
                cache_meta={
                    "pubmed_max_names_per_cid": cfg.pubmed_max_names_per_cid,
                    "pubmed_embed_row_limit": cfg.pubmed_embed_row_limit,
                    "pubmed_deduplicate_global_surfaces": cfg.pubmed_deduplicate_global_surfaces,
                    "pubmed_surface_count": len(pubmed_surfaces),
                    "pubmed_surfaces_csv_size": int((cfg.artifact_dir / "pubmed_fallback_surfaces.csv").stat().st_size)
                    if (cfg.artifact_dir / "pubmed_fallback_surfaces.csv").exists()
                    else None,
                },
            )
        else:
            pubmed_surfaces = pd.DataFrame()
            pubmed_emb = None

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cfg.pipeline_stage == "cpu":
        refmet_scores_path = restore_or_none(
            cfg.scratch_dir / "refmet_neighbor_scores.npy",
            cfg.artifact_dir / "refmet_neighbor_scores.npy",
            cfg,
        )
        refmet_indices_path = restore_or_none(
            cfg.scratch_dir / "refmet_neighbor_indices.npy",
            cfg.artifact_dir / "refmet_neighbor_indices.npy",
            cfg,
        )
        if refmet_scores_path is None or refmet_indices_path is None:
            raise FileNotFoundError(
                "CPU stage requires cached RefMet neighbor arrays. "
                "Run with PIPELINE_STAGE='gpu' or PIPELINE_STAGE='all' first."
            )
    else:
        refmet_scores_path, refmet_indices_path = run_refmet_search(cfg, refmet_emb, query_emb)
    if cfg.pipeline_stage == "gpu":
        if cfg.run_pubmed_fallback and pubmed_emb is not None and not pubmed_surfaces.empty and cfg.pubmed_fallback_query_mode == "all":
            all_query_ids = np.arange(len(lab), dtype=np.int64)
            pubmed_top1 = search_pubmed_fallback(cfg, lab, pubmed_surfaces, pubmed_emb, query_emb, query_ids=all_query_ids)
            pubmed_top1.to_csv(cfg.results_dir / "pubmed_fallback_top1_gpu_stage.csv", index=False)
        elif cfg.run_pubmed_fallback and cfg.pubmed_fallback_query_mode != "all":
            print(
                "GPU stage complete. PubMed fallback search was not run because "
                f"PUBMED_FALLBACK_QUERY_MODE={cfg.pubmed_fallback_query_mode!r} depends on CPU reranker decisions. "
                "For a true two-stage run with PubMed fallback cached before CPU scoring, set it to 'all'."
            )
        save_json(
            cfg.results_dir / "gpu_stage_manifest.json",
            {
                "refmet_embeddings": str(cfg.artifact_dir / "refmet_label_embeddings.npy"),
                "query_embeddings": str(cfg.artifact_dir / "metabolite_query_embeddings.npy"),
                "refmet_neighbor_scores": str(cfg.artifact_dir / "refmet_neighbor_scores.npy"),
                "refmet_neighbor_indices": str(cfg.artifact_dir / "refmet_neighbor_indices.npy"),
                "pubmed_embeddings": str(cfg.artifact_dir / "pubmed_fallback_surface_embeddings.npy") if pubmed_emb is not None else None,
                "pipeline_stage": cfg.pipeline_stage,
            },
        )
        print("\nGPU stage complete. Run again with PIPELINE_STAGE='cpu' for parallel reranker scoring and reporting.")
        return

    embedding_predictions = embedding_only_predictions(cfg, lab, refmet_targets, refmet_scores_path, refmet_indices_path)
    weights = tune_reranker(cfg, lab, refmet_targets, refmet_scores_path, refmet_indices_path)
    predictions = predict_refmet(cfg, lab, refmet_targets, refmet_scores_path, refmet_indices_path, weights)
    predictions = assign_decisions(predictions, cfg)
    write_confidence_tier_summary(predictions, cfg)
    tier_diagnostics = write_confidence_tier_diagnostics(predictions, cfg)
    predictions.to_csv(cfg.results_dir / "refmet_predictions_with_decisions.csv", index=False)

    strict_predictions = predict_refmet(
        cfg,
        lab,
        refmet_targets,
        refmet_scores_path,
        refmet_indices_path,
        weights,
        strict_no_query_surface=True,
        suffix="_strict_no_query_surface",
    )
    strict_predictions = assign_decisions(strict_predictions, cfg)
    write_confidence_tier_summary(strict_predictions, cfg, suffix="_strict_no_query_surface")
    write_confidence_tier_diagnostics(strict_predictions, cfg, suffix="_strict_no_query_surface")
    strict_predictions.to_csv(cfg.results_dir / "refmet_predictions_strict_no_query_surface_with_decisions.csv", index=False)

    if cfg.run_pubmed_fallback and pubmed_emb is not None and not pubmed_surfaces.empty:
        fallback_query_ids = select_pubmed_fallback_query_ids(predictions, cfg)
        cached_pubmed_neighbors = (
            (cfg.scratch_dir / "pubmed_fallback_neighbor_scores.npy").exists()
            or (cfg.artifact_dir / "pubmed_fallback_neighbor_scores.npy").exists()
            or (cfg.artifact_dir / "pubmed_fallback_neighbor_scores.npy.parts").exists()
        )
        if cfg.pipeline_stage == "cpu" and not cached_pubmed_neighbors:
            print(
                "CPU stage is skipping PubMed/PubChem fallback search because cached fallback neighbors were not found. "
                "Run GPU stage with PUBMED_FALLBACK_QUERY_MODE='all' to precompute fallback for a strict two-stage workflow."
            )
        else:
            pubmed_top1 = search_pubmed_fallback(cfg, lab, pubmed_surfaces, pubmed_emb, query_emb, query_ids=fallback_query_ids)

    routed = add_curation_routes(predictions, pubmed_top1, cfg)
    routed.to_csv(cfg.results_dir / "refmet_plus_pubmed_curation_routes.csv", index=False)
    route_summary = routed.groupby(["split", "curation_route"]).size().reset_index(name="n")
    route_summary["fraction_within_split"] = route_summary["n"] / route_summary.groupby("split")["n"].transform("sum")
    route_summary.to_csv(cfg.results_dir / "curation_route_summary.csv", index=False)

    metrics_rows = [
        metrics_for_predictions(embedding_predictions, "embedding_only_full_test", "test", nonidentical=False),
        metrics_for_predictions(embedding_predictions, "embedding_only_nonidentical_test", "test", nonidentical=True),
        metrics_for_predictions(predictions, "hybrid_rerank_full_test", "test", nonidentical=False),
        metrics_for_predictions(predictions, "hybrid_rerank_nonidentical_test", "test", nonidentical=True),
        metrics_for_predictions(strict_predictions, "hybrid_rerank_nonidentical_no_query_surface_test", "test", nonidentical=True),
    ]
    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[["evaluation", "split", "nonidentical_gold_surface_only", "n", "MRR", "Hit@1", "Hit@5", "Hit@10", "Hit@50", "Hit@256", "median_rank"]]
    metrics.to_csv(cfg.results_dir / "evaluation_metrics.csv", index=False)
    save_json(cfg.results_dir / "evaluation_metrics.json", metrics.to_dict(orient="records"))

    ci = pd.concat(
        [
            bootstrap_ci(embedding_predictions, "embedding_only_full_test", cfg, nonidentical=False),
            bootstrap_ci(embedding_predictions, "embedding_only_nonidentical_test", cfg, nonidentical=True),
            bootstrap_ci(predictions, "hybrid_rerank_full_test", cfg, nonidentical=False),
            bootstrap_ci(predictions, "hybrid_rerank_nonidentical_test", cfg, nonidentical=True),
            bootstrap_ci(strict_predictions, "hybrid_rerank_nonidentical_no_query_surface_test", cfg, nonidentical=True),
        ],
        ignore_index=True,
    )
    ci.to_csv(cfg.results_dir / "bootstrap_confidence_intervals.csv", index=False)

    conf = confidence_curve(predictions, cfg)
    conf.to_csv(cfg.results_dir / "confidence_threshold_curve.csv", index=False)
    if not conf.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(conf["coverage"], conf["top1_precision_among_accepted"], marker="o")
        plt.xlabel("Accepted coverage")
        plt.ylabel("Top-1 precision among accepted")
        plt.title("RefMet confidence routing curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "confidence_threshold_curve.png", dpi=200)
        plt.close()

    open_rows, open_metrics = synthetic_open_set_audit(cfg, lab, refmet_targets, refmet_scores_path, refmet_indices_path, weights)
    open_rows.to_csv(cfg.results_dir / "synthetic_open_set_holdout_rows.csv", index=False)
    open_metrics.to_csv(cfg.results_dir / "synthetic_open_set_metrics.csv", index=False)

    write_review_samples(predictions, cfg)
    write_review_samples(strict_predictions, cfg, suffix="_strict_no_query_surface")
    save_metric_plots(metrics, route_summary, cfg)
    write_summary(cfg, lab, refmet_targets, metrics, pubmed_summary)

    print("\nDone.")
    print(f"Results folder: {cfg.results_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
