from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from scipy import sparse

from chem_synonymizer.config import get_model_preset, load_config
from chem_synonymizer.corruptions import corrupt_many
from chem_synonymizer.data import dataframe_arrays, describe_data, load_synonym_csv
from chem_synonymizer.diagnostics import bucket_metrics_to_frame, make_examples, metric_row
from chem_synonymizer.embed import choose_device, encode_texts_to_memmap, load_hf_model
from chem_synonymizer.evaluate import evaluate_neighbors, evaluate_neighbors_by_bucket, pretty_metrics, search_faiss_neighbors
from chem_synonymizer.index import build_faiss_index_from_memmap
from chem_synonymizer.lexical import fit_char_tfidf, retrieve_tfidf_neighbors
from chem_synonymizer.plots import plot_bucket_metrics, plot_hitrate_by_model, plot_mrr_by_model, plot_robustness
from chem_synonymizer.split import sample_query_indices
from chem_synonymizer.utils import (
    array_sha1,
    config_hash,
    file_fingerprint,
    local_model_fingerprint,
    make_run_dir,
    resolve_model_name,
    save_json,
    set_seed,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval robustness under corrupted query names.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--baseline", choices=["lexical", "semantic"], default=None)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--model-name", default=None, help="HF model id or local checkpoint path. Overrides the model preset path.")
    parser.add_argument("--pooling", default=None, choices=["mean", "cls"])
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--corpus-prefix", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-frac", type=float, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--faiss-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--index-type", default=None, choices=["flat", "ivf_flat", "ivfpq"])
    parser.add_argument("--set", action="append", default=None, help="Override config values, e.g. --set robustness.corruption_levels=0,0.2,0.5")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def optional_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    return None if value.lower() in {"none", "null"} else value


def resolve_model(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model_name:
        key = args.model_key or slugify(Path(args.model_name).name)
        try:
            model = get_model_preset(cfg, key)
        except KeyError:
            model = {"key": key, "corpus_prefix": None, "query_prefix": None, "pooling": "cls"}
        model["model_name"] = args.model_name
    else:
        model = get_model_preset(cfg, args.model_key)

    if args.pooling:
        model["pooling"] = args.pooling
    if args.query_prefix is not None:
        model["query_prefix"] = optional_prefix(args.query_prefix)
    if args.corpus_prefix is not None:
        model["corpus_prefix"] = optional_prefix(args.corpus_prefix)
    model["model_name"] = resolve_model_name(ROOT, model["model_name"])
    return model


def level_name(level: float) -> str:
    return str(level).replace(".", "p")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.models_config, args.set)
    if args.csv_path:
        cfg["data"]["csv_path"] = args.csv_path
    if args.eval_frac is not None:
        cfg["eval"]["eval_frac"] = args.eval_frac
    if args.limit_rows is not None:
        cfg["data"]["limit_rows"] = args.limit_rows
    if args.device:
        cfg["compute"]["device"] = args.device
    if args.faiss_device:
        cfg["semantic"]["faiss_device"] = args.faiss_device
    if args.index_type:
        cfg["semantic"]["index_type"] = args.index_type

    baseline = args.baseline or cfg.get("robustness", {}).get("baseline", "lexical")
    model = resolve_model(cfg, args) if baseline == "semantic" else None
    if model:
        cfg["model"]["key"] = model["key"]

    label = model["key"] if model else "char_tfidf_edit"
    run_name = args.run_name or cfg.get("project", {}).get("run_name") or f"robustness_{baseline}_{label}"
    results_dir = resolve_path(cfg["project"]["results_dir"])
    run_dir = make_run_dir(results_dir, run_name)
    artifacts_dir = run_dir / "artifacts"

    seed = int(cfg["eval"]["seed"])
    set_seed(seed)
    csv_path = resolve_path(cfg["data"]["csv_path"])
    device = choose_device(cfg.get("compute", {}).get("device", "auto"))
    faiss_device = cfg.get("semantic", {}).get("faiss_device", "cpu")

    df = load_synonym_csv(
        csv_path=csv_path,
        cid_col=cfg["data"]["cid_col"],
        name_col=cfg["data"]["name_col"],
        normalize_names=cfg["data"]["normalize_names"],
        filter_empty_names=cfg["data"]["filter_empty_names"],
        remove_singletons=cfg["data"]["remove_singletons"],
        limit_rows=cfg["data"].get("limit_rows"),
    )
    arrays = dataframe_arrays(df)
    names = arrays["names"]
    ids = arrays["ids"]
    cid_size_arr = arrays["cid_size_arr"]
    q_idx = sample_query_indices(len(df), cfg["eval"]["eval_frac"], seed)
    clean_query_names = [names[int(i)] for i in q_idx]
    k_values = [int(k) for k in cfg["eval"]["k_values"]]
    k_retrieve = min(len(df), max(k_values) + 1)
    levels = [float(x) for x in cfg["robustness"]["corruption_levels"]]
    operations = list(cfg["robustness"]["operations"])
    csv_fingerprint = file_fingerprint(csv_path)
    embedding_cache_dir = None
    index_cache_dir = None
    data_key = None
    embedding_key = None
    index_key = None
    if baseline == "semantic":
        data_cache_cfg = {
            "csv": csv_fingerprint,
            "cid_col": cfg["data"]["cid_col"],
            "name_col": cfg["data"]["name_col"],
            "normalize_names": cfg["data"]["normalize_names"],
            "filter_empty_names": cfg["data"]["filter_empty_names"],
            "remove_singletons": cfg["data"]["remove_singletons"],
            "limit_rows": cfg["data"].get("limit_rows"),
        }
        embedding_cache_cfg = {
            "data_key": config_hash(data_cache_cfg),
            **model,
            "max_length": cfg["model"]["max_length"],
            "dtype": cfg["model"]["dtype"],
            "l2_normalize": cfg["model"]["l2_normalize"],
        }
        model_source = local_model_fingerprint(model["model_name"])
        if model_source is not None:
            embedding_cache_cfg["model_source"] = model_source
        index_cache_cfg = {
            "embedding_key": config_hash(embedding_cache_cfg),
            "index_type": cfg["semantic"]["index_type"],
            "nlist": cfg["semantic"]["nlist"],
            "train_size": cfg["semantic"]["train_size"],
            "m_pq": cfg["semantic"]["m_pq"],
            "nbits": cfg["semantic"]["nbits"],
            "seed": seed,
        }
        data_key = config_hash(data_cache_cfg)
        embedding_key = config_hash(embedding_cache_cfg)
        index_key = config_hash(index_cache_cfg)
        cache_root = cfg.get("project", {}).get("cache_dir")
        if cache_root:
            embedding_cache_dir = resolve_path(cache_root) / "semantic" / f"data_{data_key}" / f"{slugify(model['key'])}_{embedding_key}"
        else:
            embedding_cache_dir = artifacts_dir
        index_cache_dir = embedding_cache_dir / "indexes" / f"index_{index_key}"
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        save_json(data_cache_cfg, embedding_cache_dir.parent / "data_config.json")
        save_json(embedding_cache_cfg, embedding_cache_dir / "embedding_config.json")
        save_json(index_cache_cfg, index_cache_dir / "index_config.json")

    cfg["runtime"] = {
        "baseline": baseline,
        "run_dir": str(run_dir),
        "embedding_cache_dir": str(embedding_cache_dir) if embedding_cache_dir else None,
        "index_cache_dir": str(index_cache_dir) if index_cache_dir else None,
        "data_key": data_key,
        "embedding_key": embedding_key,
        "index_key": index_key,
        "csv": csv_fingerprint,
        "data": describe_data(df),
        "n_queries": int(len(q_idx)),
        "model": model,
        "device": device if baseline == "semantic" else None,
        "faiss_device": faiss_device if baseline == "semantic" else None,
    }
    save_json(cfg, run_dir / "config.json")
    save_json(
        {
            "embedding_cache_dir": str(embedding_cache_dir) if embedding_cache_dir else None,
            "index_cache_dir": str(index_cache_dir) if index_cache_dir else None,
            "data_key": data_key,
            "embedding_key": embedding_key,
            "index_key": index_key,
        },
        run_dir / "cache_info.json",
    )
    df.to_csv(artifacts_dir / "prepared_data.csv", index=False)
    pd.Series(q_idx).to_csv(artifacts_dir / "query_indices.csv", index=False, header=["row_index"])

    print(f"Run directory: {run_dir}")
    print(f"Rows after filtering: {len(df):,}; queries: {len(q_idx):,}")

    vectorizer = matrix = tokenizer = hf_model = index = None
    emb_corpus = None
    semantic_queries = {}
    if baseline == "lexical":
        vectorizer, matrix = fit_char_tfidf(names, cfg["lexical"])
        joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
        sparse.save_npz(artifacts_dir / "corpus_tfidf.npz", matrix)
    else:
        print(f"Loading model: {model['model_name']} on {device}")
        print(f"FAISS device: {faiss_device}")
        tokenizer, hf_model = load_hf_model(model["model_name"], device)
        emb_corpus = encode_texts_to_memmap(
            texts=names,
            out_path=embedding_cache_dir / "corpus_embeddings.npy",
            tokenizer=tokenizer,
            model=hf_model,
            model_name=model["model_name"],
            prefix=model.get("corpus_prefix"),
            pooling=model.get("pooling", "mean"),
            batch_size=cfg["model"]["batch_size"],
            max_length=cfg["model"]["max_length"],
            device=device,
            normalize=cfg["model"]["l2_normalize"],
            out_dtype=cfg["model"]["dtype"],
            run_key=embedding_key,
        )

        for i, level in enumerate(levels):
            query_names = corrupt_many(clean_query_names, level=level, seed=seed + i + 1000, operations=operations)
            pd.DataFrame({"row_index": q_idx, "clean_name": clean_query_names, "query_name": query_names}).to_csv(
                artifacts_dir / f"corrupted_queries_level_{level_name(level)}.csv",
                index=False,
            )
            query_cache_cfg = {
                "embedding_key": embedding_key,
                "seed": seed,
                "eval_frac": cfg["eval"]["eval_frac"],
                "q_idx_sha1": array_sha1(q_idx),
            }
            if level == 0.0:
                query_prefix = "split"
            else:
                query_cache_cfg.update({"corruption_level": level, "operations": operations})
                query_prefix = "robust"
            query_key = config_hash(query_cache_cfg)
            query_cache_dir = embedding_cache_dir / "queries" / f"{query_prefix}_{query_key}"
            query_cache_dir.mkdir(parents=True, exist_ok=True)
            save_json(query_cache_cfg, query_cache_dir / "query_config.json")
            emb_query = encode_texts_to_memmap(
                texts=query_names,
                out_path=query_cache_dir / "query_embeddings.npy",
                tokenizer=tokenizer,
                model=hf_model,
                model_name=model["model_name"],
                prefix=model.get("query_prefix"),
                pooling=model.get("pooling", "mean"),
                batch_size=cfg["model"]["query_batch_size"],
                max_length=cfg["model"]["max_length"],
                device=device,
                normalize=cfg["model"]["l2_normalize"],
                out_dtype=cfg["model"]["dtype"],
                run_key=query_key,
            )
            semantic_queries[level] = (query_names, emb_query)

        if device.startswith("cuda"):
            print("Releasing transformer model GPU memory before FAISS...")
            del hf_model
            del tokenizer
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        index, _ = build_faiss_index_from_memmap(
            xb_memmap=emb_corpus,
            index_path=index_cache_dir / f"faiss_{cfg['semantic']['index_type']}.index",
            index_type=cfg["semantic"]["index_type"],
            nlist=cfg["semantic"]["nlist"],
            nprobe=cfg["semantic"]["nprobe"],
            train_size=cfg["semantic"]["train_size"],
            add_batch_size=cfg["semantic"]["add_batch_size"],
            m_pq=cfg["semantic"]["m_pq"],
            nbits=cfg["semantic"]["nbits"],
            seed=seed,
            device=faiss_device,
            use_float16=cfg["semantic"]["use_float16_faiss"],
            run_key=index_key,
        )

    rows = []
    clean_metrics = None
    clean_bucket_metrics = None
    last_neighbors = None
    last_queries = None
    t0 = time.perf_counter()

    for i, level in enumerate(levels):
        if baseline == "lexical":
            query_names = corrupt_many(clean_query_names, level=level, seed=seed + i + 1000, operations=operations)
            pd.DataFrame({"row_index": q_idx, "clean_name": clean_query_names, "query_name": query_names}).to_csv(
                artifacts_dir / f"corrupted_queries_level_{level_name(level)}.csv",
                index=False,
            )
            neighbor_indices = retrieve_tfidf_neighbors(
                names=names,
                vectorizer=vectorizer,
                matrix=matrix,
                query_texts=query_names,
                k_retrieve=k_retrieve,
                batch_size=cfg["lexical"]["batch_size"],
                edit_distance_rerank=cfg["lexical"]["edit_distance_rerank"],
                candidate_multiplier=cfg["lexical"]["candidate_multiplier"],
                tfidf_weight=cfg["lexical"]["tfidf_weight"],
                n_jobs=cfg["lexical"]["n_jobs"],
            )
        else:
            query_names, emb_query = semantic_queries[level]
            neighbor_indices, _ = search_faiss_neighbors(
                index=index,
                query_embeddings=emb_query,
                k_retrieve=k_retrieve,
                batch_size=cfg["semantic"]["search_batch_size"],
            )

        metrics = evaluate_neighbors(q_idx, neighbor_indices, ids, cid_size_arr, k_values)
        rows.append(metric_row(metrics, baseline, label, run_dir.name, {"corruption_level": level}))
        print(f"\nCorruption level {level}")
        print(pretty_metrics(metrics, k_values))

        if clean_metrics is None or level == 0.0:
            clean_metrics = metrics
            clean_bucket_metrics = evaluate_neighbors_by_bucket(q_idx, neighbor_indices, ids, cid_size_arr, k_values)
        last_neighbors = neighbor_indices
        last_queries = query_names

    robustness_df = pd.DataFrame(rows)
    robustness_df.to_csv(run_dir / "robustness_metrics.csv", index=False)
    bucket_df = bucket_metrics_to_frame(clean_bucket_metrics, baseline, label, run_dir.name)
    bucket_df.to_csv(run_dir / "bucket_metrics.csv", index=False)
    examples = make_examples(
        q_idx=q_idx,
        neighbor_indices=last_neighbors,
        names=names,
        ids=ids,
        n_examples=cfg["eval"]["n_examples"],
        k=max(k_values),
        seed=seed,
        query_names=last_queries,
    )
    examples.to_csv(run_dir / "examples.csv", index=False)

    payload = {
        "run_name": run_dir.name,
        "baseline": baseline,
        "model_key": label,
        "model_name": model["model_name"] if model else "char_tfidf_edit",
        "metrics": clean_metrics,
        "elapsed_sec": time.perf_counter() - t0,
        "robustness": rows,
    }
    save_json(payload, run_dir / "metrics.json")

    summary_row = pd.DataFrame([metric_row(clean_metrics, baseline, label, run_dir.name)])
    plot_mrr_by_model(summary_row, run_dir / "plots" / "mrr_by_model.png")
    plot_hitrate_by_model(summary_row, run_dir / "plots" / "hitrate_by_model.png")
    plot_bucket_metrics(bucket_df, run_dir / "plots" / "bucket_mrr.png")
    plot_robustness(robustness_df, run_dir / "plots" / "robustness_mrr.png")

    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
