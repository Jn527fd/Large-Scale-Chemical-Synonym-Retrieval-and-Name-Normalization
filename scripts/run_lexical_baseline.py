from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from scipy import sparse

from chem_synonymizer.config import load_config
from chem_synonymizer.data import dataframe_arrays, describe_data, load_synonym_csv
from chem_synonymizer.diagnostics import bucket_metrics_to_frame, make_examples, metric_row
from chem_synonymizer.evaluate import evaluate_neighbors, evaluate_neighbors_by_bucket, pretty_metrics
from chem_synonymizer.lexical import fit_char_tfidf, retrieve_tfidf_neighbors
from chem_synonymizer.plots import plot_bucket_metrics, plot_hitrate_by_model, plot_mrr_by_model
from chem_synonymizer.split import sample_query_indices
from chem_synonymizer.utils import file_fingerprint, make_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lexical TF-IDF retrieval baseline.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-frac", type=float, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--set", action="append", default=None, help="Override config values, e.g. --set lexical.ngram_min=2")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.models_config, args.set)
    if args.csv_path:
        cfg["data"]["csv_path"] = args.csv_path
    if args.eval_frac is not None:
        cfg["eval"]["eval_frac"] = args.eval_frac
    if args.limit_rows is not None:
        cfg["data"]["limit_rows"] = args.limit_rows

    run_name = args.run_name or cfg.get("project", {}).get("run_name") or "lexical"
    results_dir = resolve_path(cfg["project"]["results_dir"])
    run_dir = make_run_dir(results_dir, run_name)
    artifacts_dir = run_dir / "artifacts"

    seed = int(cfg["eval"]["seed"])
    set_seed(seed)
    csv_path = resolve_path(cfg["data"]["csv_path"])

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
    k_values = [int(k) for k in cfg["eval"]["k_values"]]
    k_retrieve = min(len(df), max(k_values) + 1)

    cfg["runtime"] = {
        "baseline": "lexical",
        "run_dir": str(run_dir),
        "csv": file_fingerprint(csv_path),
        "data": describe_data(df),
        "n_queries": int(len(q_idx)),
    }
    save_json(cfg, run_dir / "config.json")
    df.to_csv(artifacts_dir / "prepared_data.csv", index=False)
    pd.Series(q_idx).to_csv(artifacts_dir / "query_indices.csv", index=False, header=["row_index"])

    print(f"Run directory: {run_dir}")
    print(f"Rows after filtering: {len(df):,}; queries: {len(q_idx):,}")
    t0 = time.perf_counter()
    vectorizer, matrix = fit_char_tfidf(names, cfg["lexical"])
    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
    sparse.save_npz(artifacts_dir / "corpus_tfidf.npz", matrix)

    neighbor_indices = retrieve_tfidf_neighbors(
        names=names,
        vectorizer=vectorizer,
        matrix=matrix,
        query_indices=q_idx,
        k_retrieve=k_retrieve,
        batch_size=cfg["lexical"]["batch_size"],
        edit_distance_rerank=cfg["lexical"]["edit_distance_rerank"],
        candidate_multiplier=cfg["lexical"]["candidate_multiplier"],
        tfidf_weight=cfg["lexical"]["tfidf_weight"],
        n_jobs=cfg["lexical"]["n_jobs"],
    )
    metrics = evaluate_neighbors(q_idx, neighbor_indices, ids, cid_size_arr, k_values)
    bucket_metrics = evaluate_neighbors_by_bucket(q_idx, neighbor_indices, ids, cid_size_arr, k_values)
    elapsed = time.perf_counter() - t0

    payload = {
        "run_name": run_dir.name,
        "baseline": "lexical",
        "model_key": "char_tfidf_edit",
        "metrics": metrics,
        "elapsed_sec": elapsed,
    }
    save_json(payload, run_dir / "metrics.json")
    bucket_df = bucket_metrics_to_frame(bucket_metrics, "lexical", "char_tfidf_edit", run_dir.name)
    bucket_df.to_csv(run_dir / "bucket_metrics.csv", index=False)
    examples = make_examples(
        q_idx=q_idx,
        neighbor_indices=neighbor_indices,
        names=names,
        ids=ids,
        n_examples=cfg["eval"]["n_examples"],
        k=max(k_values),
        seed=seed,
    )
    examples.to_csv(run_dir / "examples.csv", index=False)

    summary_row = pd.DataFrame([metric_row(metrics, "lexical", "char_tfidf_edit", run_dir.name)])
    plot_mrr_by_model(summary_row, run_dir / "plots" / "mrr_by_model.png")
    plot_hitrate_by_model(summary_row, run_dir / "plots" / "hitrate_by_model.png")
    plot_bucket_metrics(bucket_df, run_dir / "plots" / "bucket_mrr.png")

    print(pretty_metrics(metrics, k_values))
    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
