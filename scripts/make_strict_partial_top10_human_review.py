"""Create strict top-10 human-review tables for the RefMet evaluation.

This script uses existing V4 outputs only. It does not run embeddings,
FAISS, or the reranker again.
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_dir() -> Path:
    candidates = [
        Path("/content/drive/MyDrive/ChemSynonymizer/chem_synonymizer"),
        Path("/content/drive/MyDrive/ChemSynonymizer"),
        Path("/content/chem_synonymizer"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    project_dir = next((p for p in candidates if (p / "results").exists()), None)
    if project_dir is None:
        raise FileNotFoundError("Could not find project directory. Pass --project-dir explicitly.")
    return project_dir


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype("string").str.lower().isin(["true", "1", "yes"])


def compact_text(s) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def token_set(s) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def surface_relation(
    a,
    b,
    sim_threshold: float,
    substring_min_len: int,
    substring_min_len_ratio: float,
) -> str:
    a = "" if pd.isna(a) else str(a).strip().lower()
    b = "" if pd.isna(b) else str(b).strip().lower()
    if not a or not b:
        return "missing_surface"
    if a == b:
        return "exact_normalized_equal"

    ac = compact_text(a)
    bc = compact_text(b)
    if ac and ac == bc:
        return "compact_equal"

    min_len = min(len(ac), len(bc))
    max_len = max(len(ac), len(bc))
    if (
        min_len >= substring_min_len
        and max_len
        and min_len / max_len >= substring_min_len_ratio
        and (ac in bc or bc in ac)
    ):
        return "compact_substring"

    seq = SequenceMatcher(None, a, b).ratio()
    if seq >= sim_threshold:
        return f"seq_ratio_ge_{sim_threshold:.2f}"

    at = token_set(a)
    bt = token_set(b)
    if at and bt:
        smaller = at if len(at) <= len(bt) else bt
        larger = bt if len(at) <= len(bt) else at
        if len(smaller) >= 2 and smaller.issubset(larger):
            return "token_subset"

    return "nontrivial_surface_difference"


def rank_metrics(df: pd.DataFrame, rank_col: str, ks=(1, 5, 10, 25, 50, 100, 256)) -> dict:
    ranks = pd.to_numeric(df[rank_col], errors="coerce")
    reciprocal = np.where(ranks.notna() & (ranks > 0), 1.0 / ranks, 0.0)
    out = {"MRR": float(np.mean(reciprocal)), "n": int(len(df))}
    for k in ks:
        out[f"Hit@{k}"] = float(np.mean(ranks.notna() & (ranks <= k)))
    return out


def sample_group(df: pd.DataFrame, label: str, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sample(n=min(n, len(df)), random_state=seed).copy()
    out["review_stratum"] = label
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--surface-sim-threshold", type=float, default=0.90)
    parser.add_argument("--substring-min-len", type=int, default=5)
    parser.add_argument("--substring-min-len-ratio", type=float, default=0.50)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve() if args.project_dir else find_project_dir().resolve()
    run_dir = args.run_dir or project_dir / "results" / "external_lab_refmet_model_comparison_reproducible"
    out_dir = args.out_dir or project_dir / "results" / "publication_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "conservative_no_query_surface_predictions.csv"
    top_hits_path = run_dir / "v4_final_test_top_hits.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing conservative predictions: {pred_path}")
    if not top_hits_path.exists():
        raise FileNotFoundError(f"Missing top-hit table: {top_hits_path}")

    print("PROJECT_DIR:", project_dir)
    print("RUN_DIR:", run_dir)
    print("OUT_DIR:", out_dir)
    print("Loading conservative predictions:", pred_path)
    pred = pd.read_csv(pred_path, low_memory=False)
    print("conservative predictions:", pred.shape)
    print("Loading top hits:", top_hits_path)
    top_hits = pd.read_csv(top_hits_path, low_memory=False)
    print("top hits:", top_hits.shape)

    rel = lambda a, b: surface_relation(
        a,
        b,
        sim_threshold=args.surface_sim_threshold,
        substring_min_len=args.substring_min_len,
        substring_min_len_ratio=args.substring_min_len_ratio,
    )

    pred["query_gold_surface_relation"] = [
        rel(a, b) for a, b in zip(pred["metabolite_normalized"], pred["refmet_normalized"])
    ]
    pred["query_gold_surface_trivial"] = pred["query_gold_surface_relation"].ne(
        "nontrivial_surface_difference"
    )
    review_base = pred[~pred["query_gold_surface_trivial"]].copy()
    review_query_ids = set(review_base["query_id"])

    base_cols = [
        "query_id",
        "metabolite_name",
        "refmet_name",
        "metabolite_normalized",
        "refmet_normalized",
        "gold_refmet_cids",
    ]
    base_cols = [c for c in base_cols if c in review_base.columns]

    th = top_hits[top_hits["query_id"].isin(review_query_ids)].copy()
    th["rank"] = pd.to_numeric(th["rank"], errors="coerce")
    th = th.sort_values(["query_id", "rank"]).reset_index(drop=True)
    th = th.merge(review_base[base_cols], on="query_id", how="left")
    th["candidate_query_surface_relation"] = [
        rel(a, b) for a, b in zip(th["metabolite_normalized"], th["candidate_refmet_normalized"])
    ]
    th["candidate_query_surface_trivial"] = th["candidate_query_surface_relation"].ne(
        "nontrivial_surface_difference"
    )
    th["exact_correct_bool"] = to_bool_series(th["exact_correct"])
    th["identity_correct_bool"] = to_bool_series(th["identity_correct"])

    strict_hits = th[~th["candidate_query_surface_trivial"]].copy()
    strict_hits["strict_candidate_rank"] = strict_hits.groupby("query_id").cumcount() + 1
    strict_top10 = strict_hits[strict_hits["strict_candidate_rank"] <= args.top_k].copy()

    exact_rank = strict_hits[strict_hits["exact_correct_bool"]].groupby("query_id")[
        "strict_candidate_rank"
    ].min()
    identity_rank = strict_hits[strict_hits["identity_correct_bool"]].groupby("query_id")[
        "strict_candidate_rank"
    ].min()
    review_base = review_base.merge(exact_rank.rename("strict_partial_exact_rank"), on="query_id", how="left")
    review_base = review_base.merge(
        identity_rank.rename("strict_partial_identity_rank"), on="query_id", how="left"
    )

    metrics = pd.DataFrame(
        [
            {
                "group": "strict_partial_no_gold_or_candidate_surface_match",
                "system": "strict_partial_exact",
                **rank_metrics(review_base, "strict_partial_exact_rank"),
            },
            {
                "group": "strict_partial_no_gold_or_candidate_surface_match",
                "system": "strict_partial_identity",
                **rank_metrics(review_base, "strict_partial_identity_rank"),
            },
        ]
    )

    summary = {
        "conservative_input_queries": int(len(pred)),
        "removed_query_gold_partial_surface_match": int(pred["query_gold_surface_trivial"].sum()),
        "kept_strict_partial_review_queries": int(len(review_base)),
        "kept_fraction_of_conservative_input": float(len(review_base) / len(pred)) if len(pred) else 0.0,
        "top_hit_rows_loaded": int(len(top_hits)),
        "candidate_rows_for_review_queries": int(len(th)),
        "removed_candidate_query_partial_surface_rows": int(th["candidate_query_surface_trivial"].sum()),
        "strict_top10_rows": int(len(strict_top10)),
        "queries_with_at_least_10_strict_candidates": int(
            (strict_top10.groupby("query_id").size() >= args.top_k).sum()
        ),
        "surface_similarity_threshold": args.surface_sim_threshold,
        "substring_min_len": args.substring_min_len,
        "substring_min_len_ratio": args.substring_min_len_ratio,
    }

    long_base_cols = [
        "query_id",
        "metabolite_name",
        "refmet_name",
        "gold_refmet_cids",
        "metabolite_normalized",
        "refmet_normalized",
        "query_gold_surface_relation",
        "strict_partial_exact_rank",
        "strict_partial_identity_rank",
    ]
    long_base_cols = [c for c in long_base_cols if c in review_base.columns]
    strict_top10 = strict_top10.merge(
        review_base[long_base_cols], on="query_id", how="left", suffixes=("", "_base")
    )
    for c in ["metabolite_name", "refmet_name", "gold_refmet_cids", "metabolite_normalized", "refmet_normalized"]:
        base_c = c + "_base"
        if base_c in strict_top10.columns:
            strict_top10[c] = strict_top10[base_c]

    review_long_cols = [
        "query_id",
        "metabolite_name",
        "refmet_name",
        "gold_refmet_cids",
        "query_gold_surface_relation",
        "strict_partial_exact_rank",
        "strict_partial_identity_rank",
        "strict_candidate_rank",
        "rank",
        "candidate_refmet_name",
        "candidate_refmet_normalized",
        "candidate_refmet_cids",
        "exact_correct",
        "identity_correct",
        "identity_reason",
        "embedding_score",
        "rerank_score",
        "candidate_query_surface_relation",
    ]
    review_long_cols = [c for c in review_long_cols if c in strict_top10.columns]
    review_long = strict_top10[review_long_cols].copy()
    review_long["human_same_compound"] = ""
    review_long["human_acceptable_refmet_alias"] = ""
    review_long["human_notes"] = ""

    wide_base_cols = [
        "query_id",
        "metabolite_name",
        "refmet_name",
        "gold_refmet_cids",
        "query_gold_surface_relation",
        "strict_partial_exact_rank",
        "strict_partial_identity_rank",
    ]
    wide_base_cols = [c for c in wide_base_cols if c in review_base.columns]
    wide = review_base[wide_base_cols].copy()
    for k in range(1, args.top_k + 1):
        kth = strict_top10[strict_top10["strict_candidate_rank"].eq(k)][
            [
                "query_id",
                "candidate_refmet_name",
                "candidate_refmet_cids",
                "exact_correct_bool",
                "identity_correct_bool",
                "identity_reason",
                "embedding_score",
                "rerank_score",
                "rank",
            ]
        ].copy()
        kth = kth.rename(
            columns={
                "candidate_refmet_name": f"top{k}_name",
                "candidate_refmet_cids": f"top{k}_cids",
                "exact_correct_bool": f"top{k}_exact_correct",
                "identity_correct_bool": f"top{k}_identity_correct",
                "identity_reason": f"top{k}_identity_reason",
                "embedding_score": f"top{k}_embedding_score",
                "rerank_score": f"top{k}_rerank_score",
                "rank": f"top{k}_original_rank",
            }
        )
        wide = wide.merge(kth, on="query_id", how="left")
    wide["human_selected_correct_rank"] = ""
    wide["human_any_top10_same_compound"] = ""
    wide["human_notes"] = ""

    exact_rank_num = pd.to_numeric(review_base["strict_partial_exact_rank"], errors="coerce")
    identity_rank_num = pd.to_numeric(review_base["strict_partial_identity_rank"], errors="coerce")
    sample_queries = pd.concat(
        [
            sample_group(review_base[exact_rank_num.eq(1)], "strict_exact_top1_success", 60, args.random_seed),
            sample_group(
                review_base[(~exact_rank_num.eq(1)) & identity_rank_num.eq(1)],
                "strict_identity_top1_success_not_exact",
                60,
                args.random_seed + 1,
            ),
            sample_group(
                review_base[exact_rank_num.between(2, args.top_k, inclusive="both")],
                "gold_in_top10_not_rank1",
                80,
                args.random_seed + 2,
            ),
            sample_group(
                review_base[exact_rank_num.isna() | (exact_rank_num > args.top_k)],
                "gold_not_in_strict_top10",
                100,
                args.random_seed + 3,
            ),
        ],
        ignore_index=True,
    )
    sample_wide = sample_queries[["query_id", "review_stratum"]].merge(wide, on="query_id", how="left")
    sample_long = sample_queries[["query_id", "review_stratum"]].merge(review_long, on="query_id", how="left")

    relation_counts = pred["query_gold_surface_relation"].value_counts(dropna=False).rename_axis(
        "query_gold_surface_relation"
    ).reset_index(name="n_queries")
    candidate_relation_counts = th["candidate_query_surface_relation"].value_counts(dropna=False).rename_axis(
        "candidate_query_surface_relation"
    ).reset_index(name="n_candidate_rows")

    pd.DataFrame([summary]).to_csv(out_dir / "strict_partial_top10_review_summary.csv", index=False)
    metrics.to_csv(out_dir / "strict_partial_top10_review_metrics.csv", index=False)
    relation_counts.to_csv(out_dir / "strict_partial_query_gold_surface_relation_counts.csv", index=False)
    candidate_relation_counts.to_csv(
        out_dir / "strict_partial_candidate_query_surface_relation_counts.csv", index=False
    )
    review_long.to_csv(out_dir / "strict_partial_top10_human_review_long.csv", index=False)
    wide.to_csv(out_dir / "strict_partial_top10_human_review_wide.csv", index=False)
    sample_wide.to_csv(out_dir / "strict_partial_top10_human_review_sample_wide.csv", index=False)
    sample_long.to_csv(out_dir / "strict_partial_top10_human_review_sample_long.csv", index=False)

    # The input top-hit table is already reranker-ranked. Save explicit reranker
    # filenames as well so downstream review materials do not confuse this with
    # embedding-only candidate order.
    pd.DataFrame([summary]).to_csv(out_dir / "strict_partial_top10_reranker_review_summary.csv", index=False)
    metrics.to_csv(out_dir / "strict_partial_top10_reranker_review_metrics.csv", index=False)
    review_long.to_csv(out_dir / "strict_partial_top10_human_review_reranker_long.csv", index=False)
    wide.to_csv(out_dir / "strict_partial_top10_human_review_reranker_wide.csv", index=False)
    sample_wide.to_csv(out_dir / "strict_partial_top10_human_review_reranker_sample_wide.csv", index=False)
    sample_long.to_csv(out_dir / "strict_partial_top10_human_review_reranker_sample_long.csv", index=False)

    print("Summary")
    print(pd.DataFrame([summary]).to_string(index=False))
    print()
    print("Metrics")
    print(metrics.to_string(index=False))
    print()
    print("Query/gold relation counts")
    print(relation_counts.to_string(index=False))
    print()
    print("Saved outputs in:", out_dir)


if __name__ == "__main__":
    main()
