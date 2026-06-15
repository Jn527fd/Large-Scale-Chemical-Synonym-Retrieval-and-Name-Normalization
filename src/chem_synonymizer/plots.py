from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_metric_runs(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in Path(results_dir).glob("*/metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        if "MRR" not in metrics:
            continue
        label = payload.get("model_key") or payload.get("baseline") or metrics_path.parent.name
        rows.append(
            {
                "run_dir": str(metrics_path.parent),
                "run_name": payload.get("run_name", metrics_path.parent.name),
                "baseline": payload.get("baseline", ""),
                "model_key": label,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def load_bucket_metrics(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(results_dir).glob("*/bucket_metrics.csv"):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_robustness_metrics(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(results_dir).glob("*/robustness_metrics.csv"):
        try:
            frame = pd.read_csv(path)
            frame["run_dir"] = str(path.parent)
            frames.append(frame)
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_mrr_by_model(metrics: pd.DataFrame, out_path: str | Path) -> Path | None:
    if metrics.empty or "MRR" not in metrics:
        return None
    out_path = Path(out_path)
    frame = metrics.sort_values("MRR", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["model_key"], frame["MRR"], color="#2f6f73")
    ax.set_ylabel("MRR")
    ax.set_xlabel("Model")
    ax.set_ylim(0, min(1.0, max(frame["MRR"].max() * 1.15, 0.05)))
    ax.set_title("MRR by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_hitrate_by_model(metrics: pd.DataFrame, out_path: str | Path) -> Path | None:
    hit_cols = [col for col in ["HitRate@1", "HitRate@5", "HitRate@10"] if col in metrics.columns]
    if metrics.empty or not hit_cols:
        return None
    out_path = Path(out_path)
    frame = metrics.sort_values("model_key")
    x = range(len(frame))
    width = 0.8 / len(hit_cols)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#375a7f", "#d08c3c", "#4f7f4f"]
    for i, col in enumerate(hit_cols):
        offsets = [v - 0.4 + (i + 0.5) * width for v in x]
        ax.bar(offsets, frame[col], width=width, label=col, color=colors[i % len(colors)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["model_key"], rotation=30, ha="right")
    ax.set_ylabel("HitRate")
    ax.set_ylim(0, 1)
    ax.set_title("HitRate@k by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_bucket_metrics(bucket_metrics: pd.DataFrame, out_path: str | Path, metric: str = "MRR") -> Path | None:
    if bucket_metrics.empty or metric not in bucket_metrics.columns:
        return None
    out_path = Path(out_path)
    frame = bucket_metrics.copy()
    frame["label"] = frame["model_key"].fillna(frame["baseline"])
    frame = frame.sort_values(["label", "bucket_order"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, group in frame.groupby("label"):
        ax.plot(group["bucket"], group[metric], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Synonym set size bucket")
    ax.set_ylabel(metric)
    ax.set_ylim(0, min(1.0, max(frame[metric].max() * 1.15, 0.05)))
    ax.set_title(f"{metric} by synonym-set-size bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_robustness(robustness: pd.DataFrame, out_path: str | Path, metric: str = "MRR") -> Path | None:
    if robustness.empty or metric not in robustness.columns:
        return None
    out_path = Path(out_path)
    frame = robustness.copy()
    frame["label"] = frame["model_key"].fillna(frame["baseline"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, group in frame.groupby("label"):
        group = group.sort_values("corruption_level")
        ax.plot(group["corruption_level"], group[metric], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Corruption level")
    ax.set_ylabel(metric)
    ax.set_ylim(0, min(1.0, max(frame[metric].max() * 1.15, 0.05)))
    ax.set_title(f"Robustness degradation ({metric})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def make_all_plots(results_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "summary_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    metrics = load_metric_runs(results_dir)
    buckets = load_bucket_metrics(results_dir)
    robustness = load_robustness_metrics(results_dir)

    for maybe_path in [
        plot_mrr_by_model(metrics, output_dir / "mrr_by_model.png"),
        plot_hitrate_by_model(metrics, output_dir / "hitrate_by_model.png"),
        plot_bucket_metrics(buckets, output_dir / "bucket_mrr.png"),
        plot_robustness(robustness, output_dir / "robustness_mrr.png"),
    ]:
        if maybe_path is not None:
            created.append(Path(maybe_path))
    return created
