from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from chem_synonymizer.config import get_model_preset, load_config
from chem_synonymizer.data import describe_data, load_synonym_csv
from chem_synonymizer.embed import choose_device
from chem_synonymizer.finetune import build_synonym_training_data, train_synonym_model
from chem_synonymizer.split import sample_query_indices, split_unseen_compounds
from chem_synonymizer.utils import (
    file_fingerprint,
    make_run_dir,
    resolve_model_name,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SapBERT on CID/name synonym sets.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--base-model-key", default="sapbert")
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--objective", choices=["contrastive", "triplet"], required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-frac", type=float, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-hard-negatives", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def training_indices(df: pd.DataFrame, split_name: str, eval_frac: float, seed: int) -> tuple[pd.Series, pd.Series]:
    if split_name == "all":
        train_idx = pd.Series(range(len(df)), dtype="int64")
        eval_idx = pd.Series(sample_query_indices(len(df), eval_frac, seed), dtype="int64")
        return train_idx, eval_idx
    if split_name == "compound":
        split = split_unseen_compounds(df, eval_frac, seed)
        return pd.Series(split["train_idx"], dtype="int64"), pd.Series(split["eval_idx"], dtype="int64")
    if split_name != "surface":
        raise ValueError(f"Unknown fine-tuning split: {split_name}")

    eval_idx = sample_query_indices(len(df), eval_frac, seed)
    train_mask = pd.Series(True, index=range(len(df)))
    train_mask.iloc[eval_idx] = False
    train_idx = pd.Series(train_mask[train_mask].index.to_numpy(dtype="int64"))
    return train_idx, pd.Series(eval_idx, dtype="int64")


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

    finetune_cfg = cfg.get("finetune", {})
    run_name = args.run_name or f"sapbert_{args.objective}"
    results_dir = resolve_path(cfg["project"]["results_dir"])
    run_dir = make_run_dir(results_dir, run_name)
    artifacts_dir = run_dir / "artifacts"
    output_dir = Path(args.output_dir) if args.output_dir else artifacts_dir / "model"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if output_dir.exists() and (output_dir / "config.json").exists() and not args.overwrite:
        print(f"Fine-tuned model already exists, skipping training: {output_dir}")
        print("Pass --overwrite to retrain this checkpoint.")
        return

    seed = int(cfg["eval"]["seed"])
    set_seed(seed)
    csv_path = resolve_path(cfg["data"]["csv_path"])
    device = choose_device(cfg.get("compute", {}).get("device", "auto"))
    split_name = str(finetune_cfg.get("split", "surface"))

    df = load_synonym_csv(
        csv_path=csv_path,
        cid_col=cfg["data"]["cid_col"],
        name_col=cfg["data"]["name_col"],
        normalize_names=cfg["data"]["normalize_names"],
        filter_empty_names=cfg["data"]["filter_empty_names"],
        remove_singletons=cfg["data"]["remove_singletons"],
        limit_rows=cfg["data"].get("limit_rows"),
    )
    train_idx, eval_idx = training_indices(df, split_name, float(cfg["eval"]["eval_frac"]), seed)
    train_data = build_synonym_training_data(df, train_idx.to_numpy(dtype="int64"))

    base_model = get_model_preset(cfg, args.base_model_key)
    if args.base_model_name:
        base_model["model_name"] = args.base_model_name
    base_model["model_name"] = resolve_model_name(ROOT, base_model["model_name"])
    cfg["runtime"] = {
        "baseline": "fine_tune",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "csv": file_fingerprint(csv_path),
        "data": describe_data(df),
        "base_model": base_model,
        "objective": args.objective,
        "split": split_name,
        "n_train_rows": int(len(train_idx)),
        "n_eval_rows": int(len(eval_idx)),
        "n_trainable_cids": int(len(train_data.groups)),
        "device": device,
        "use_hard_negatives": bool(args.use_hard_negatives),
    }
    save_json(cfg, run_dir / "finetune_config.json")
    df.to_csv(artifacts_dir / "prepared_data.csv", index=False)
    train_idx.to_csv(artifacts_dir / "finetune_train_indices.csv", index=False, header=["row_index"])
    eval_idx.to_csv(artifacts_dir / "finetune_eval_indices.csv", index=False, header=["row_index"])

    print(f"Run directory: {run_dir}")
    print(f"Output model: {output_dir}")
    print(f"Rows after filtering: {len(df):,}")
    print(f"Fine-tune split: {split_name}; train rows: {len(train_idx):,}; held-out rows: {len(eval_idx):,}")
    print(f"Trainable CID groups: {len(train_data.groups):,}")
    print(f"Loading base model: {base_model['model_name']} on {device}")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model["model_name"], use_fast=True)
    model = AutoModel.from_pretrained(base_model["model_name"])

    t0 = time.perf_counter()
    history = train_synonym_model(
        data=train_data,
        tokenizer=tokenizer,
        model=model,
        output_dir=output_dir,
        objective=args.objective,
        device=device,
        pooling=base_model.get("pooling", "cls"),
        seed=seed,
        epochs=int(finetune_cfg.get("epochs", 1)),
        steps_per_epoch=int(finetune_cfg.get("steps_per_epoch", 2000)),
        batch_size=int(finetune_cfg.get("batch_size", 128)),
        max_length=int(cfg["model"]["max_length"]),
        learning_rate=float(finetune_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(finetune_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(finetune_cfg.get("warmup_ratio", 0.06)),
        gradient_accumulation_steps=int(finetune_cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(finetune_cfg.get("max_grad_norm", 1.0)),
        temperature=float(finetune_cfg.get("temperature", 0.05)),
        margin=float(finetune_cfg.get("margin", 0.2)),
        mixed_precision=str(finetune_cfg.get("mixed_precision", "fp16")),
        use_hard_negatives=bool(args.use_hard_negatives),
        hard_negative_prefix_len=int(finetune_cfg.get("hard_negative_prefix_len", 4)),
    )
    elapsed = time.perf_counter() - t0
    pd.DataFrame(history).to_csv(run_dir / "finetune_history.csv", index=False)
    save_json(
        {
            "run_name": run_dir.name,
            "baseline": "fine_tune",
            "base_model_key": base_model["key"],
            "objective": args.objective,
            "use_hard_negatives": bool(args.use_hard_negatives),
            "output_dir": str(output_dir),
            "elapsed_sec": elapsed,
            "last_history": history[-1] if history else None,
        },
        run_dir / "finetune_metrics.json",
    )

    print(f"Saved fine-tuned checkpoint to {output_dir}")
    print(f"Saved fine-tuning history to {run_dir / 'finetune_history.csv'}")


if __name__ == "__main__":
    main()
