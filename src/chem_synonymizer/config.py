from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import coerce_scalar


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "e5_base": {
        "model_name": "intfloat/e5-base-v2",
        "corpus_prefix": "passage: ",
        "query_prefix": "query: ",
        "pooling": "mean",
    },
    "e5_large": {
        "model_name": "intfloat/e5-large-v2",
        "corpus_prefix": "passage: ",
        "query_prefix": "query: ",
        "pooling": "mean",
    },
    "bge_base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "corpus_prefix": None,
        "query_prefix": None,
        "pooling": "cls",
    },
    "biomedbert": {
        "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "corpus_prefix": None,
        "query_prefix": None,
        "pooling": "mean",
    },
    "sapbert": {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "corpus_prefix": None,
        "query_prefix": None,
        "pooling": "cls",
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(out, key, coerce_scalar(value))
    return out


def load_config(
    config_path: str | Path,
    models_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    presets = dict(MODEL_PRESETS)
    if models_path:
        model_file = load_yaml(models_path)
        presets = deep_merge(presets, model_file.get("model_presets", model_file))
    config["model_presets"] = deep_merge(presets, config.get("model_presets", {}))
    return apply_cli_overrides(config, overrides)


def get_model_preset(config: dict[str, Any], model_key: str | None = None) -> dict[str, Any]:
    key = model_key or config.get("model", {}).get("key")
    presets = config.get("model_presets", MODEL_PRESETS)
    if key not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown model key '{key}'. Available: {available}")
    return {"key": key, **presets[key]}


def resolve_project_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path
