from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[1]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "run"


def make_run_dir(results_dir: str | Path, run_name: str | None = None, stable: bool = True) -> Path:
    name = slugify(run_name) if run_name else "run"
    run_dir = Path(results_dir) / name
    if not stable:
        from datetime import datetime

        run_dir = Path(results_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    return run_dir


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha1_16": file_sha1(path)[:16],
    }


def resolve_model_name(root: str | Path, model_name: str | Path) -> str:
    value = str(model_name)
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    candidate = Path(root) / path
    if candidate.exists():
        return str(candidate.resolve())
    return value


def local_model_fingerprint(model_name: str | Path) -> dict[str, Any] | None:
    path = Path(str(model_name))
    if not path.exists():
        return None

    files = []
    for rel in [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        item = path / rel
        if not item.exists():
            continue
        stat = item.stat()
        files.append({"name": rel, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})

    return {"type": "local", "path": str(path.resolve()), "files": files}


def file_sha1(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(to_jsonable(config), sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def array_sha1(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_contiguous_f32(x: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [coerce_scalar(v.strip()) for v in value.split(",")]
    return value
