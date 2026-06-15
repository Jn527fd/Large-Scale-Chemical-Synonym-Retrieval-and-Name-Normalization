from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .utils import load_json, save_json


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_hf_model(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def mean_pool(last_hidden_state: Any, attention_mask: Any):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: Any, attention_mask: Any):
    return last_hidden_state[:, 0]


def pool_outputs(last_hidden_state: Any, attention_mask: Any, pooling: str):
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    if pooling == "cls":
        return cls_pool(last_hidden_state, attention_mask)
    raise ValueError(f"Unknown pooling: {pooling}")


def numpy_dtype(name: str | type | np.dtype) -> np.dtype:
    if isinstance(name, np.dtype):
        return name
    if name in (np.float16, "float16", "np.float16"):
        return np.dtype("float16")
    if name in (np.float32, "float32", "np.float32"):
        return np.dtype("float32")
    raise ValueError(f"Unsupported embedding dtype: {name}")


def encode_texts_to_memmap(
    texts: list[str],
    out_path: str | Path,
    tokenizer: Any,
    model: Any,
    model_name: str,
    prefix: str | None = None,
    pooling: str = "mean",
    batch_size: int = 128,
    max_length: int = 96,
    device: str = "cpu",
    normalize: bool = True,
    out_dtype: str | type | np.dtype = "float16",
    run_key: str | None = None,
) -> np.ndarray:
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = numpy_dtype(out_dtype)
    n = len(texts)
    d = int(model.config.hidden_size)
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    expected_meta = {
        "n": n,
        "d": d,
        "prefix": prefix,
        "pooling": pooling,
        "max_length": int(max_length),
        "model_name": model_name,
        "run_key": run_key,
        "dtype": np_dtype.name,
        "normalize": bool(normalize),
    }

    if out_path.exists() and meta_path.exists():
        try:
            old_meta = load_json(meta_path)
            arr = np.load(out_path, mmap_mode="r")
            if old_meta == expected_meta and arr.shape == (n, d):
                return arr
        except Exception:
            pass

    if out_path.exists():
        out_path.unlink()

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np_dtype, shape=(n, d))
    autocast_enabled = device.startswith("cuda") and torch.cuda.is_available()
    t0 = time.perf_counter()

    for start in tqdm(range(0, n, int(batch_size)), desc=f"Encoding {out_path.name}"):
        end = min(start + int(batch_size), n)
        batch = [str(x) for x in texts[start:end]]
        if prefix is not None:
            batch = [prefix + x for x in batch]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                out = model(**enc)
                emb = pool_outputs(out.last_hidden_state, enc["attention_mask"], pooling)
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        arr[start:end] = emb.detach().cpu().numpy().astype(np_dtype, copy=False)
        del enc, out, emb, batch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    arr.flush()
    elapsed = time.perf_counter() - t0
    save_json(expected_meta, meta_path)
    save_json({"elapsed_sec": elapsed}, out_path.with_suffix(out_path.suffix + ".timing.json"))
    return np.load(out_path, mmap_mode="r")
