from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .utils import ensure_contiguous_f32, load_json, save_json


def _gpu_id(device: str) -> int:
    if ":" in device:
        return int(device.split(":", 1)[1])
    return 0


def faiss_gpu_available(faiss: Any, device: str) -> bool:
    return device.startswith("cuda") and hasattr(faiss, "StandardGpuResources")


def build_faiss_index_from_memmap(
    xb_memmap: np.ndarray,
    index_path: str | Path,
    index_type: str = "flat",
    nlist: int = 4096,
    nprobe: int = 64,
    train_size: int = 200_000,
    add_batch_size: int = 250_000,
    m_pq: int = 64,
    nbits: int = 8,
    seed: int = 0,
    device: str = "cpu",
    use_float16: bool = True,
    run_key: str | None = None,
) -> tuple[Any, Any]:
    import faiss

    n, d = xb_memmap.shape
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")

    if index_type != "flat":
        nlist_eff = min(max(int(nlist), 1), max(int(n), 1), max(int(train_size), 1))
    else:
        nlist_eff = int(nlist)

    expected_meta = {
        "index_type": index_type,
        "n": int(n),
        "d": int(d),
        "nlist": int(nlist_eff),
        "train_size": int(train_size),
        "m_pq": int(m_pq),
        "nbits": int(nbits),
        "seed": int(seed),
        "run_key": run_key,
    }

    if index_path.exists() and meta_path.exists():
        try:
            old_meta = load_json(meta_path)
            if old_meta == expected_meta:
                cpu_index = faiss.read_index(str(index_path))
                return _maybe_move_to_gpu(faiss, cpu_index, device, use_float16, nprobe), cpu_index
        except Exception:
            pass

    quantizer = faiss.IndexFlatIP(d)
    if index_type == "flat":
        cpu_index = faiss.IndexFlatIP(d)
    elif index_type == "ivf_flat":
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist_eff, faiss.METRIC_INNER_PRODUCT)
    elif index_type == "ivfpq":
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist_eff, m_pq, nbits)
        cpu_index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    index = _maybe_move_to_gpu(faiss, cpu_index, device, use_float16, nprobe)

    if index_type != "flat":
        rng = np.random.default_rng(seed)
        train_size_eff = min(int(train_size), int(n))
        train_idx = np.sort(rng.choice(n, size=train_size_eff, replace=False))
        xtrain = ensure_contiguous_f32(xb_memmap[train_idx])
        index.train(xtrain)
        del xtrain

    t0 = time.perf_counter()
    for start in tqdm(range(0, n, int(add_batch_size)), desc="FAISS add"):
        end = min(start + int(add_batch_size), n)
        xb = ensure_contiguous_f32(xb_memmap[start:end])
        index.add(xb)
        del xb

    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)

    if faiss_gpu_available(faiss, device):
        cpu_final = faiss.index_gpu_to_cpu(index)
    else:
        cpu_final = index

    faiss.write_index(cpu_final, str(index_path))
    elapsed = time.perf_counter() - t0
    save_json(expected_meta, meta_path)
    save_json({"elapsed_sec": elapsed}, index_path.with_suffix(index_path.suffix + ".timing.json"))
    return index, cpu_final


def _maybe_move_to_gpu(faiss: Any, cpu_index: Any, device: str, use_float16: bool, nprobe: int) -> Any:
    if faiss_gpu_available(faiss, device):
        res = faiss.StandardGpuResources()
        opts = faiss.GpuClonerOptions()
        opts.useFloat16 = bool(use_float16)
        gpu_index = faiss.index_cpu_to_gpu(res, _gpu_id(device), cpu_index, opts)
        if hasattr(gpu_index, "nprobe"):
            gpu_index.nprobe = int(nprobe)
        return gpu_index
    if hasattr(cpu_index, "nprobe"):
        cpu_index.nprobe = int(nprobe)
    return cpu_index
