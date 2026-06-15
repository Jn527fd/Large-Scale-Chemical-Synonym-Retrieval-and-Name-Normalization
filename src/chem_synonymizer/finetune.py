from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .embed import pool_outputs
from .utils import save_json


@dataclass
class SynonymTrainingData:
    names: list[str]
    ids: np.ndarray
    train_idx: np.ndarray
    groups: list[np.ndarray]
    group_cids: list[Any]


def build_synonym_training_data(df: pd.DataFrame, train_idx: np.ndarray) -> SynonymTrainingData:
    names = df["name"].astype(str).tolist()
    ids = df["cid"].to_numpy()
    groups_by_cid: dict[Any, list[int]] = defaultdict(list)
    for raw_i in np.asarray(train_idx, dtype=np.int64):
        i = int(raw_i)
        groups_by_cid[ids[i]].append(i)

    groups = []
    group_cids = []
    for cid, rows in groups_by_cid.items():
        if len(rows) < 2:
            continue
        groups.append(np.asarray(rows, dtype=np.int64))
        group_cids.append(cid)

    if not groups:
        raise ValueError("No trainable synonym groups remain. Need at least one CID with two train names.")

    return SynonymTrainingData(
        names=names,
        ids=ids,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        groups=groups,
        group_cids=group_cids,
    )


def _hard_key(text: str, prefix_len: int) -> tuple[str, int]:
    compact = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    prefix = compact[: max(int(prefix_len), 1)] or "<empty>"
    length_bin = min(len(compact) // 8, 24)
    return prefix, length_bin


class SynonymBatchFactory:
    def __init__(
        self,
        data: SynonymTrainingData,
        seed: int,
        use_hard_negatives: bool = False,
        hard_negative_prefix_len: int = 4,
    ) -> None:
        self.data = data
        self.rng = np.random.default_rng(seed)
        self.use_hard_negatives = bool(use_hard_negatives)
        self.hard_negative_prefix_len = int(hard_negative_prefix_len)
        self.group_count = len(data.groups)
        self.group_ids = np.arange(self.group_count, dtype=np.int64)
        self.train_pool = np.concatenate(data.groups)
        self.prefix_buckets = self._build_prefix_buckets() if self.use_hard_negatives else {}

    def _build_prefix_buckets(self) -> dict[tuple[str, int], np.ndarray]:
        buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i in self.train_pool:
            buckets[_hard_key(self.data.names[int(i)], self.hard_negative_prefix_len)].append(int(i))
        return {key: np.asarray(rows, dtype=np.int64) for key, rows in buckets.items() if len(rows) > 1}

    def _sample_two(self, group: np.ndarray) -> tuple[int, int]:
        pair = self.rng.choice(group, size=2, replace=False)
        return int(pair[0]), int(pair[1])

    def _sample_random_negative(self, anchor_cid: Any) -> int:
        for _ in range(100):
            group_i = int(self.rng.integers(0, self.group_count))
            if self.data.group_cids[group_i] != anchor_cid:
                return int(self.rng.choice(self.data.groups[group_i]))
        candidates = self.train_pool[self.data.ids[self.train_pool] != anchor_cid]
        if len(candidates) == 0:
            raise ValueError("Could not sample a negative row from a different CID.")
        return int(self.rng.choice(candidates))

    def _sample_hard_negative(self, anchor_idx: int, anchor_cid: Any) -> int:
        key = _hard_key(self.data.names[int(anchor_idx)], self.hard_negative_prefix_len)
        bucket = self.prefix_buckets.get(key)
        if bucket is not None:
            for _ in range(30):
                candidate = int(self.rng.choice(bucket))
                if self.data.ids[candidate] != anchor_cid:
                    return candidate
        return self._sample_random_negative(anchor_cid)

    def contrastive_batch(self, batch_size: int) -> tuple[list[str], torch.Tensor]:
        batch_size = max(int(batch_size), 2)
        replace = self.group_count < batch_size
        chosen_groups = self.rng.choice(self.group_ids, size=batch_size, replace=replace)
        row_indices: list[int] = []
        labels: list[int] = []
        for group_i in chosen_groups:
            a, p = self._sample_two(self.data.groups[int(group_i)])
            row_indices.extend([a, p])
            labels.extend([int(group_i), int(group_i)])
        texts = [self.data.names[i] for i in row_indices]
        return texts, torch.tensor(labels, dtype=torch.long)

    def triplet_batch(self, batch_size: int) -> tuple[list[str], int]:
        batch_size = max(int(batch_size), 1)
        replace = self.group_count < batch_size
        chosen_groups = self.rng.choice(self.group_ids, size=batch_size, replace=replace)
        anchors: list[int] = []
        positives: list[int] = []
        negatives: list[int] = []
        for group_i in chosen_groups:
            group_i = int(group_i)
            anchor_idx, positive_idx = self._sample_two(self.data.groups[group_i])
            anchor_cid = self.data.group_cids[group_i]
            if self.use_hard_negatives:
                negative_idx = self._sample_hard_negative(anchor_idx, anchor_cid)
            else:
                negative_idx = self._sample_random_negative(anchor_cid)
            anchors.append(anchor_idx)
            positives.append(positive_idx)
            negatives.append(negative_idx)

        ordered = anchors + positives + negatives
        texts = [self.data.names[i] for i in ordered]
        return texts, batch_size


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = labels.to(embeddings.device)
    sim = torch.matmul(embeddings, embeddings.T) / float(temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=embeddings.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    logits_mask = ~eye
    exp_logits = torch.exp(sim) * logits_mask.float()
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    mean_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1)
    return -mean_log_prob[valid].mean()


def triplet_cosine_loss(
    embeddings: torch.Tensor,
    triplet_count: int,
    margin: float,
) -> torch.Tensor:
    anchors = embeddings[:triplet_count]
    positives = embeddings[triplet_count : 2 * triplet_count]
    negatives = embeddings[2 * triplet_count : 3 * triplet_count]
    pos_sim = (anchors * positives).sum(dim=1)
    neg_sim = (anchors * negatives).sum(dim=1)
    return torch.relu(neg_sim - pos_sim + float(margin)).mean()


def encode_train_batch(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    max_length: int,
    device: str,
    pooling: str,
    mixed_precision: str,
) -> torch.Tensor:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    ).to(device)
    if mixed_precision == "bf16":
        dtype = torch.bfloat16
        enabled = device.startswith("cuda")
    elif mixed_precision == "fp16":
        dtype = torch.float16
        enabled = device.startswith("cuda")
    else:
        dtype = torch.float32
        enabled = False

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=dtype, enabled=enabled):
        out = model(**enc)
        embeddings = pool_outputs(out.last_hidden_state, enc["attention_mask"], pooling)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def train_synonym_model(
    *,
    data: SynonymTrainingData,
    tokenizer: Any,
    model: Any,
    output_dir: str | Path,
    objective: str,
    device: str,
    pooling: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    temperature: float,
    margin: float,
    mixed_precision: str,
    use_hard_negatives: bool,
    hard_negative_prefix_len: int,
) -> list[dict[str, Any]]:
    from transformers import get_linear_schedule_with_warmup

    if objective not in {"contrastive", "triplet"}:
        raise ValueError(f"Unknown fine-tuning objective: {objective}")

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.train()

    factory = SynonymBatchFactory(
        data=data,
        seed=seed,
        use_hard_negatives=use_hard_negatives,
        hard_negative_prefix_len=hard_negative_prefix_len,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    total_batches = max(int(epochs), 1) * max(int(steps_per_epoch), 1)
    grad_accum = max(int(gradient_accumulation_steps), 1)
    total_optim_steps = max((total_batches + grad_accum - 1) // grad_accum, 1)
    warmup_steps = int(total_optim_steps * float(warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)
    use_scaler = mixed_precision == "fp16" and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history: list[dict[str, Any]] = []
    rolling_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(total_batches), desc=f"Fine-tuning SapBERT ({objective})")
    t0 = time.perf_counter()

    for batch_i in progress:
        if objective == "contrastive":
            texts, labels = factory.contrastive_batch(batch_size)
            embeddings = encode_train_batch(texts, tokenizer, model, max_length, device, pooling, mixed_precision)
            loss = supervised_contrastive_loss(embeddings, labels, temperature)
        else:
            texts, triplet_count = factory.triplet_batch(batch_size)
            embeddings = encode_train_batch(texts, tokenizer, model, max_length, device, pooling, mixed_precision)
            loss = triplet_cosine_loss(embeddings, triplet_count, margin)

        scaled_loss = loss / grad_accum
        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = ((batch_i + 1) % grad_accum == 0) or (batch_i + 1 == total_batches)
        if should_step:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().cpu())
        rolling_loss = (0.95 * rolling_loss) + (0.05 * loss_value) if batch_i else loss_value
        progress.set_postfix(loss=f"{rolling_loss:.4f}")
        if (batch_i + 1) % max(int(steps_per_epoch), 1) == 0:
            epoch = (batch_i + 1) // max(int(steps_per_epoch), 1)
            history.append(
                {
                    "epoch": int(epoch),
                    "step": int(batch_i + 1),
                    "loss": loss_value,
                    "rolling_loss": float(rolling_loss),
                    "elapsed_sec": float(time.perf_counter() - t0),
                }
            )

        del embeddings, loss, scaled_loss, texts
        if device.startswith("cuda") and (batch_i + 1) % 25 == 0:
            torch.cuda.empty_cache()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json({"history": history}, output_dir / "training_history.json")
    return history
