from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .normalize import normalize_name


def load_synonym_csv(
    csv_path: str | Path,
    cid_col: str = "cid",
    name_col: str = "name",
    normalize_names: bool = True,
    filter_empty_names: bool = True,
    remove_singletons: bool = True,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    missing = [col for col in (cid_col, name_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")

    if limit_rows is not None:
        df = df.head(int(limit_rows)).copy()

    df = df[[cid_col, name_col]].copy()
    df = df.rename(columns={cid_col: "cid", name_col: "name"})
    df["original_name"] = df["name"]

    if normalize_names:
        df["name"] = df["name"].map(normalize_name)

    if filter_empty_names:
        df = df[df["name"].notna()].copy()
        df = df[df["name"].astype(str).str.len() > 0].copy()

    df = df[df["cid"].notna()].copy()
    df = df.reset_index(drop=True)
    df = add_cid_sizes(df)

    if remove_singletons:
        df = df[df["cid_size"] >= 2].copy().reset_index(drop=True)
        df = add_cid_sizes(df)

    if df.empty:
        raise ValueError("No rows remain after filtering. Check CSV path, column names, and singleton filtering.")

    return df


def add_cid_sizes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sizes = out["cid"].value_counts()
    out["cid_size"] = out["cid"].map(sizes).astype(np.int32)
    return out


def dataframe_arrays(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "names": df["name"].astype(str).tolist(),
        "ids": df["cid"].to_numpy(),
        "cid_size_arr": df["cid_size"].to_numpy(dtype=np.int32),
    }


def describe_data(df: pd.DataFrame) -> dict[str, Any]:
    cid_sizes = df["cid_size"]
    return {
        "n_rows": int(len(df)),
        "n_unique_cids": int(df["cid"].nunique()),
        "min_cid_size": int(cid_sizes.min()),
        "max_cid_size": int(cid_sizes.max()),
        "mean_cid_size": float(cid_sizes.mean()),
        "median_cid_size": float(cid_sizes.median()),
    }
