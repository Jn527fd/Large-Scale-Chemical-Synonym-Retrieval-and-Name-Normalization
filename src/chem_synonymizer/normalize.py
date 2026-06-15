from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalize_name(value: object) -> str:
    """Normalize a chemical synonym surface form, matching the prototype."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
