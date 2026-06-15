from __future__ import annotations

import random
import string
from typing import Callable


Operation = Callable[[str, random.Random], str]


def typo(text: str, rng: random.Random) -> str:
    if not text:
        return text
    positions = [i for i, ch in enumerate(text) if ch.isalnum()]
    if not positions:
        return text
    pos = rng.choice(positions)
    replacement = rng.choice(string.ascii_lowercase + string.digits)
    return text[:pos] + replacement + text[pos + 1 :]


def character_swap(text: str, rng: random.Random) -> str:
    if len(text) < 2:
        return text
    pos = rng.randrange(0, len(text) - 1)
    chars = list(text)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def drop_punctuation(text: str, rng: random.Random) -> str:
    del rng
    return "".join(ch for ch in text if ch not in string.punctuation)


def drop_token(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    pos = rng.randrange(0, len(tokens))
    return " ".join(tokens[:pos] + tokens[pos + 1 :])


def spacing_change(text: str, rng: random.Random) -> str:
    if " " in text:
        if rng.random() < 0.5:
            return text.replace(" ", "")
        return text.replace(" ", "  ")
    if len(text) > 4:
        pos = rng.randrange(1, len(text) - 1)
        return text[:pos] + " " + text[pos:]
    return text


OPERATIONS: dict[str, Operation] = {
    "typo": typo,
    "swap": character_swap,
    "character_swap": character_swap,
    "drop_punctuation": drop_punctuation,
    "dropped_punctuation": drop_punctuation,
    "drop_token": drop_token,
    "dropped_token": drop_token,
    "spacing": spacing_change,
    "spacing_change": spacing_change,
}


def corrupt_text(text: str, level: float, rng: random.Random, operations: list[str]) -> str:
    if level <= 0:
        return text
    out = text
    applied = False
    for name in operations:
        op = OPERATIONS[name]
        if rng.random() <= level:
            out = op(out, rng)
            applied = True
    if not applied and operations:
        out = OPERATIONS[rng.choice(operations)](out, rng)
    return out


def corrupt_many(
    texts: list[str],
    level: float,
    seed: int,
    operations: list[str] | None = None,
) -> list[str]:
    operations = operations or ["typo", "swap", "drop_punctuation", "drop_token", "spacing"]
    unknown = [name for name in operations if name not in OPERATIONS]
    if unknown:
        raise ValueError(f"Unknown corruption operation(s): {unknown}")
    rng = random.Random(seed)
    return [corrupt_text(str(text), float(level), rng, operations) for text in texts]
