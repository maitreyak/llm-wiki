"""SQuAD-style answer metrics: normalized token F1 and exact match."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    # yes/no/noanswer must match exactly (HotpotQA convention)
    if normalize_answer(gold) in ("yes", "no", "noanswer"):
        if normalize_answer(prediction) != normalize_answer(gold):
            return 0.0
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_scores(prediction: str, golds: list[str]) -> tuple[float, float]:
    """(F1, EM) maximized over acceptable gold answers."""
    if not golds:
        return 0.0, 0.0
    f1 = max(f1_score(prediction, g) for g in golds)
    em = max(exact_match(prediction, g) for g in golds)
    return f1, em
