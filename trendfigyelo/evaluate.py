"""Visszakeresés kiértékelése: hit@k egy kézzel összeállított kérdéslistán."""
import pandas as pd


def hit_at_k(results, expected, k):
    """1, ha a várt cikkek közül legalább egy benne van az első k találatban."""
    return int(bool(set(results[:k]) & set(expected)))


def evaluate(questions, retrieve_fn):
    """questions: [{"q": ..., "expected": [arXiv-azonosítók]}]; retrieve_fn(kérdés, k) -> találatok DataFrame-je."""
    rows = []
    for q in questions:
        ids = retrieve_fn(q["q"], 10).id.tolist()
        rows.append({"kérdés": q["q"], "hit@1": hit_at_k(ids, q["expected"], 1),
                     "hit@5": hit_at_k(ids, q["expected"], 5), "hit@10": hit_at_k(ids, q["expected"], 10)})
    res = pd.DataFrame(rows)
    return res, res[["hit@1", "hit@5", "hit@10"]].mean().round(2)
