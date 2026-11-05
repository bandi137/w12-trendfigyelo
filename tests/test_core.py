"""Offline tesztek: kitalált cikkeken, TF-IDF-vektorokkal és szimulált LLM-mel (GPU és API-kulcs nélkül).
Futtatás a repó gyökeréből:  python -m pytest -q
"""
import json
import random
import re
import types
from datetime import date

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import trendfigyelo as tf

TOPICS = {
    "agent": "agents tool planning autonomous agent benchmark web navigation",
    "mt": "translation low-resource bleu parallel corpus translation",
    "halluc": "hallucination factuality grounding detection faithfulness",
    "speech": "speech recognition asr audio acoustic spoken",
}
GENERIC = "large language model models paper propose results show approach performance".split()
MONTHS = [f"2025-{m:02d}" for m in (11, 12)] + [f"2026-{m:02d}" for m in range(1, 11)]


@pytest.fixture(scope="module")
def corpus():
    """Az 'agent' téma nő, az 'mt' csökken, a másik kettő állandó. Egy cikkben ritka zajszó van."""
    random.seed(0)
    rows, i = [], 0
    for mi, m in enumerate(MONTHS):
        weights = {"agent": 1 + mi, "mt": 12 - mi, "halluc": 6, "speech": 6}
        for t, n in weights.items():
            for _ in range(n * 3):
                i += 1
                words = TOPICS[t].split()
                abstract = " ".join(random.choices(words, k=20) + random.choices(GENERIC, k=15))
                rows.append({"id": f"26{mi:02d}.{i:05d}", "title": " ".join(random.sample(words, 4)),
                             "abstract": abstract, "month": m, "published": m + "-15", "true": t})
    df = pd.DataFrame(rows)
    df.loc[0, "abstract"] += " tajik odrl sealed"          # ritka zaj: nem lehet kulcsszó
    texts = (df.title + ". " + df.abstract).tolist()
    vec = TfidfVectorizer().fit(texts)
    svd = TruncatedSVD(20, random_state=0).fit(vec.transform(texts))

    def enc(t):
        v = svd.transform(vec.transform(t))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return df, enc(texts), lambda q: enc([q])[0]


class FakeLLM:
    """A google-genai kliens felületét utánozza; a témaelnevezésnél szándékosan kihagy egy témát."""
    def __init__(self):
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.calls.append(contents)
        if "TÉMÁK:" in contents:
            ids = re.findall(r"^Téma (\d+)$", contents, re.M)
            data = {i: {"nev": f"Téma {i} (Topic {i})", "leiras": f"Leírás {i}."} for i in ids[:-1]}
            return types.SimpleNamespace(text=json.dumps(data, ensure_ascii=False))
        if "CIKKEK:" not in contents:
            return types.SimpleNamespace(text="Összefoglaló.")
        hit = re.search(r"\[(\d{4}\.\d{5})\]", contents.split("CIKKEK:")[1])   # a megkapott cikkek közül
        return types.SimpleNamespace(text=f"Válasz [{hit.group(1)}] és kitalált [2601.99999].")


def test_month_ranges():
    r = tf.month_ranges(12, today=date(2026, 11, 4))
    assert len(r) == 12
    assert r[0] == ("202511010000", "202512010000", "2025-11")
    assert r[-1] == ("202610010000", "202611010000", "2026-10")
    assert tf.month_ranges(1, today=date(2026, 1, 20))[0][2] == "2025-12"


def test_retrieve_and_month_filter(corpus):
    df, emb, q = corpus
    hits = tf.retrieve(df, emb, q("hallucination grounding factuality"), k=5)
    assert (df.loc[hits.index, "true"] == "halluc").all()
    late = tf.retrieve(df, emb, q("agents tool use"), k=5, month_from="2026-06")
    assert (late.month >= "2026-06").all()


def test_rag_ask_flags_invented_citation(corpus):
    df, emb, q = corpus
    rag = tf.RAG(df, emb, q, FakeLLM(), "fake", top_k=3)
    rag.ask("speech recognition audio", show=False)
    entry = rag.log[-1]
    assert entry["invalid"] == ["2601.99999"]
    valid = set(entry["cited"]) - set(entry["invalid"])
    assert len(valid) == 1 and valid.pop() in entry["sources"]


def test_topics_keywords_and_growth(corpus):
    df, emb, _ = corpus
    labels, keywords = tf.find_topics(emb, (df.title + ". " + df.abstract).values, n_topics=4)
    all_kw = " ".join(keywords.values())
    for noise in ("tajik", "odrl", "sealed", "model", "paper"):
        assert noise not in all_kw.split(", "), noise
    share, growth = tf.topic_trends(df.month, labels)
    by_true = pd.Series(labels).groupby(df["true"]).agg(lambda s: s.mode()[0])
    assert growth.index[0] == by_true["agent"]          # leggyorsabban nő
    assert growth.index[-1] == by_true["mt"]            # leggyorsabban csökken
    assert "agent" in keywords[by_true["agent"]]


def test_naming_cache_table_plot_summary(corpus, tmp_path):
    df, emb, _ = corpus
    df = df.copy()
    df["topic"], keywords = tf.find_topics(emb, (df.title + ". " + df.abstract).values, n_topics=4)
    share, growth = tf.topic_trends(df.month, df.topic)
    ids = [int(t) for t in growth.index]
    llm = FakeLLM()
    names = tf.name_topics(llm, "fake", df, keywords, ids, "cs.CL", str(tmp_path))
    assert names[ids[0]]["nev"].startswith("Téma")
    assert names[ids[-1]]["nev"] == keywords[ids[-1]].split(", ")[0]   # kihagyott téma -> tartalék név
    n_calls = len(llm.calls)
    tf.name_topics(llm, "fake", df, keywords, ids, "cs.CL", str(tmp_path))
    assert len(llm.calls) == n_calls                                   # másodszor a gyorsítótárból
    table = tf.trend_table(df, ids, names, keywords, growth)
    assert list(table.columns) == ["topic", "név", "leírás", "kulcsszavak", "cikkek", "növekedés"]
    assert table.cikkek.sum() == len(df)
    tf.style_table(table).to_html()
    tf.plot_trends(share, growth, names, ids, "cs.CL", path=str(tmp_path / "t.png"))
    assert (tmp_path / "t.png").exists()
    assert tf.summarize_trends(llm, "fake", names, growth, ids, "cs.CL")


def test_parse_names_robust():
    kw = {0: "agent, tool", 1: "speech, asr"}
    assert tf.parse_names("nem json", [0, 1], kw)[0]["nev"] == "agent"
    out = tf.parse_names('```json\n{"0": {"nev": "Ágensek (Agents)", "leiras": "x"}}\n```', [0, 1], kw)
    assert out[0]["nev"] == "Ágensek (Agents)" and out[1]["nev"] == "speech"


def test_evaluate(corpus):
    df, emb, q = corpus
    top = tf.retrieve(df, emb, q("speech recognition audio"), k=10).id.tolist()
    res, summary = tf.evaluate([{"q": "speech recognition audio", "expected": [top[3]]}],
                               lambda question, k: tf.retrieve(df, emb, q(question), k=k))
    assert summary.to_dict() == {"hit@1": 0.0, "hit@5": 1.0, "hit@10": 1.0}


def test_generate_retries_on_quota(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    state = {"n": 0}

    class Flaky:
        def __init__(self):
            self.models = self

        def generate_content(self, model, contents, config=None):
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return types.SimpleNamespace(text="ok")

    assert tf.generate(Flaky(), "m", "p") == "ok" and state["n"] == 3
    with pytest.raises(RuntimeError):
        tf.generate(types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content=lambda **k: (_ for _ in ()).throw(RuntimeError("404 NOT_FOUND")))), "m", "p")
