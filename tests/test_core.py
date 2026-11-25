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
    last = df.month.max()                                    # kevesebb cikk a kért számnál: nem tölt fel régiekkel
    n_last = int((df.month == last).sum())
    few = tf.retrieve(df, emb, q("agents tool use"), k=n_last + 10, month_from=last)
    assert len(few) == n_last and (few.month == last).all() and (few.score > -1).all()
    assert list(few.score) == sorted(few.score, reverse=True)
    assert tf.retrieve(df, emb, q("agents"), k=5, month_from="2099-01").empty   # üres időszak: nincs találat


def test_rag_ask_empty_period_does_not_call_llm(corpus):
    df, emb, q = corpus
    class NoLLM:
        @property
        def models(self): raise AssertionError("a modellt nem szabad hívni")
    rag = tf.RAG(df, emb, q, NoLLM(), "fake")
    answer = rag.ask("Milyen ágensek vannak?", month_from="2099-01", show=False)
    assert "nincs cikk" in answer and rag.log[-1]["sources"] == ""


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
    assert len(llm.calls) == n_calls + 1                               # hiányos eredménynél újra kérdez
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


def test_parse_names_wrong_types_fall_back():
    kw = {0: "agent, tool", 1: "speech, asr"}
    for bad in ('["nem szótár"]', '"csak szöveg"', '{"0": ["lista a szótár helyett"], "1": 5}',
                '{"0": {"nev": 42, "leiras": ["lista"]}, "1": {"nev": null}}', '{"0": {"nev": "  "}}'):
        out = tf.parse_names(bad, [0, 1], kw)                    # kivétel helyett tartalék név
        assert out[0] == {"nev": "agent", "leiras": ""} and out[1]["nev"] == "speech", bad
    ok = tf.parse_names('{"0": {"nev": "Ágensek", "leiras": 7}, "1": {"nev": "Beszéd", "leiras": "jó"}}', [0, 1], kw)
    assert ok[0] == {"nev": "Ágensek", "leiras": ""} and ok[1] == {"nev": "Beszéd", "leiras": "jó"}


def test_parse_names_accepts_variant_shapes():
    kw = {0: "agent, tool", 1: "speech, asr"}
    want = {0: {"nev": "Ágensek", "leiras": "Eszközt használó modellek."}, 1: {"nev": "Beszéd", "leiras": "ASR."}}
    for text in (
        '{"0": {"név": "Ágensek", "leírás": "Eszközt használó modellek."}, "1": {"Nev": "Beszéd", "Leiras": "ASR."}}',
        '{"0": {"name": "Ágensek", "description": "Eszközt használó modellek."}, "1": {"nev": "Beszéd", "leiras": "ASR."}}',
        '{"Téma 0": {"nev": "Ágensek", "leiras": "Eszközt használó modellek."}, "Téma 1": {"nev": "Beszéd", "leiras": "ASR."}}',
        '[{"tema": 0, "nev": "Ágensek", "leiras": "Eszközt használó modellek."}, {"téma": "1", "nev": "Beszéd", "leiras": "ASR."}]',
        '{"temak": {"0": {"nev": "Ágensek", "leiras": "Eszközt használó modellek."}, "1": {"nev": "Beszéd", "leiras": "ASR."}}}',
        '{"topics": [{"id": 0, "nev": "Ágensek", "leiras": "Eszközt használó modellek."}, {"id": 1, "nev": "Beszéd", "leiras": "ASR."}]}',
    ):
        assert tf.parse_names(text, [0, 1], kw) == want, text
    assert tf.parse_names('{"0": "Ágensek"}', [0, 1], kw)[0] == {"nev": "Ágensek", "leiras": ""}


def test_naming_retries_when_descriptions_missing(corpus, tmp_path):
    df, emb, _ = corpus
    df = df.copy()
    df["topic"], keywords = tf.find_topics(emb, (df.title + ". " + df.abstract).values, n_topics=2)
    ids = [0, 1]

    class LLM:
        def __init__(self, text):
            self.models, self.text, self.calls = self, text, 0

        def generate_content(self, model, contents, config=None):
            self.calls += 1
            return types.SimpleNamespace(text=self.text)
    bad = LLM('{"0": {"nev": "A"}, "1": {"nev": "B"}}')
    assert tf.name_topics(bad, "fake", df, keywords, ids, "cs.CL", str(tmp_path))[0]["leiras"] == ""
    assert (tmp_path / "valasz_temanevek.txt").read_text(encoding="utf-8") == bad.text
    good = LLM('{"0": {"nev": "A", "leiras": "a"}, "1": {"nev": "B", "leiras": "b"}}')
    assert tf.name_topics(good, "fake", df, keywords, ids, "cs.CL", str(tmp_path))[1]["leiras"] == "b"
    assert good.calls == 1                                              # a leírás nélküli tárolt nevek nem számítanak
    tf.name_topics(good, "fake", df, keywords, ids, "cs.CL", str(tmp_path))
    assert good.calls == 1                                              # a teljes eredmény már a Drive-ról jön


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


class ServerError(Exception):
    """A google-genai ServerError-jához hasonló: a HTTP-kód a .code-ban."""
    def __init__(self, code, msg):
        super().__init__(f"{code} {msg}")
        self.code = code


def _client(errors):
    calls = []
    def gc(model, contents, config=None):
        calls.append(1)
        if errors:
            raise errors.pop(0)
        return types.SimpleNamespace(text="ok")
    return types.SimpleNamespace(models=types.SimpleNamespace(generate_content=gc)), calls


def test_generate_retries_on_server_overload():
    waits = []
    client, calls = _client([ServerError(503, "UNAVAILABLE. This model is currently experiencing high demand."),
                             ServerError(500, "INTERNAL")])
    assert tf.generate(client, "m", "p", sleep=waits.append) == "ok" and len(calls) == 3
    assert waits == [5, 10]                                            # rövid, duplázódó várakozás
    client, calls = _client([ServerError(503, "UNAVAILABLE")] * 5)     # tartós túlterhelés: érthető hibaüzenet
    with pytest.raises(RuntimeError, match="túlterhelt.*5 próbálkozás|5 próbálkozás.*túlterhelt") as err:
        tf.generate(client, "m", "p", sleep=waits.append)
    assert len(calls) == 5 and isinstance(err.value.__cause__, ServerError)
    client, calls = _client([ServerError(400, "INVALID_ARGUMENT")])     # nem átmeneti: azonnal továbbmegy
    with pytest.raises(ServerError):
        tf.generate(client, "m", "p", sleep=waits.append)
    assert len(calls) == 1


def test_load_questions(tmp_path, capsys):
    p = tmp_path / "kerdesek.csv"
    p.write_text(
        "﻿vart_cikkek,keszitette,kerdes\n"
        "2512.10100; arXiv:2602.22144v2,SuleSz,Mi az a RAG, és hogyan csökkenti a hallucinációt?\n"
        '2610.36980 2605.30931,bandi137,"How are LLM agents evaluated, briefly?"\n'
        "\n"
        "2512.10100,SuleSz,\n"
        "csak egy oszlop\n"
        "9999.99999,SuleSz,Elírt azonosító\n",
        encoding="utf-8")
    qs = tf.load_questions(str(p), known_ids={"2512.10100", "2602.22144", "2610.36980", "2605.30931"})
    out = capsys.readouterr().out
    assert [q["q"] for q in qs] == ["Mi az a RAG, és hogyan csökkenti a hallucinációt?",
                                    "How are LLM agents evaluated, briefly?", "Elírt azonosító"]
    assert qs[0]["expected"] == ["2512.10100", "2602.22144"]          # szövegként, nem 2512.101
    assert qs[1]["expected"] == ["2610.36980", "2605.30931"] and qs[1]["keszitette"] == "bandi137"
    assert "5. sor kihagyva" in out and "6. sor kihagyva" in out and "9999.99999" in out


def test_shipped_questions_file_loads():
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "eval" / "kerdesek.csv"
    assert tf.load_questions(str(path)) is not None


def test_answer_language_follows_question():
    import pandas as pd
    hits = pd.DataFrame([dict(id="2501.00001", title="T", month="2025-11", abstract="a", score=0.9)])
    for q, lang in [("What are recent approaches to evaluating LLM agents?", "en"),
                    ("How is hallucination reduced in vision-language models?", "en"),
                    ("Is it possible to translate low-resource languages?", "en"),
                    ("Milyen módszereket javasolnak a hallucináció csökkentésére?", "hu"),
                    ("Hogyan csokkentik a hallucinaciokat a nyelvi modellek", "hu"),
                    ("Mi a RAG?", "hu")]:
        assert tf.detect_language(q) == lang, q
        p = tf.build_prompt(q, hits)
        assert ("Answer language: English" in p) == (lang == "en")
        assert ("A válasz nyelve: magyar" in p) == (lang == "hu")
        assert "{" not in p.split("KÉRDÉS:")[1]


def test_answer_language_can_be_set_by_hand():
    import pandas as pd
    hits = pd.DataFrame([dict(id="2501.00001", title="T", month="2025-11", abstract="a", score=0.9)])
    q = "How does MI research use van der Waals forces?"          # a felismerés itt magyarnak veheti
    assert "Answer language: English" in tf.build_prompt(q, hits, lang="en")
    assert "A válasz nyelve: magyar" in tf.build_prompt("What is RAG?", hits, lang="hu")
    assert tf.resolve_language(q, "auto") == tf.detect_language(q) and tf.resolve_language(q, "en") == "en"


def test_uncited_answer_flagged(corpus):
    df, emb, q = corpus
    hits = tf.retrieve(df, emb, q("agents"), k=3)
    assert tf.check_citations("Nincs hivatkozás a válaszban.", hits)["uncited"]
    assert not tf.check_citations(f"Lásd [{hits.id.iloc[0]}].", hits)["uncited"]
    assert tf.check_citations("[2601.99999]", hits)["uncited"]          # csak kitalált azonosító: az sem forrás
    assert not tf.check_citations("x", hits.iloc[:0])["uncited"]        # nincs forrás: nincs mit számon kérni
