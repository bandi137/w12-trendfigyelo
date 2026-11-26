"""Visszakeresés kiértékelése: hit@k egy kézzel összeállított kérdéslistán (eval/kerdesek.csv)."""
import csv
import re

import pandas as pd

from .rag import detect_language

ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}")


def load_questions(path, known_ids=None):
    """Beolvassa az eval/kerdesek.csv-t: vart_cikkek, keszitette, kerdes (a kérdés az utolsó oszlop,
    így idézőjel nélküli vessző is lehet benne). Az azonosítók szövegként maradnak.
    Ha `known_ids` meg van adva, figyelmeztet az adatok között nem szereplő azonosítókra; ha egy kérdés egyik várt
    cikke sincs az adatban (pl. az időablakon kívül esik), a kérdést kihagyja, mert a keresés nem is találhatná meg."""
    questions, skipped = [], 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    for line_no, row in enumerate(rows[1:], start=2):           # 1. sor: fejléc
        if not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
            continue
        if len(row) < 3:
            print(f"⚠ {line_no}. sor kihagyva: 3 oszlop kell (vart_cikkek,keszitette,kerdes): {row}")
            continue
        expected = ARXIV_ID.findall(row[0])
        question = ",".join(row[2:]).strip()
        if not expected or not question:
            print(f"⚠ {line_no}. sor kihagyva: hiányzik a kérdés vagy a várt cikk azonosítója.")
            continue
        if known_ids is not None:
            unknown = [i for i in expected if i not in known_ids]
            if unknown and len(unknown) == len(expected):
                print(f"⚠ {line_no}. sor kihagyva: a várt cikk nincs az adatok között (elírás, vagy kívül esik a letöltött "
                      f"hónapokon?): {unknown}")
                skipped += 1
                continue
            if unknown:
                print(f"⚠ {line_no}. sor: ezek az azonosítók nincsenek az adatok között, a többivel értékelem: {unknown}")
                expected = [i for i in expected if i not in unknown]
        questions.append({"q": question, "expected": expected, "keszitette": row[1].strip()})
    print(len(questions), "tesztkérdés betöltve:", path, f"({skipped} kihagyva, lásd fent)" if skipped else "")
    return questions


def hit_at_k(results, expected, k):
    """1, ha a várt cikkek közül legalább egy benne van az első k találatban."""
    return int(bool(set(results[:k]) & set(expected)))


def first_rank(results, expected):
    """A várt cikkek közül az első helyezése a találatok között (1-től), vagy None, ha egyik sincs köztük."""
    want = set(expected)
    return next((i for i, r in enumerate(results, 1) if r in want), None)


def evaluate(questions, retrieve_fn):
    """questions: [{"q": ..., "expected": [arXiv-azonosítók]}]; retrieve_fn(kérdés, k) -> találatok DataFrame-je.
    Kérdésenként: hit@1/5/10, a jó cikk helyezése az első 10 találat között (helyezés; üres, ha nincs köztük),
    a kérdés nyelve és a várt cikk(ek)."""
    rows = []
    for q in questions:
        ids = retrieve_fn(q["q"], 10).id.tolist()
        rows.append({"kérdés": q["q"], "készítette": q.get("keszitette", ""), "hit@1": hit_at_k(ids, q["expected"], 1),
                     "hit@5": hit_at_k(ids, q["expected"], 5), "hit@10": hit_at_k(ids, q["expected"], 10),
                     "helyezés": first_rank(ids, q["expected"]), "nyelv_kod": detect_language(q["q"]),
                     "várt cikk": ", ".join(q["expected"])})
    res = pd.DataFrame(rows)
    res["helyezés"] = res["helyezés"].astype("Int64")
    return res, res[["hit@1", "hit@5", "hit@10"]].mean().round(2)
