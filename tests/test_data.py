"""A hónaponkénti arXiv-tárolás és a vektor-gyorsítótár tesztjei (hálózat nélkül, szimulált letöltővel)."""
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

import trendfigyelo as tf

TODAY = date(2026, 11, 14)          # lezárt hónapok: 2026-08, 2026-09, 2026-10


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def __call__(self, category, start, end, label, per_month):
        self.calls.append(label)
        return pd.DataFrame([{"id": f"{label[2:4]}{label[5:7]}.{i:05d}", "title": f"T {label} {i}",
                              "abstract": "a", "published": label + "-10", "month": label,
                              "authors": "X", "url": "u"} for i in range(per_month)])


def load(d, dl, refresh="auto", today=TODAY, per_month=3):
    return tf.load_or_download(str(d), "cs.CL", 3, per_month, refresh=refresh, today=today, downloader=dl)


def test_months_downloaded_once_then_from_drive(tmp_path):
    dl = FakeDownloader()
    df = load(tmp_path, dl)
    assert dl.calls == ["2026-08", "2026-09", "2026-10"] and len(df) == 9
    assert sorted(os.listdir(tmp_path / "honapok")) == ["2026-08.csv", "2026-09.csv", "2026-10.csv"]
    meta = json.load(open(tmp_path / "letoltes.json"))
    assert meta["2026-10"] == dict(letoltve="2026-11-14", cikkek=3, per_month=3, category="cs.CL", mintavetel="napi")

    dl.calls.clear()
    df2 = load(tmp_path, dl)                                  # ugyanaz a nap: semmi letöltés
    assert dl.calls == [] and df2.id.tolist() == df.id.tolist()

    df3 = load(tmp_path, dl, today=date(2026, 12, 1))          # új hónap: csak az töltődik le
    assert dl.calls == ["2026-11"] and sorted(df3.month.unique()) == ["2026-09", "2026-10", "2026-11"]
    assert tf.download_dates(str(tmp_path), ["2026-09", "2026-11"]) == "2026-11-14 – 2026-12-01"
    assert tf.download_dates(str(tmp_path), ["2026-09"]) == "2026-11-14"


def test_refresh_modes_and_setting_change(tmp_path):
    dl = FakeDownloader()
    load(tmp_path, dl)
    dl.calls.clear()
    load(tmp_path, dl, refresh="mindig")
    assert dl.calls == ["2026-08", "2026-09", "2026-10"]
    dl.calls.clear()
    df = load(tmp_path, dl, refresh="soha", today=date(2027, 3, 1))   # az ablak továbblépett, de nem tölt
    assert dl.calls == [] and df.month.max() == "2026-10"
    load(tmp_path, dl, per_month=5)                                    # más PER_MONTH: újra kell tölteni
    assert dl.calls == ["2026-08", "2026-09", "2026-10"]
    with pytest.raises(ValueError):
        load(tmp_path, dl, refresh="néha")


def test_old_sampling_months_are_redownloaded(tmp_path):
    """A régi („hónap vége”) mintavétellel letöltött hónapok – a régi egyben tárolt papers.csv-ből bontottak is –
    egyszer újratöltődnek napi bontással; utána a Drive-ról jönnek."""
    old = pd.concat([FakeDownloader()("cs.CL", "", "", m, 3) for m in ("2026-08", "2026-09", "2026-10")])
    old.to_csv(tmp_path / "papers.csv", index=False)
    dl = FakeDownloader()
    load(tmp_path, dl)
    assert dl.calls == ["2026-08", "2026-09", "2026-10"]
    meta = json.load(open(tmp_path / "letoltes.json"))
    assert {m["mintavetel"] for m in meta.values()} == {"napi"}
    assert tf.sampling_of(str(tmp_path), ["2026-08", "2026-10"]) == "napi"
    meta["2026-08"].pop("mintavetel")
    (tmp_path / "letoltes.json").write_text(json.dumps(meta))
    assert tf.sampling_of(str(tmp_path), ["2026-08", "2026-10"]) == "honap_vege"
    dl.calls.clear()
    load(tmp_path, dl)
    assert dl.calls == ["2026-08"]


def test_download_month_queries_every_day_with_quota():
    calls = []

    def fetch(query, max_results):
        calls.append((query, max_results))
        day = query.split("[")[1][:8]
        n = 1 if day.endswith("06") else max_results + 5            # egy napon kevés cikk, máskor sok
        return [{"id": f"{day[2:6]}.{int(day[6:8]) * 100 + i:05d}", "title": f"T{i}", "abstract": "a", "published": day,
                 "authors": "X", "url": "u"} for i in range(min(n, max_results))] + \
               ([{"id": f"{day[2:6]}.00100", "title": "dupla", "abstract": "a", "published": day,
                  "authors": "X", "url": "u"}] if day.endswith("01") else [])
    assert len(tf.day_ranges("202602010000", "202603010000")) == 28
    assert tf.per_day_quota(3000, 30) == 100 and tf.per_day_quota(400, 31) == 13
    df = tf.download_month("cs.CL", "202511010000", "202512010000", "2025-11", 300, fetch=fetch)
    assert len(calls) == 30 and all(m == 10 for _, m in calls)                  # 30 nap, naponta 10
    assert calls[0][0] == "cat:cs.CL AND submittedDate:[202511010000 TO 202511020000]"
    assert calls[-1][0].endswith("[202511300000 TO 202512010000]")
    assert len(df) == 29 * 10 + 1 and df.id.is_unique and (df.month == "2025-11").all()   # kevés napon mind; dupla nélkül
    assert list(df.columns) == ["id", "title", "abstract", "published", "month", "authors", "url"]


class FakeEmbedder:
    """Determinisztikus „vektor” a szövegből; számolja, hány szöveget vektorizált."""
    def __init__(self, name="e5"):
        self.model_name, self.n, self.texts = name, 0, 0

    def passages(self, texts):
        self.n += 1
        self.texts += len(texts)
        return np.array([[len(t), sum(map(ord, t)) % 97, 1.0, 0.0] for t in texts])


def test_embeddings_only_new_or_changed_articles_recomputed(tmp_path):
    df = FakeDownloader()("cs.CL", "", "", "2026-10", 5)
    e = FakeEmbedder()
    first = tf.load_or_embed(df, e, str(tmp_path))
    assert e.texts == 5
    tf.load_or_embed(df, e, str(tmp_path))
    assert e.n == 1                                            # semmi sem változott: betölti
    changed = df.copy()
    changed.loc[changed.index[2], "abstract"] = "Teljesen más absztrakt, ugyanazzal az azonosítóval."
    out = tf.load_or_embed(changed, e, str(tmp_path))
    assert e.texts == 6                                        # csak a módosult absztraktú cikk számolódik újra
    assert not np.allclose(out[2], first[2]) and np.allclose(np.delete(out, 2, 0), np.delete(first, 2, 0))
    sub = tf.load_or_embed(changed.iloc[1:], e, str(tmp_path))
    assert e.texts == 6 and np.allclose(sub, out[1:])          # kevesebb cikk (pl. kilépő hónap): nincs vektorizálás
    e2 = FakeEmbedder("masik-modell")
    assert len(tf.load_or_embed(df.iloc[1:], e2, str(tmp_path))) == 4 and e2.texts == 4   # más modell: mind újra


def test_embeddings_old_format_cache_recomputed_once(tmp_path):
    import json
    df = FakeDownloader()("cs.CL", "", "", "2026-10", 3)
    np.save(tmp_path / "embeddings.npy", np.zeros((3, 4)))     # régi formátum: csak azonosító-ellenőrzőösszeg
    (tmp_path / "embeddings.json").write_text(json.dumps(dict(model="e5", n=3, ids_sha1="x")))
    e = FakeEmbedder()
    tf.load_or_embed(df, e, str(tmp_path))
    tf.load_or_embed(df, e, str(tmp_path))
    assert e.texts == 3 and e.n == 1


def test_report_states_monthly_sample():
    table = pd.DataFrame(columns=["név", "leírás", "kulcsszavak", "cikkek", "növekedés"])
    meta = dict(category="cs.CL", months="2026-08 – 2026-10", n_papers=9, per_month=400)
    assert "legfeljebb 400 cikk" in tf.build_report(meta, table)
    assert "legfeljebb" not in tf.build_report({k: v for k, v in meta.items() if k != "per_month"}, table)
    daily = tf.build_report({**meta, "per_month": 3000, "sampling": "napi"}, table)
    assert "legfeljebb 3000 cikk szerepel, a hónap minden napjáról" in daily and "utolsó napjain" not in daily
    assert "0,5 százalékpontot adunk (simítás)" in daily and "1 %-ról 10 %-ra nőve 7×, nem 10×" in daily
    assert "utolsó napjain" in tf.build_report({**meta, "sampling": "honap_vege"}, table)


def test_growth_note_matches_formula():
    import pandas as pd
    months = ["2026-01"] * 100 + ["2026-02"] * 100
    labels = [0] * 1 + [1] * 99 + [0] * 10 + [1] * 90           # a 0. téma: 1 % → 10 %
    _, growth = tf.topic_trends(pd.Series(months), pd.Series(labels), window=1)
    assert round(float(growth[0]), 2) == 7.0                    # amit a leírás példája állít
    assert "7×, nem 10×" in tf.trends.growth_note(1) and "+0,5 sz.pont" in tf.trends.growth_note(3, short=True)


def test_per_month_limit_from_meta(tmp_path):
    assert tf.per_month_limit(str(tmp_path), ["2026-10"]) is None
    load(tmp_path, FakeDownloader(), per_month=4)
    assert tf.per_month_limit(str(tmp_path), ["2026-08", "2026-10"]) == 4


def test_report_map_text_uses_actual_vector_size():
    import plotly.graph_objects as go
    table = pd.DataFrame(columns=["név", "leírás", "kulcsszavak", "cikkek", "növekedés"])
    meta = dict(category="cs.CL", months="2026-08 – 2026-10", n_papers=9, emb_dim=384)
    html = tf.build_report(meta, table, map_fig=go.Figure())
    assert "a 384 dimenziós vektorokat" in html and "768" not in html
    html = tf.build_report({k: v for k, v in meta.items() if k != "emb_dim"}, table, map_fig=go.Figure())
    assert "a UMAP a vektorokat" in html


def test_report_shows_download_date():
    table = pd.DataFrame(columns=["név", "leírás", "kulcsszavak", "cikkek", "növekedés"])
    html = tf.build_report(dict(category="cs.CL", months="2026-08 – 2026-10", n_papers=9,
                                downloaded="2026-11-14"), table)
    assert "adat letöltve: 2026-11-14" in html
