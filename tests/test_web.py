"""A webes felület, a HTML-jelentés és a közzététel tesztjei (hálózat és Colab nélkül)."""
import re
import types

import numpy as np
import pytest

import trendfigyelo as tf
from test_core import FakeLLM, corpus  # noqa: F401  (pytest-fixture)


def test_md_lite_is_safe_and_links_arxiv():
    out = tf.md_lite("<script>alert(1)</script> **fontos** és *dőlt* [2512.19070]\n\n* első\n* második")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<strong>fontos</strong>" in out and "<em>dőlt</em>" in out
    assert 'href="https://arxiv.org/abs/2512.19070"' in out
    assert out.count("<li>") == 2 and "<ul>" in out


@pytest.fixture()
def app(corpus, tmp_path):  # noqa: F811
    df, emb, q = corpus
    df = df.copy()
    df.loc[0, "title"] = '<img src=x onerror="alert(1)">'
    llm = FakeLLM()
    rag = tf.RAG(df, emb, q, llm, "fake", top_k=3)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "kerdesek.csv").write_text(
        "vart_cikkek,keszitette,kerdes\n" + df.id.iloc[5] + ",SuleSz,speech recognition, audio\n", encoding="utf-8")
    return tf.TrendApp(df, emb, types.SimpleNamespace(query=q), llm, rag, data_dir=str(tmp_path),
                       settings=dict(category="cs.CL", months=12, emb_model="e5", llm_model="fake",
                                     n_topics=4, n_show=4, top_k=3),
                       repo_dir=str(tmp_path), token_fn=lambda: None)


def test_app_ask_links_sources(app):
    md = app.ask("speech recognition audio", k=3)
    assert md.count("https://arxiv.org/abs/") >= 4          # a válaszban és a forráslistában
    assert "⚠" in md and "2601.99999" in md                 # kitalált hivatkozás jelzése
    assert app.ask("   ") == "Írj be egy kérdést."


def test_app_report_contains_everything_and_escapes(app):
    app.ask("speech recognition audio")
    app.run_eval()
    html = app.build_html(archive=["2026-11-12", "2026-11-13"])
    for part in ('id="trendek"', 'id="terkep"', 'href="#terkep"', "plotly_click", "lejátszás", 'id="osszefoglalo"', 'id="kerdesek"', 'id="kiertekeles"', 'id="archiv"',
                 'href="archiv/2026-11-13.html"', "Süle Szilvia", "cdn.plot.ly"):
        assert part in html, part
    assert "<img src=x" not in html                        # a cikkcím nem futtatható HTML-ként (a térképen sem)
    assert html.count("cdn.plot.ly") == 1                   # a Plotly egyszer töltődik be a két grafikonhoz
    assert 'href="2026-11-12.html"' in app.build_html(archive=["2026-11-12"], archive_prefix="")
    assert app.preview().endswith("jelentes_elonezet.html")


def test_app_publish_requires_token(app):
    with pytest.raises(tf.PublishError, match="GITHUB_TOKEN"):
        app.publish()


def test_app_publish_sends_index_archive_and_nojekyll(app, monkeypatch):
    sent = {}
    monkeypatch.setattr("trendfigyelo.app.list_archive", lambda token, repo: ["2026-11-01"])
    monkeypatch.setattr("trendfigyelo.app.publish_files",
                        lambda files, token, repo, message: sent.update(files=files, token=token, message=message)
                        or {"sha": "abc1234def", "commit_url": "u", "pages_url": tf.pages_url(repo)})
    app.token_fn = lambda: "tok"
    res = app.publish()
    paths = sorted(sent["files"])
    assert paths[0] == "docs/.nojekyll" and paths[1].startswith("docs/archiv/20") and paths[2] == "docs/index.html"
    assert 'href="archiv/2026-11-01.html"' in sent["files"]["docs/index.html"]
    assert res["pages_url"] == "https://bandi137.github.io/w12-trendfigyelo/"


class FakeGitHub:
    """A GitHub Git Data API-ját utánozza; az első ref-frissítésnél opcionálisan ütközést (422) jelez."""
    def __init__(self, conflict_once=False):
        self.headers, self.calls, self.conflict = {}, [], conflict_once

    def _r(self, code, data=None):
        return types.SimpleNamespace(status_code=code, json=lambda: data or {}, content=b"x", text=str(data))

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url))
        if url.endswith("/git/ref/heads/main"):
            return self._r(200, {"object": {"sha": f"head{len(self.calls)}"}})
        if "/git/commits/" in url:
            return self._r(200, {"tree": {"sha": "tree0"}})
        if url.endswith("/contents/docs/archiv"):
            return self._r(200, [{"name": "2026-11-12.html"}, {"name": "README.md"}])
        return self._r(404)

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._r(201, {"sha": "tree1" if url.endswith("/trees") else "commit1"})

    def patch(self, url, json=None, timeout=None):
        self.calls.append(("PATCH", url, json))
        if self.conflict:
            self.conflict = False
            return self._r(422, {"message": "Update is not a fast forward"})
        return self._r(200, {})


def test_publish_files_single_commit_and_retry_on_conflict():
    gh = FakeGitHub(conflict_once=True)
    res = tf.publish_files({"docs/index.html": "<html>", "docs/.nojekyll": ""}, "tok", "bandi137/w12-trendfigyelo",
                           session=gh)
    assert gh.headers["Authorization"] == "Bearer tok"
    trees = [c for c in gh.calls if c[0] == "POST" and c[1].endswith("/trees")]
    assert len(trees) == 2                                       # ütközés után újrakezdte
    assert {e["path"] for e in trees[0][2]["tree"]} == {"docs/index.html", "docs/.nojekyll"}
    patches = [c for c in gh.calls if c[0] == "PATCH"]
    assert patches[-1][2] == {"sha": "commit1", "force": False}
    assert res["commit_url"].endswith("/commit/commit1")


def test_list_archive_and_errors():
    assert tf.list_archive("tok", "a/b", session=FakeGitHub()) == ["2026-11-12"]

    class Denied(FakeGitHub):
        def get(self, url, params=None, timeout=None):
            return self._r(403, {"message": "Resource not accessible"})
    with pytest.raises(tf.PublishError, match="írási joga"):
        tf.publish_files({"docs/index.html": "x"}, "tok", "a/b", session=Denied())


def test_ui_builds(app):
    pytest.importorskip("gradio")
    ui = app.build_ui()
    assert ui is not None


def test_figure_follows_n_show(app):
    app.compute_trends(4)
    lines = lambda fig: [t for t in fig.data if t.mode == "lines"]     # a nyers havi pontok külön nyomvonalak
    assert len(lines(app.figure())) == 4                 # n_show = 4
    assert len(lines(app.figure(3))) == 3
    app.s["n_show"] = 2
    assert len(lines(app.figure())) == 2
    assert app.build_html().count("Téma ") >= 2          # a jelentés is ugyanennyi témát használ


def test_spread_labels_no_overlap_and_names_short():
    from trendfigyelo.report import short_name, spread_labels
    out = spread_labels([10.0, 10.1, 10.2, 3.0], gap=1.0, top=11.0)
    assert out[3] == 3.0 and sorted(out)[-1] <= 11.0
    s = sorted(out)
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(s, s[1:]))
    assert short_name("Tudásdesztilláció (Knowledge distillation)", 34) == "Tudásdesztilláció"
    assert len(short_name("x" * 80)) <= 34


def test_figure_highlights_top_three_and_labels(app):
    app.compute_trends(4)
    fig = app.figure(4)
    assert len(fig.layout.annotations) >= 4 + 2          # 4 felirat + 2 sáv
    assert fig.layout.showlegend is False


def test_launch_requires_password_when_secret_set(app, monkeypatch):
    seen = {}
    class FakeUI:
        def launch(self, **kw): seen.update(kw); return "ok"
    monkeypatch.setattr(app, "build_ui", lambda: FakeUI())
    app.launch(share=False, password="titok")
    assert seen["auth"] == ("w12", "titok") and seen["share"] is False
    seen.clear()
    monkeypatch.setattr("trendfigyelo.app.colab_secret", lambda name: None)
    app.launch(share=False)                                     # helyi indítás jelszó nélkül is mehet
    assert "auth" not in seen
    seen.clear()
    with pytest.raises(RuntimeError, match="UI_PASSWORD"):      # nyilvános link jelszó nélkül: nem indul
        app.launch(share=True)
    assert not seen


def test_map_coords_cached_and_recomputed(corpus, tmp_path):  # noqa: F811
    df, emb, _ = corpus
    xy = tf.map_coords(emb, df.id, str(tmp_path), method="tsne")
    assert xy.shape == (len(df), 2) and abs(xy.mean()) < 1e-6
    assert (tmp_path / "terkep.npy").exists()
    again = tf.map_coords(emb, df.id, str(tmp_path), method="tsne")          # a Drive-ról töltődik
    assert np.array_equal(xy, again)
    np.save(tmp_path / "terkep.npy", np.zeros_like(xy))                      # ha a cikklista változik: újraszámol
    other = tf.map_coords(emb[:-1], df.id[:-1], str(tmp_path), method="tsne")
    assert other.shape == (len(df) - 1, 2) and other.any()
    labels = df.true.astype("category").cat.codes.to_numpy()                 # a témák nem számítanak: betöltés
    assert np.array_equal(tf.map_coords(emb[:-1], df.id[:-1], str(tmp_path), labels=labels[:-1], method="tsne"), other)


def test_display_sample_keeps_monthly_shares():
    from trendfigyelo.tmap import display_sample
    months = np.repeat(["2025-01", "2025-02", "2025-03"], [3000, 6000, 1000])
    idx = display_sample(months, 2000, seed=1)
    assert abs(len(idx) - 2000) <= 3 and np.all(np.diff(idx) > 0)              # rendezett, ismétlés nélkül
    counts = {m: int((months[idx] == m).sum()) for m in ["2025-01", "2025-02", "2025-03"]}
    assert counts == {"2025-01": 600, "2025-02": 1200, "2025-03": 200}        # a havi arány megmarad
    assert np.array_equal(idx, display_sample(months, 2000, seed=1))          # determinisztikus
    assert len(display_sample(months, None)) == len(months) == len(display_sample(months, 20000))


def test_topic_map_samples_points_but_keeps_sources(corpus):  # noqa: F811
    df, emb, q = corpus
    df = df.copy()
    df["topic"], _ = tf.find_topics(emb, (df.title + ". " + df.abstract).values, n_topics=4)
    _, growth = tf.topic_trends(df.month, df.topic)
    names = {int(t): {"nev": f"Téma {t}", "leiras": ""} for t in growth.index}
    xy = np.random.default_rng(0).normal(size=(len(df), 2))
    full = tf.topic_map(xy, df, names, growth, "cs.CL")
    small = tf.topic_map(xy, df, names, growth, "cs.CL", max_points=len(df) // 3)
    shown = sum(len(tr.x) for tr in small.data)
    assert shown < len(df) and f"{len(df)} cikk (a képen {shown} pont" in small.layout.title.text
    assert "a képen" not in full.layout.title.text
    assert [a.text for a in small.layout.annotations] == [a.text for a in full.layout.annotations]  # nevek: mind
    rag = tf.RAG(df, emb, q, llm=None, model="fake")
    qmap = tf.map_question(rag, {"question": "agents", "sources": ", ".join(df.id[-3:])})
    with_q = tf.topic_map(xy, df, names, growth, "cs.CL", max_points=5, questions=[qmap])
    forras = next(tr for tr in with_q.data if tr.name == "q-forras")
    assert len(forras.x) == 3                                                 # a források a mintától függetlenül


def test_pull_blurs_topics_gradually():
    from trendfigyelo.tmap import _pull
    rng = np.random.default_rng(0)
    xy, topics = rng.normal(size=(400, 2)), rng.integers(0, 4, 400)           # teljesen kevert felhő
    spread = lambda c: np.mean([c[topics == t].std(axis=0).mean() for t in range(4)])  # noqa: E731
    s0, s3, s6 = spread(_pull(xy, topics, 0)), spread(_pull(xy, topics, 0.3)), spread(_pull(xy, topics, 0.6))
    assert np.allclose(_pull(xy, topics, 0), xy) and s0 > s3 > s6            # fokozatos: 0 = változatlan
    assert np.isclose(s6 / s0, 0.4, atol=0.02)                                # a távolság 60 %-ával húz befelé


def test_topic_map_layers_colors_labels(corpus):  # noqa: F811
    df, emb, _ = corpus
    df = df.copy()
    df["topic"], _ = tf.find_topics(emb, (df.title + ". " + df.abstract).values, n_topics=4)
    share, growth = tf.topic_trends(df.month, df.topic)
    names = {int(t): {"nev": f"<b>Téma {t}</b> (Topic {t})", "leiras": ""} for t in growth.index}
    xy = np.random.default_rng(0).normal(size=(len(df), 2))
    fig = tf.topic_map(xy, df, names, growth, "cs.CL")
    months = sorted(df.month.unique())
    assert [tr.name for tr in fig.data] == months                              # hónaponként egy réteg
    assert sum(len(tr.x) for tr in fig.data) == len(df)
    steps = fig.layout.sliders[0].steps                                        # "mind" + minden hónap
    assert [st.label for st in steps] == ["mind"] + months and not fig.frames   # restyle, nem animáció (Plotly 2.x)
    assert all(st.method == "restyle" and len(st.args[0]["marker.opacity"]) == len(months) for st in steps)
    assert not fig.layout.updatemenus                                         # a lejátszás gomb csak a jelentésben
    assert tf.topic_map(xy, df, names, growth, "cs.CL", play_button=True).layout.updatemenus[0].buttons[0].method == "skip"
    assert len(fig.layout.annotations) == len(growth)                         # minden témának van neve
    assert sum(a.name == "kiemelt" for a in fig.layout.annotations) == 3
    assert all("<b>Téma" not in a.text for a in fig.layout.annotations)       # a név escape-elve
    top = int(growth.index[0])                                                # a leggyorsabban növő téma piros
    i = int(np.flatnonzero(df.topic.to_numpy() == top)[0])
    tr = next(t for t in fig.data if t.name == df.month.iloc[i])
    assert max(tr.marker.color) > 0 and tr.marker.cmin == -tr.marker.cmax


def test_spread_islands_separates_and_keeps_layout():
    from trendfigyelo.tmap import _landscape, _spread_islands
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [0.3, 0.1], [3, 0], [3.2, 2.5], [-2, 2]], dtype=float)   # az első kettő egymásra lóg
    topics = np.repeat(np.arange(5), 60)
    xy = centers[topics] + rng.normal(scale=0.3, size=(300, 2))
    out = _spread_islands(xy, topics)
    c = np.array([np.median(out[topics == t], axis=0) for t in range(5)])
    r = np.array([np.percentile(np.linalg.norm(out[topics == t] - c[t], axis=1), 75) for t in range(5)])
    d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(5) * 99
    assert (d >= 0.8 * (r[:, None] + r[None]) * 0.999).all()                # legfeljebb kis átfedés (sep=0,8)
    for t in range(5):                                                       # a sziget alakja nem változik
        assert np.allclose(out[topics == t] - out[topics == t].mean(0), xy[topics == t] - xy[topics == t].mean(0))
    assert np.argmax(c[:, 0]) in (2, 3) and np.argmin(c[:, 0]) == 4         # a szigetek sorrendje megmarad
    rot = _landscape(np.c_[rng.normal(size=200) * 0.2, rng.normal(size=200) * 3])
    assert np.ptp(rot[:, 0]) > np.ptp(rot[:, 1])                            # fekvő tájolás


def test_map_click_regex_accepts_real_arxiv_ids(app):
    """A térkép kattintásos hivatkozása: a jelentésbe kerülő JavaScript-minta valódi azonosítót elfogad.
    (A korábbi hiba: a mintában duplán escape-elt \\d miatt minden azonosítót elutasított.)"""
    import re
    import shutil
    import subprocess
    html = app.build_html()
    m = re.search(r"&& (/\^.*?\$/)\.test\(p\.customdata\[4\]\)", html)
    assert m, "a kattintás-kezelő mintája nincs a jelentésben"
    js_regex = m.group(1)
    pattern = js_regex[1:-1]
    for good in ("2601.09725", "2512.1907"):
        assert re.search(pattern, good), (pattern, good)
    for bad in ("2601.097", "x2601.09725", "\\d{4}.12345"):
        assert not re.search(pattern, bad), (pattern, bad)
    if shutil.which("node"):                                 # ha van Node.js: a böngésző motorjával is
        out = subprocess.run(["node", "-e", f"console.log({js_regex}.test('2601.09725'), {js_regex}.test('260.09725'))"],
                             capture_output=True, text=True, timeout=30)
        assert out.stdout.split() == ["true", "false"], out.stdout + out.stderr


def test_trend_chart_switches_to_legend_on_narrow_screens(app):
    app.compute_trends()
    fig = app.figure()
    ends = [a for a in fig.layout.annotations if a.name == "veg"]
    assert len(ends) == app.s["n_show"] or len(ends) == len([t for t in fig.data if t.mode == "lines"])
    assert fig.layout.showlegend is False                                   # asztalon: nevek a vonalak végén
    assert all(t.showlegend for t in fig.data if t.mode == "lines")         # keskenyen: jelmagyarázat (a szkript kapcsolja)
    html = app.build_html()
    assert "width < 640" in html and "'veg'" in html and "showlegend: n" in html


def test_map_coords_notice_change_beyond_first_components(corpus, tmp_path):  # noqa: F811
    df, emb, _ = corpus
    xy = tf.map_coords(emb, df.id, str(tmp_path), method="tsne")
    changed = np.array(emb, dtype=np.float32).copy()
    changed[0, -1] += 0.5                                      # az utolsó komponens változik (régen nem számított)
    np.save(tmp_path / "terkep.npy", np.zeros_like(xy))
    assert tf.map_coords(changed, df.id, str(tmp_path), method="tsne").any()   # újraszámolta, nem a nullákat töltötte be


def test_app_ask_language_and_uncited_warning(app):
    class Echo:
        models = None
        def __init__(self): self.models = self; self.prompts = []
        def generate_content(self, model, contents, config=None):
            self.prompts.append(contents)
            return types.SimpleNamespace(text="Általános válasz hivatkozás nélkül.")
    llm = Echo()
    app.rag.llm = llm
    out = app.ask("Milyen ágensek vannak?", lang="en")
    assert "Answer language: English" in llm.prompts[-1]
    assert "egyetlen átadott forrásra sem hivatkozik" in out and app.rag.log[-1]["lang"] == "en"


def test_question_position_follows_most_similar_papers():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [-5.0, 3.0]])
    scores = np.array([0.1, 0.9, 0.89, 0.2])
    pos = tf.question_position(xy, scores, k=2)
    assert 9.9 < pos[0] <= 10 and 0 < pos[1] < 1                       # a két leghasonlóbb cikk közé esik
    assert np.allclose(tf.question_position(xy, scores, k=1), xy[1])


def test_topic_map_questions_layers_menu_and_slider(corpus):  # noqa: F811
    df, emb, q = corpus
    df = df.copy()
    df["topic"], _ = tf.find_topics(emb, (df.title + ". " + df.abstract).values, n_topics=4)
    share, growth = tf.topic_trends(df.month, df.topic)
    names = {int(t): {"nev": f"Téma {t}", "leiras": ""} for t in growth.index}
    xy = np.random.default_rng(0).normal(size=(len(df), 2))
    rag = tf.RAG(df, emb, q, llm=None, model="fake")
    qs = [tf.map_question(rag, {"question": "agents <b>tool</b> use", "sources": ", ".join(df.id[:3])}),
          tf.map_question(rag, "speech recognition")]
    assert len(qs[1]["sources"]) == rag.top_k and qs[0]["scores"].shape == (len(df),)
    fig = tf.topic_map(xy, df, names, growth, "cs.CL", questions=qs, play_button=True)
    months = sorted(df.month.unique())
    names_ = [t.name for t in fig.data]
    assert names_[:len(months)] == months and names_[len(months):] == ["q-vonal", "q-forras", "q-csillag"] * 2
    assert all(t.type == "scatter" for t in fig.data[len(months):])               # SVG-réteg: mindig a pontok fölött
    assert [t.visible for t in fig.data[len(months):]] == [True] * 3 + [False] * 3   # az első kérdés látszik
    star = fig.data[len(months) + 2]
    assert "<b>" not in star.customdata[0][0] and "★" in fig.layout.title.text      # escape-elt kérdés; jelmagyarázat
    assert len(fig.data[len(months) + 1].x) == 3                                    # három bekarikázott forrás
    cimkek = [a for a in fig.layout.annotations if a.name == "q-cimke"]
    assert [a.text for a in cimkek] == ["<b>1</b>", "<b>2</b>", "<b>3</b>"] + [f"<b>{n}</b>" for n in range(1, rag.top_k + 1)]
    assert [a.visible for a in cimkek] == [True] * 3 + [False] * rag.top_k         # csak a látható kérdés címkéi
    menus = {m.type: m for m in fig.layout.updatemenus}
    assert fig.layout.updatemenus[0].type == "buttons"                              # a lejátszás az első (a JS erre épít)
    assert len(menus["dropdown"].buttons) == 3                                      # 2 kérdés + „egyik sem”
    second = menus["dropdown"].buttons[1].args
    assert second[0]["visible"][-3:] == [True] * 3 and menus["dropdown"].buttons[1].method == "update"
    idx = [i for i, a in enumerate(fig.layout.annotations) if a.name == "q-cimke"]
    assert [second[1][f"annotations[{i}].visible"] for i in idx] == [False] * 3 + [True] * rag.top_k
    last = tf.topic_map(xy, df, names, growth, "cs.CL", questions=qs, active_question="last")
    assert [t.visible for t in last.data[len(months):]] == [False] * 3 + [True] * 3
    for st in fig.layout.sliders[0].steps:                                          # a csúszka csak a hónap-rétegeket állítja
        assert list(st.args[1]) == list(range(len(months)))
    one = tf.topic_map(xy, df, names, growth, "cs.CL", questions=qs[:1])
    assert not one.layout.updatemenus                                               # egy kérdésnél nincs menü


def test_app_question_map_and_report(app):
    class Cite:
        models = None
        def __init__(self): self.models = self
        def generate_content(self, model, contents, config=None):
            ids = re.findall(r"^\[(\d{4}\.\d{4,5})\]", contents, re.M)
            return types.SimpleNamespace(text=f"Válasz [{ids[0]}]." if ids else "{}")
    app.rag.llm = Cite()
    app.rag.log.clear()
    assert not app.has_questions() and app.asked_questions() == []
    app.rag.ask("A notebookban feltett kérdés", show=False)                  # a notebookból: a jelentésbe igen,
    assert not app.has_questions() and len(app.asked_questions()) == 1      # a felület jelölőnégyzetét nem kapcsolja
    app.rag.log.clear()
    app.ask("Milyen ágensek vannak?")
    assert app.has_questions() and len(app.asked_questions(ui_only=True)) == len(app.asked_questions()) == 1
    fig = app.map_figure(questions=app.asked_questions(), active_question="last")
    assert [t.name for t in fig.data].count("q-csillag") == 1
    html = app.build_html()
    assert "a csillag a kérdés helye" in html and "menüből választható" not in html


def test_pull_stays_effective_with_spreading():
    """A „Témák összehúzása” csúszka a széthúzás mellett is hat (korábban 0 és ~0,6 között hatástalan volt)."""
    from trendfigyelo.tmap import PULL, _landscape, _layout, _pull, _spread_islands
    rng = np.random.default_rng(3)
    topics = np.repeat(np.arange(6), 80)
    xy = rng.normal(size=(6, 2))[topics] * 0.6 + rng.normal(size=(480, 2))      # erősen átfedő témák

    def compactness(p):
        out = _layout(xy, topics, p, True)
        c = np.array([np.median(out[topics == t], axis=0) for t in range(6)])
        return np.mean([out[topics == t].std(axis=0).mean() for t in range(6)]) / c.std(axis=0).mean()
    vals = [compactness(p) for p in (0, 0.25, 0.5, 0.75)]
    assert vals[0] > vals[1] > vals[2] > vals[3]                               # fokozatosan tömörebb foltok
    old = _landscape(_spread_islands(_pull(xy, topics, PULL), topics))          # az alapérték a korábbi kép
    assert np.allclose(_layout(xy, topics, PULL, True), old, atol=1e-6)
