"""A webes felület, a HTML-jelentés és a közzététel tesztjei (hálózat és Colab nélkül)."""
import types

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
    for part in ('id="trendek"', 'id="osszefoglalo"', 'id="kerdesek"', 'id="kiertekeles"', 'id="archiv"',
                 'href="archiv/2026-11-13.html"', "Süle Szilvia", "cdn.plot.ly"):
        assert part in html, part
    assert "<img src=x" not in html                        # a cikkcím nem futtatható HTML-ként
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
