"""Önálló (egyfájlos) HTML-jelentés a trendekről, a kérdés–válaszokról és a kiértékelésről.

A modell- és cikkszövegek nem megbízható bemenetek: mindent HTML-escape-elünk, a válaszokban csak
egy szűk, biztonságos Markdown-részhalmazt alakítunk át (félkövér, dőlt, felsorolás, arXiv-hivatkozás).
"""
import html
import math
import re
import textwrap
from datetime import datetime

import pandas as pd

from .trends import growth_note
from .verify import ICONS, as_list, summary_counts

ARXIV_REF = re.compile(r"\[(\d{4}\.\d{4,5})\]")
PALETTE = ["#2a6fdb", "#e0663a", "#2e9e6b", "#a64ca6", "#c9a227", "#3aa5c4", "#d1495b", "#6b7a8f"]


def arxiv_url(paper_id):
    return f"https://arxiv.org/abs/{paper_id}"


def md_lite(text):
    """Biztonságos mini-Markdown: escape, majd **félkövér**, *dőlt*, felsorolás és [arXiv-azonosító] → link."""
    def inline(s):
        s = html.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        return ARXIV_REF.sub(lambda m: f'[<a href="{arxiv_url(m.group(1))}">{m.group(1)}</a>]', s)

    out, items = [], []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        for line in block.splitlines():
            m = re.match(r"\s*[*\-•]\s+(.*)", line)
            if m:
                items.append(f"<li>{inline(m.group(1))}</li>")
                continue
            if items:
                out.append("<ul>" + "".join(items) + "</ul>")
                items = []
            if line.strip():
                out.append(f"<p>{inline(line.strip())}</p>")
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
            items = []
    return "\n".join(out)


GREY = "#9aa3ad"
N_HIGHLIGHT = 3     # ennyi legjobban növő téma kap színt és vastag vonalat, a többi halványszürke


def spread_labels(values, gap, top=None):
    """Feliratok függőleges helyzete úgy, hogy ne fedjék egymást: a szomszédos feliratok legalább `gap`
    távolságra kerülnek. Bemenet és kimenet ugyanabban a sorrendben; `top`: a legmagasabb megengedett hely."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = {}
    prev = None
    for i in order:
        pos[i] = values[i] if prev is None else max(values[i], prev + gap)
        prev = pos[i]
    if top is not None and order and pos[order[-1]] > top:        # túllógtunk felül: visszatoljuk lefelé
        nxt = top
        for i in reversed(order):
            pos[i] = min(pos[i], nxt)
            nxt = pos[i] - gap
    return [pos[i] for i in range(len(values))]


def short_name(name, n=34):
    """Rövid felirat: ha hosszú, előbb az angol zárójeles rész marad ki (a teljes név az egérmutatóra látszik)."""
    if len(name) > n:
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name) or name
    return textwrap.shorten(name, n, placeholder="…")


def trend_figure(share, growth, names, topic_ids, category, n=6, window=3):
    """Interaktív Plotly-grafikon a leggyorsabban növő n téma havi részarányáról.

    A legjobb N_HIGHLIGHT téma színes és vastag (mögötte halványan a nyers havi érték), a többi halványszürke;
    a nevek a vonalak végén állnak (nincs külön jelmagyarázat); az első és az utolsó `window` hónap sávval jelölve,
    mert a növekedés ezek simított átlagának hányadosa (trends.growth_note). A vonalak 2 havi mozgóátlagok.
    Keskeny képernyőn (a jelentésben, TREND_SCRIPT) a vonalvégi nevek helyett jelmagyarázat jelenik meg alul."""
    import plotly.graph_objects as go
    top = list(topic_ids[:n])
    raw = share[top] * 100
    smooth = raw.rolling(2, min_periods=1).mean()
    months = list(smooth.index)
    fig = go.Figure()

    for side, label, x0, x1 in (("left", "bázis", -0.5, window - 0.5),
                                ("right", "összehasonlítás", len(months) - window - 0.5, len(months) - 0.5)):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(128,128,128,0.10)", line_width=0, layer="below",
                      annotation_text=f"{label} ({window} hó)", annotation_position=f"top {side}",
                      annotation_font=dict(size=11, color="#7a7f87"))

    for rank in range(len(top) - 1, -1, -1):          # a kiemeltek kerüljenek legfelülre
        t = top[rank]
        strong = rank < N_HIGHLIGHT
        color = PALETTE[rank] if strong else GREY
        hover = html.escape(textwrap.shorten(names[t]["nev"], 60)) + f" ({growth[t]:.1f}×)"
        if strong:
            fig.add_trace(go.Scatter(x=months, y=raw[t].round(2), mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=5, color=color, opacity=0.35)))
        legend = (f"<b>{html.escape(short_name(names[t]['nev']))}</b>" if strong
                  else html.escape(short_name(names[t]["nev"]))) + f" {growth[t]:.1f}×"
        fig.add_trace(go.Scatter(
            x=months, y=smooth[t].round(2), mode="lines", showlegend=True, name=legend, legendrank=rank,
            line=dict(width=3.2 if strong else 1.6, color=color, shape="spline", smoothing=0.6),
            opacity=1 if strong else 0.7,
            hovertemplate="%{x}: %{y:.1f}%<extra>" + hover + "</extra>"))     # a jelmagyarázat csak keskenyen látszik

    ymax = float(max(smooth.max().max(), raw[top[:N_HIGHLIGHT]].max().max()))
    ys = spread_labels([float(smooth[t].iloc[-1]) for t in top], gap=0.055 * ymax, top=ymax * 1.05)
    for rank, t in enumerate(top):
        strong = rank < N_HIGHLIGHT
        fig.add_annotation(x=months[-1], y=ys[rank], xanchor="left", xshift=8, showarrow=False, align="left",
                           text=(f"<b>{html.escape(short_name(names[t]['nev']))}</b>" if strong
                                 else html.escape(short_name(names[t]["nev"]))) + f"  {growth[t]:.1f}×",
                           font=dict(size=12 if strong else 11, color=PALETTE[rank] if strong else "#7a7f87"),
                           name="veg")

    fig.update_layout(
        title=dict(text=f"A leggyorsabban növő témák – arXiv {category}"
                        f"<br><sup>vonal: 2 havi mozgóátlag · pontok: havi nyers érték · "
                        f"{growth_note(window, short=True)}</sup>", x=0, font=dict(size=16)),
        yaxis=dict(title="A havi cikkek aránya (%)", rangemode="tozero", gridcolor="rgba(128,128,128,0.2)",
                   range=[0, ymax * 1.15], zeroline=False),
        xaxis=dict(type="category", tickangle=-45, showgrid=False, range=[-0.5, len(months) - 0.5]),
        margin=dict(l=60, r=270, t=80, b=20), height=500 + 14 * max(0, len(top) - 8),
        hovermode="closest", showlegend=False,
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.2, yanchor="top", font=dict(size=11),
                    traceorder="normal", itemclick=False, itemdoubleclick=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a7f87"))
    return fig


# A jelentésben (fig.to_html(post_script=TREND_SCRIPT)): keskeny képernyőn (< 640 px) a vonalvégi nevek helyett
# jelmagyarázat alul, kis jobb margó és rövid cím; a rögzített 270 px-es jobb margó mobilon szinte az egész
# rajzterületet elvette. A {plot_id}-t a Plotly cseréli.
TREND_SCRIPT = r"""
(function () {
  var gd = document.getElementById('{plot_id}');
  var full = {title: gd.layout.title.text, height: gd.layout.height, margin: JSON.parse(JSON.stringify(gd.layout.margin)),
              ytitle: (gd.layout.yaxis.title && gd.layout.yaxis.title.text) || ''};
  var nLegend = gd.data.filter(function (t) { return t.mode === 'lines'; }).length;
  function fit() {
    var width = gd.parentElement ? gd.parentElement.clientWidth : gd.clientWidth;
    if (!width) return;
    var n = width < 640;
    var upd = {width: width, showlegend: n, 'title.text': n ? full.title.split('<br>')[0] : full.title,
               'margin.r': n ? 12 : full.margin.r, 'margin.l': n ? 44 : full.margin.l,
               'margin.b': n ? 30 + 22 * nLegend : full.margin.b,
               height: n ? 360 + 22 * nLegend : full.height,
               'yaxis.title.text': n ? '%' : full.ytitle};
    (gd.layout.annotations || []).forEach(function (a, i) {
      if (a.name === 'veg') upd['annotations[' + i + '].visible'] = !n;
    });
    gd.style.height = upd.height + 'px';
    if (gd.parentElement && gd.parentElement.style.height) gd.parentElement.style.height = upd.height + 'px';
    Plotly.relayout(gd, upd);
  }
  var t = null;
  fit();
  window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(fit, 120); });
})();
"""


def _table(df, columns, formats=None, classes=None):
    formats, classes = formats or {}, classes or {}
    head = "".join(f'<th class="{classes.get(c, "")}">{html.escape(c)}</th>' for c in columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            cell = formats[c](v) if c in formats else html.escape(str(v))
            cells.append(f'<td class="{classes.get(c, "")}">{cell}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _growth_cell(max_growth):
    def fmt(v):
        width = max(4, min(100, 100 * float(v) / max_growth)) if max_growth else 0
        return f'<div class="bar"><span style="width:{width:.0f}%"></span><b>{float(v):.2f}×</b></div>'
    return fmt


def _sample_note(meta):
    """A mintavétel leírása a trendek fölé: 'napi' – a hónap minden napjáról; 'honap_vege' – a régi letöltés."""
    if not meta.get("per_month"):
        return ""
    n = int(meta["per_month"])
    if meta.get("sampling") == "napi":
        return (f" Havonta legfeljebb {n} cikk szerepel, a hónap minden napjáról (naponta legfeljebb kb. "
                f"{math.ceil(n / 30)}; ha egy napon kevesebb jelent meg, mind), így a trendek az egész hónapot lefedik.")
    return (f" Havonta legfeljebb {n} cikk szerepel: az arXiv-letöltés a hónap utolsó napjain beküldött cikkeket "
            "veszi át, ezért a trendek erre a mintára vonatkoznak.")


EVAL_LEAD = ("A válasz minősége azon múlik, hogy a keresés megtalálja-e a jó cikket: a nyelvi modell csak az első "
             "néhány (alapból 5) találatot kapja meg. Ezért tesztkérdéseket írtunk, mindegyikhez kikerestük azt a cikket, "
             "amelyik valóban válaszol rá, és megnéztük, hányadik helyen hozza a keresés (<code>eval/kerdesek.csv</code>). "
             "Egy négyzet egy kérdés: minél több a sötétkék, annál jobb. A szakirodalom ezt hit@k-nak hívja: például "
             "a hit@5 azt adja meg, a kérdések hányad részénél van a jó cikk az első 5 találat között.")
SCORES_LEAD = ("A keresés mellett a nyelvi modell válaszait is ellenőriztük: a pontozó a választ a modellnek átadott "
               "cikkek absztraktjaival vetette össze, és 1–5 ponttal értékelte, mennyire pontos, teljes és a forrásokkal "
               "alátámasztott (5: pontos és teljes, helyes hivatkozásokkal; alacsonyabb: hiányos válasz, hibás hivatkozás, "
               "vagy a kapott cikkek nem elegendőek a válaszhoz). Egy sávra mutatva látszik az indoklás.")


def _eval_section(eval_df, eval_summary, scores, plotly_loaded):
    """Kiértékelés: közérthető bevezető, összesítő csempék, négyzetrács a helyezésekről, kérdéstáblázat
    és (ha van) a kézi pontozás ábrája. A régi, helyezés nélküli eredményeknél a régi táblázat marad."""
    from .evalviz import RANK_BINS, eval_figure, rank_bin, rank_text, scores_figure
    e = html.escape
    cfg = {"displayModeBar": False, "responsive": True}
    parts = ['<section id="kiertekeles"><h2>Kiértékelés</h2>']

    def plot(f):
        nonlocal plotly_loaded
        out = f.to_html(include_plotlyjs=False if plotly_loaded else "cdn", full_html=False, config=cfg)
        plotly_loaded = True
        return f'<div class="chart">{out}</div>'

    if eval_df is not None:
        n = len(eval_df)
        parts.append(f'<p class="lead">{EVAL_LEAD}</p>')
        labels = {"hit@1": "az első helyen", "hit@5": "az első 5 között", "hit@10": "az első 10 között"}
        parts.append('<div class="stats">' + "".join(
            f'<div class="stat"><span>{labels.get(k, e(k))}</span><b>{int(eval_df[k].sum())}/{n}</b>'
            f'<small>{e(k)} = {float(v):.2f}</small></div>' for k, v in (eval_summary or {}).items() if k in eval_df) + "</div>")
        if "helyezés" in eval_df:
            parts.append(plot(eval_figure(eval_df)))

            def badge(r):
                name, color = RANK_BINS[rank_bin(r)]
                ink = "#fff" if rank_bin(r) < 2 else "#1d2127"
                return f'<span class="rank" style="background:{color};color:{ink}" title="{e(name)}">{rank_text(r)}</span>'
            from .rag import detect_language
            langs = eval_df["nyelv_kod"] if "nyelv_kod" in eval_df else [detect_language(str(q)) for q in eval_df["kérdés"]]
            view = eval_df.assign(nyelv=[{"hu": "magyar", "en": "angol"}.get(l, l) for l in langs])
            cols = ["kérdés", "nyelv", "helyezés"] + (["várt cikk"] if "várt cikk" in view else [])
            parts.append(_table(view.rename(columns={"helyezés": "a jó cikk helye"}),
                                [c if c != "helyezés" else "a jó cikk helye" for c in cols],
                                formats={"a jó cikk helye": badge,
                                         "várt cikk": lambda v: ", ".join(f'<a href="{arxiv_url(i.strip())}">{e(i.strip())}</a>'
                                                                         for i in str(v).split(",") if i.strip())},
                                classes={"a jó cikk helye": "num", "nyelv": "muted small"}))
        else:
            parts.append(_table(eval_df, list(eval_df.columns),
                                classes={c: "num" for c in eval_df.columns if c.startswith("hit")}))
    if scores is not None and len(scores):
        parts.append(f'<h3>A válaszok kézi pontozása</h3><p class="lead">{SCORES_LEAD}</p>{plot(scores_figure(scores, title=False))}')
    parts.append("</section>")
    return "\n".join(parts)


def archive_title(label):
    """Az archív fájlnév olvasható alakja: '2026-11-25_1712' -> '2026-11-25 17:12' (a régi, csak dátumos marad)."""
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(-\d+)?", str(label))
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}" + (f" ({m.group(4)[1:]}.)" if m.group(4) else "") if m else str(label)


def _claims_html(claims, error=""):
    """A hivatkozások tartalmi ellenőrzése a kérdés–válasz kártyán: összegzés, és lenyitva állításonként az ítélet."""
    e = html.escape
    if isinstance(error, str) and error.strip():
        return f'<p class="warn">⚠ A hivatkozások tartalmi ellenőrzése nem futott le: {e(error)}</p>'
    if not claims:
        return ""
    cnt = summary_counts(claims)
    label = {"igen": "alátámasztja", "részben": "részben", "nem": "nem támasztja alá", None: "nem értékelt"}
    rows = "".join(
        f'<li><span title="{label.get(c.get("itelet"))}">{ICONS.get(c.get("itelet"), "·")}</span> '
        f'{e(c.get("mondat", ""))} <span class="muted small">[{e(", ".join(c.get("ids", [])))}]'
        + (f' – {e(c["indoklas"])}' if c.get("indoklas") else "") + "</span></li>" for c in claims)
    cls = "warn" if cnt["nem"] or cnt["részben"] else "muted small"
    head = (f"Hivatkozások tartalmi ellenőrzése: {ICONS['igen']} {cnt['igen']} alátámasztott · "
            f"{ICONS['részben']} {cnt['részben']} részben · {ICONS['nem']} {cnt['nem']} nem alátámasztott"
            + (f" · {cnt[None]} nem értékelt" if cnt[None] else ""))
    return (f'<details class="claims"><summary class="{cls}">{head}</summary><ul class="claims">{rows}</ul>'
            '<p class="muted small">Egy második modellhívás vetette össze az állításokat a hivatkozott cikkek '
            "absztraktjával; jelzés, nem bizonyíték.</p></details>")


def build_report(meta, table, fig=None, summary="", qa_log=None, papers=None, eval_df=None, eval_summary=None,
                 archive=None, n_show=8, archive_prefix="archiv/", map_fig=None, scores=None):
    """meta: dict(category, months, n_papers, downloaded, emb_model, llm_model, code_version, generated, authors, repo_url).
    archive: korábbi jelentések dátumcímkéi; archive_prefix: "archiv/" a kezdőoldalon, "" az archív oldalakon.
    map_fig: a tématérkép (tmap.topic_map), ha van; scores: a kézzel pontozott válaszok (evalviz.load_scores)."""
    papers = papers if papers is not None else pd.DataFrame(columns=["id", "title"])
    titles = dict(zip(papers.id.astype(str), papers.title)) if len(papers) else {}
    e = html.escape
    generated = meta.get("generated") or datetime.now().strftime("%Y-%m-%d %H:%M")
    downloaded = f' · adat letöltve: {e(meta["downloaded"])}' if meta.get("downloaded") else ""

    sections = []
    top = table.head(n_show)
    sample_note = _sample_note(meta)
    chart = fig.to_html(include_plotlyjs="cdn", full_html=False, post_script=TREND_SCRIPT,
                        config={"displayModeBar": False, "responsive": True}) \
        if fig is not None else ""
    sections.append(f"""
<section id="trendek">
  <h2>Leggyorsabban növő témák</h2>
  <p class="lead">A témákat a cikkek vektorainak csoportosítása adja. {growth_note()} A neveket és a leírásokat
  nyelvi modell adta a kulcsszavak és a cikkcímek alapján.{sample_note}</p>
  <div class="chart">{chart}</div>
  {_table(top, ["név", "leírás", "kulcsszavak", "cikkek", "növekedés"],
          formats={"név": lambda v: f"<strong>{e(str(v))}</strong>", "növekedés": _growth_cell(float(top["növekedés"].max() or 1))},
          classes={"kulcsszavak": "muted small", "cikkek": "num", "növekedés": "growth"})}
</section>""")

    if map_fig is not None:
        dim_text = f"{int(meta['emb_dim'])} dimenziós " if meta.get("emb_dim") else ""   # a használt modelltől függ
        n_q = sum(getattr(t, "name", "") == "q-csillag" for t in map_fig.data)
        q_text = ("A „Kérdések és válaszok” kérdései is a térképen vannak: a csillag a kérdés helye (a hozzá "
                  "leghasonlóbb 10 cikk helyének súlyozott átlaga), a karikák a modellnek átadott források"
                  + ("; a bal felső menüből választható, melyik kérdés látsszon" if n_q > 1 else "") + ".\n  "
                  if n_q else "")
        from .tmap import MAP_SCRIPT
        map_html = map_fig.to_html(include_plotlyjs=False if fig is not None else "cdn", full_html=False,
                                   auto_play=False, post_script=MAP_SCRIPT,
                                   config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        sections.append(f"""
<section id="terkep">
  <h2>Tématérkép</h2>
  <p class="lead">Minden pont egy cikk. A cikkek vektorait a UMAP-módszer vetíti síkba, így a hasonló tartalmú
  cikkek egymás mellé kerülnek, és a témák egymásba folynak. Hogy a témák kivehetők legyenek, a cikkek a témájuk
  közepe felé húzódnak, a foltok pedig kissé széthúzódnak (ez csak a megjelenítést érinti). A szín a téma növekedése
  (ugyanaz a simított mutató, mint a trendeknél): piros, ha a részaránya nőtt, kék, ha csökkent. A csúszkával egy hónap cikkei emelhetők ki, a lejátszás végigléptet
  a hónapokon. Egy pontra mutatva látszik a cikk címe, kattintásra megnyílik az arXivon.
  {q_text}A tengelyeknek nincs jelentése és mértékegysége: a UMAP a {dim_text}vektorokat úgy vetíti két
  dimenzióra, hogy a szomszédságok megmaradjanak. A közelség hasonlóságot jelent; a távoli foltok közti távolság nem mérőszám.</p>
  <div class="chart">{map_html}</div>
</section>""")

    if summary:
        sections.append(f'<section id="osszefoglalo"><h2>Összefoglaló</h2><div class="prose">{md_lite(summary)}</div></section>')

    if qa_log:
        cards = []
        for item in qa_log:
            ids = [s.strip() for s in str(item.get("sources", "")).split(",") if s.strip()]
            src = "".join(f'<li><a href="{arxiv_url(i)}">{e(i)}</a> {e(titles.get(i, ""))}</li>' for i in ids)
            warn = f'<p class="warn">⚠ A modell nem kapott forrásra is hivatkozott: {e(", ".join(item["invalid"]))}</p>' \
                if item.get("invalid") else ""
            if str(item.get("uncited", "")).lower() == "true":
                warn += '<p class="warn">⚠ A válasz egyetlen átadott forrásra sem hivatkozik.</p>'
            warn += _claims_html(as_list(item.get("claims")), item.get("verify_error", ""))
            cards.append(f"""<article class="qa"><h3>{e(item["question"])}</h3><div class="prose">{md_lite(item["answer"])}</div>
{warn}<details><summary>Források ({len(ids)})</summary><ol class="sources">{src}</ol></details></article>""")
        sections.append(f'<section id="kerdesek"><h2>Kérdések és válaszok</h2>{"".join(cards)}</section>')

    plotly_loaded = fig is not None or map_fig is not None
    if eval_df is not None and len(eval_df):
        sections.append(_eval_section(eval_df, eval_summary, scores, plotly_loaded))
        plotly_loaded = True
    elif scores is not None and len(scores):
        sections.append(_eval_section(None, None, scores, plotly_loaded))

    if archive:
        links = "".join(f'<li><a href="{archive_prefix}{e(a)}.html">{e(archive_title(a))}</a></li>'
                        for a in sorted(archive, reverse=True))
        sections.append(f'<section id="archiv"><h2>Korábbi jelentések</h2><ul class="archive">{links}</ul></section>')

    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in [("trendek", "Trendek"), ("terkep", "Tématérkép"), ("osszefoglalo", "Összefoglaló"),
                  ("kerdesek", "Kérdések"), ("kiertekeles", "Kiértékelés"), ("archiv", "Archívum")] if f'id="{i}"' in "".join(sections))
    return f"""<!doctype html>
<html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kutatási trendfigyelő – {e(meta.get("category", ""))}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d2127; --muted:#666d77; --line:#e3e0d9; --card:#ffffff; --accent:#2a6fdb; --bar:#cfe0fb; --warn:#a4501b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15171b; --fg:#e7e5e0; --muted:#9aa0a8; --line:#2c3038; --card:#1c1f24; --accent:#7aa7f0; --bar:#27406b; --warn:#f0a36b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
header, main, footer {{ max-width:1080px; margin:0 auto; padding:0 16px; }}
header {{ padding-top:40px; }}
header .eyebrow {{ color:var(--muted); font-size:.85rem; letter-spacing:.06em; text-transform:uppercase; }}
h1 {{ font-size:clamp(1.7rem,4vw,2.4rem); line-height:1.2; margin:.2em 0 .3em; }}
h2 {{ font-size:1.4rem; margin:0 0 .4em; }} h3 {{ font-size:1.05rem; margin:0 0 .5em; }}
.meta {{ color:var(--muted); margin:0; }}
nav {{ display:flex; gap:18px; flex-wrap:wrap; margin:20px 0 0; padding:12px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
nav a {{ color:var(--fg); text-decoration:none; font-weight:600; font-size:.95rem; }} nav a:hover {{ color:var(--accent); }}
section {{ padding:36px 0 8px; }}
.lead {{ color:var(--muted); max-width:72ch; margin-top:0; }}
a {{ color:var(--accent); }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; margin:12px 0 20px; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:.94rem; }}
th, td {{ text-align:left; vertical-align:top; padding:10px 12px; border-bottom:1px solid var(--line); }}
th {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
tr:last-child td {{ border-bottom:0; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.muted {{ color:var(--muted); }} .small {{ font-size:.85rem; }}
td.growth {{ min-width:130px; }}
.bar {{ position:relative; height:24px; }}
.bar span {{ position:absolute; left:0; top:2px; bottom:2px; background:var(--bar); border-radius:4px; }}
.bar b {{ position:relative; padding-left:6px; line-height:24px; font-variant-numeric:tabular-nums; }}
.prose {{ max-width:75ch; }} .prose ul {{ padding-left:1.2em; }}
.qa {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:0 0 14px; }}
.qa details {{ margin-top:6px; }} .qa summary {{ cursor:pointer; color:var(--muted); }}
.sources {{ font-size:.9rem; padding-left:1.4em; }}
.warn {{ color:var(--warn); font-size:.9rem; }}
.qa summary.warn {{ color:var(--warn); }}
.stats {{ display:flex; gap:12px; flex-wrap:wrap; margin:0 0 14px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:110px; }}
.stat small {{ display:block; color:var(--muted); font-size:.75rem; }}
section h3 {{ margin:28px 0 6px; }}
.claims li {{ margin:4px 0; }}
ul.claims {{ list-style:none; padding-left:4px; }}
.rank {{ display:inline-block; min-width:2.2em; padding:1px 6px; border-radius:4px; text-align:center; font-weight:600; }}
.stat span {{ display:block; color:var(--muted); font-size:.8rem; }} .stat b {{ font-size:1.5rem; font-variant-numeric:tabular-nums; }}
.archive {{ columns:3 180px; padding-left:1.2em; }}
footer {{ margin-top:40px; padding:20px 16px 40px; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem; }}
code {{ font-size:.9em; }}
@media (max-width: 640px) {{ th.small, td.small {{ display:none; }} .qa {{ padding:14px; }} }}
</style></head>
<body>
<header>
  <div class="eyebrow">Kutatási trendfigyelő · arXiv {e(meta.get("category", ""))}</div>
  <h1>Mi történik most a(z) {e(meta.get("category", ""))} kutatásban?</h1>
  <p class="meta">{e(str(meta.get("n_papers", "")))} cikk · {e(meta.get("months", ""))}{downloaded} · jelentés: {e(generated)}</p>
  <nav>{nav}</nav>
</header>
<main>{"".join(sections)}</main>
<footer>
  Készült: {e(generated)} · Vektorizálás: {e(meta.get("emb_model", ""))} · Nyelvi modell: {e(meta.get("llm_model", ""))}
  · Kódverzió: {e(meta.get("code_version", ""))}<br>
  {e(meta.get("authors", ""))} · <a href="{e(meta.get("repo_url", ""))}">Forráskód</a> · W12 – Kutatási trendfigyelő RAG-gal
</footer>
</body></html>"""
