"""Önálló (egyfájlos) HTML-jelentés a trendekről, a kérdés–válaszokról és a kiértékelésről.

A modell- és cikkszövegek nem megbízható bemenetek: mindent HTML-escape-elünk, a válaszokban csak
egy szűk, biztonságos Markdown-részhalmazt alakítunk át (félkövér, dőlt, felsorolás, arXiv-hivatkozás).
"""
import html
import re
import textwrap
from datetime import datetime

import pandas as pd

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
    mert a növekedés ezek átlagának hányadosa. A vonalak 2 havi mozgóátlagok."""
    import plotly.graph_objects as go
    top = list(topic_ids[:n])
    raw = share[top] * 100
    smooth = raw.rolling(2, min_periods=1).mean()
    months = list(smooth.index)
    fig = go.Figure()

    for pos, label, x0, x1 in (("first", "bázis", -0.5, window - 0.5),
                               ("last", "összehasonlítás", len(months) - window - 0.5, len(months) - 0.5)):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(128,128,128,0.10)", line_width=0, layer="below",
                      annotation_text=f"{label} ({window} hó)", annotation_position="top left",
                      annotation_font=dict(size=11, color="#7a7f87"))

    for rank in range(len(top) - 1, -1, -1):          # a kiemeltek kerüljenek legfelülre
        t = top[rank]
        strong = rank < N_HIGHLIGHT
        color = PALETTE[rank] if strong else GREY
        hover = html.escape(textwrap.shorten(names[t]["nev"], 60)) + f" ({growth[t]:.1f}×)"
        if strong:
            fig.add_trace(go.Scatter(x=months, y=raw[t].round(2), mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=5, color=color, opacity=0.35)))
        fig.add_trace(go.Scatter(
            x=months, y=smooth[t].round(2), mode="lines", showlegend=False, name=hover,
            line=dict(width=3.2 if strong else 1.6, color=color, shape="spline", smoothing=0.6),
            opacity=1 if strong else 0.7,
            hovertemplate="%{x}: %{y:.1f}%<extra>" + hover + "</extra>"))

    ymax = float(max(smooth.max().max(), raw[top[:N_HIGHLIGHT]].max().max()))
    ys = spread_labels([float(smooth[t].iloc[-1]) for t in top], gap=0.055 * ymax, top=ymax * 1.05)
    for rank, t in enumerate(top):
        strong = rank < N_HIGHLIGHT
        fig.add_annotation(x=months[-1], y=ys[rank], xanchor="left", xshift=8, showarrow=False, align="left",
                           text=(f"<b>{html.escape(short_name(names[t]['nev']))}</b>" if strong
                                 else html.escape(short_name(names[t]["nev"]))) + f"  {growth[t]:.1f}×",
                           font=dict(size=12 if strong else 11, color=PALETTE[rank] if strong else "#7a7f87"))

    fig.update_layout(
        title=dict(text=f"A leggyorsabban növő témák – arXiv {category}"
                        "<br><sup>vonal: 2 havi mozgóátlag · pontok: havi nyers érték · a szám a növekedés "
                        f"(utolsó {window} hó / első {window} hó)</sup>", x=0, font=dict(size=16)),
        yaxis=dict(title="A havi cikkek aránya (%)", rangemode="tozero", gridcolor="rgba(128,128,128,0.2)",
                   range=[0, ymax * 1.15], zeroline=False),
        xaxis=dict(type="category", tickangle=-45, showgrid=False, range=[-0.5, len(months) - 0.5]),
        margin=dict(l=60, r=270, t=80, b=20), height=500 + 14 * max(0, len(top) - 8),
        hovermode="closest", showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a7f87"))
    return fig


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


def build_report(meta, table, fig=None, summary="", qa_log=None, papers=None, eval_df=None, eval_summary=None,
                 archive=None, n_show=8, archive_prefix="archiv/"):
    """meta: dict(category, months, n_papers, downloaded, emb_model, llm_model, code_version, generated, authors, repo_url).
    archive: korábbi jelentések dátumcímkéi; archive_prefix: "archiv/" a kezdőoldalon, "" az archív oldalakon."""
    papers = papers if papers is not None else pd.DataFrame(columns=["id", "title"])
    titles = dict(zip(papers.id.astype(str), papers.title)) if len(papers) else {}
    e = html.escape
    generated = meta.get("generated") or datetime.now().strftime("%Y-%m-%d %H:%M")
    downloaded = f' · adat letöltve: {e(meta["downloaded"])}' if meta.get("downloaded") else ""

    sections = []
    top = table.head(n_show)
    chart = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False, "responsive": True}) \
        if fig is not None else ""
    sections.append(f"""
<section id="trendek">
  <h2>Leggyorsabban növő témák</h2>
  <p class="lead">A témákat a cikkek vektorainak csoportosítása adja; a növekedés az utolsó 3 hónap részaránya osztva
  az első 3 hónapéval. A neveket és a leírásokat nyelvi modell adta a kulcsszavak és a cikkcímek alapján.</p>
  <div class="chart">{chart}</div>
  {_table(top, ["név", "leírás", "kulcsszavak", "cikkek", "növekedés"],
          formats={"név": lambda v: f"<strong>{e(str(v))}</strong>", "növekedés": _growth_cell(float(top["növekedés"].max() or 1))},
          classes={"kulcsszavak": "muted small", "cikkek": "num", "növekedés": "growth"})}
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
            cards.append(f"""<article class="qa"><h3>{e(item["question"])}</h3><div class="prose">{md_lite(item["answer"])}</div>
{warn}<details><summary>Források ({len(ids)})</summary><ol class="sources">{src}</ol></details></article>""")
        sections.append(f'<section id="kerdesek"><h2>Kérdések és válaszok</h2>{"".join(cards)}</section>')

    if eval_df is not None and len(eval_df):
        stats = "".join(f'<div class="stat"><span>{e(k)}</span><b>{float(v):.2f}</b></div>'
                        for k, v in (eval_summary or {}).items())
        sections.append(f"""<section id="kiertekeles"><h2>Kiértékelés</h2>
<p class="lead">Hányszor van a várt cikk az első 1, 5 és 10 találat között ({len(eval_df)} tesztkérdés, <code>eval/kerdesek.csv</code>).</p>
<div class="stats">{stats}</div>{_table(eval_df, list(eval_df.columns), classes={c: "num" for c in eval_df.columns if c.startswith("hit")})}</section>""")

    if archive:
        links = "".join(f'<li><a href="{archive_prefix}{e(a)}.html">{e(a)}</a></li>' for a in sorted(archive, reverse=True))
        sections.append(f'<section id="archiv"><h2>Korábbi jelentések</h2><ul class="archive">{links}</ul></section>')

    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in [("trendek", "Trendek"), ("osszefoglalo", "Összefoglaló"),
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
.stats {{ display:flex; gap:12px; flex-wrap:wrap; margin:0 0 14px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:110px; }}
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
