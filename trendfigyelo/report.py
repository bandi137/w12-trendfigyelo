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


def trend_figure(share, growth, names, topic_ids, category, n=6):
    """Interaktív Plotly-grafikon a leggyorsabban növő n téma havi részarányáról (2 havi mozgóátlag)."""
    import plotly.graph_objects as go
    top = topic_ids[:n]
    smooth = (share[top] * 100).rolling(2, min_periods=1).mean()
    fig = go.Figure()
    for i, t in enumerate(top):
        fig.add_trace(go.Scatter(
            x=list(smooth.index), y=smooth[t].round(2), mode="lines+markers",
            name="<br>".join(textwrap.wrap(f"{names[t]['nev']} ({growth[t]:.1f}×)", 42)),
            line=dict(width=2.5, color=PALETTE[i % len(PALETTE)]), marker=dict(size=6),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + html.escape(textwrap.shorten(names[t]["nev"], 50)) + "</extra>"))
    fig.update_layout(
        title=dict(text=f"A leggyorsabban növő témák – arXiv {category}", x=0, font=dict(size=16)),
        yaxis=dict(title="A havi cikkek aránya (%)", rangemode="tozero", gridcolor="rgba(128,128,128,0.25)"),
        xaxis=dict(type="category", tickangle=-45, showgrid=False),
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, font=dict(size=12)),
        margin=dict(l=60, r=20, t=50, b=20), height=520, hovermode="x unified",
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
    """meta: dict(category, months, n_papers, emb_model, llm_model, code_version, generated, authors, repo_url).
    archive: korábbi jelentések dátumcímkéi; archive_prefix: "archiv/" a kezdőoldalon, "" az archív oldalakon."""
    papers = papers if papers is not None else pd.DataFrame(columns=["id", "title"])
    titles = dict(zip(papers.id.astype(str), papers.title)) if len(papers) else {}
    e = html.escape
    generated = meta.get("generated") or datetime.now().strftime("%Y-%m-%d %H:%M")

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
  <p class="meta">{e(str(meta.get("n_papers", "")))} cikk · {e(meta.get("months", ""))} · frissítve: {e(generated)}</p>
  <nav>{nav}</nav>
</header>
<main>{"".join(sections)}</main>
<footer>
  Készült: {e(generated)} · Vektorizálás: {e(meta.get("emb_model", ""))} · Nyelvi modell: {e(meta.get("llm_model", ""))}
  · Kódverzió: {e(meta.get("code_version", ""))}<br>
  {e(meta.get("authors", ""))} · <a href="{e(meta.get("repo_url", ""))}">Forráskód</a> · W12 – Kutatási trendfigyelő RAG-gal
</footer>
</body></html>"""
