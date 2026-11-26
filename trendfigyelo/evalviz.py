"""A kiértékelés ábrái: hányadik helyen hozza a keresés a jó cikket (négyzetrács, magyar és angol sor),
és a válaszok kézi pontozása (eval/qa_pontozas.csv). Plotly-ábrák a felülethez, a jelentéshez és a notebookhoz."""
import html
import os
import re
import textwrap

import pandas as pd

from .rag import detect_language

# Sorrendi skála egy árnyalatban (sötét = jobb); a „nincs” semleges szürke. A dataviz-ellenőrző szerint
# a három kék lépés egyértelműen sorba rendezhető, és a legvilágosabb is elválik a háttértől.
RANK_BINS = [
    ("1. hely", "#1c5cab"),
    ("2–5. hely", "#3987e5"),
    ("6–10. hely", "#86b6ef"),
    ("nincs az első 10-ben", "#d6d4cd"),
]
LANG_NAMES = {"hu": "Magyar", "en": "Angol"}
INK = "#7a7f87"          # középszürke: világos és sötét háttéren is olvasható (mint a trendgrafikonon)


def rank_bin(rank):
    """A helyezés kategóriája: 0 = első, 1 = 2–5., 2 = 6–10., 3 = nincs az első 10-ben."""
    if rank is None or pd.isna(rank):
        return 3
    rank = int(rank)
    return 0 if rank == 1 else 1 if rank <= 5 else 2 if rank <= 10 else 3


def eval_overview(eval_df):
    """Darabszámok a szöveges összefoglalóhoz: nyelvenként és összesen hány kérdés, hánynál első, első 5-ben, első 10-ben."""
    out = {}
    groups = [("összes", eval_df)] + [(LANG_NAMES.get(l, l), g) for l, g in eval_df.groupby("nyelv_kod", sort=False)] \
        if "nyelv_kod" in eval_df else [("összes", eval_df)]
    for name, g in groups:
        out[name] = {"n": len(g), "hit@1": int(g["hit@1"].sum()), "hit@5": int(g["hit@5"].sum()),
                     "hit@10": int(g["hit@10"].sum())}
    return out


def overview_text(eval_df):
    """Közérthető, egybekezdéses összefoglaló (Markdown) a felület és a notebook számára."""
    o = eval_overview(eval_df)
    a = o["összes"]
    parts = [f"**{a['n']} tesztkérdésből** a jó cikk **{a['hit@1']}-nél az első helyen** van, "
             f"**{a['hit@5']}-nél az első 5 között** (ennyit kap meg alapból a nyelvi modell), "
             f"**{a['hit@10']}-nél az első 10 között**."]
    langs = [f"{k.lower()} kérdéseknél {v['hit@5']}/{v['n']}" for k, v in o.items() if k != "összes"]
    if len(langs) > 1:
        parts.append("Az első 5-ben: " + ", ".join(langs) + ".")
    return " ".join(parts)


ROW_LEN = 10   # ennél több kérdésnél a nyelv kérdései több sorba törnek (telefonon is olvasható négyzetek)


def eval_figure(eval_df, title=True, row_len=ROW_LEN):
    """Négyzetrács: egy négyzet egy tesztkérdés, nyelvenként egy (sok kérdésnél több, legfeljebb row_len hosszú)
    sor, a helyezés szerint rendezve. A szín a helyezés kategóriája, a szám a helyezés; az egér alatt a kérdés és a
    várt cikk."""
    import plotly.graph_objects as go

    df = eval_df.copy()
    if "nyelv_kod" not in df:
        df["nyelv_kod"] = [detect_language(str(q)) for q in df["kérdés"]]
    df["kat"] = [rank_bin(r) for r in df["helyezés"]]
    df["_r"] = pd.to_numeric(df["helyezés"], errors="coerce").fillna(99)
    langs = [l for l in ("hu", "en") if (df.nyelv_kod == l).any()] + \
        [l for l in df.nyelv_kod.unique() if l not in ("hu", "en")]
    width = min(max(int((df.nyelv_kod == l).sum()) for l in langs), row_len)

    z, text, hover, ylab = [], [], [], []
    for l in langs:
        g = df[df.nyelv_kod == l].sort_values(["_r"], kind="stable")
        cells_z = [k + 0.5 for k in g.kat]                              # a lépcsős színskála sávjainak közepe
        cells_t = [("–" if k == 3 else f"{int(r)}") for k, r in zip(g.kat, g._r)]
        cells_h = [f"<b>{html.escape(textwrap.shorten(str(q), 110, placeholder='…'))}</b><br>"
                   + "<br>".join(textwrap.wrap(f"Várt cikk: {v} · {RANK_BINS[k][0]}"
                                               + (f" (a(z) {int(r)}. találat)" if k < 3 else ""), 70))
                   for q, v, k, r in zip(g["kérdés"], g.get("várt cikk", [""] * len(g)), g.kat, g._r)]
        for s in range(0, len(g), width):
            pad = width - len(cells_z[s:s + width])
            z.append(cells_z[s:s + width] + [None] * pad)
            text.append(cells_t[s:s + width] + [""] * pad)
            hover.append(cells_h[s:s + width] + [""] * pad)
            # a folytatósorok felirata üres, de a kategóriáknak különbözniük kell (eltérő számú szóköz)
            ylab.append(f"{LANG_NAMES.get(l, l)} ({len(g)})<br><span style='font-size:10px'>első 5-ben "
                        f"<b>{int(g['hit@5'].sum())}/{len(g)}</b></span>" if s == 0 else " " * len(ylab))

    n = len(RANK_BINS)
    scale = []
    for i, (_, c) in enumerate(RANK_BINS):                 # lépcsős színskála: 0..3 egész értékek
        scale += [[i / n, c], [(i + 1) / n, c]]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(1, width + 1)), y=ylab,
        zmin=0, zmax=n, colorscale=scale, showscale=False, xgap=3, ygap=10,
        text=text, texttemplate="%{text}", textfont=dict(size=13), hovertext=hover, hoverinfo="text"))
    for i, lab in enumerate(ylab[1:], 1):                   # elválasztó vonal a nyelvek között
        if lab.strip():
            fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=i - 0.5, y1=i - 0.5,
                          line=dict(color="rgba(128,128,128,0.45)", width=1, dash="dot"))
    for name, color in RANK_BINS:                           # jelmagyarázat (a hőtérképnek nincs kategória-legendája)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, hoverinfo="skip",
                                 marker=dict(symbol="square", size=13, color=color)))
    fig.update_layout(
        title=dict(text="Hányadik helyen hozza a keresés a jó cikket?<br><sup>egy négyzet = egy tesztkérdés; "
                        "a szám a jó cikk helye a találatok között</sup>", x=0, font=dict(size=15)) if title else None,
        height=130 + 52 * len(ylab), margin=dict(l=10, r=10, t=70 if title else 20, b=10),
        xaxis=dict(visible=False, fixedrange=True), yaxis=dict(autorange="reversed", fixedrange=True, ticks="",
                                                             showgrid=False, ticksuffix="  "),
        legend=dict(orientation="h", x=0, y=-0.04, yanchor="top", font=dict(size=11), itemclick=False,
                    itemdoubleclick=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        hoverlabel=dict(align="left", font_size=12))
    return fig


def load_scores(path):
    """A kézzel pontozott válaszok (eval/qa_pontozas.csv: question, answer, …, pont, megjegyzes).
    A megjegyzés elején álló kód (pl. „HU04:”) adja a nyelvet és a sorszámot; ha nincs, a kérdésből felismert nyelv."""
    if not path or not os.path.exists(path):
        return None
    if str(path).endswith(".md"):                                   # kitöltött válaszlap (answersheet.py)
        from .answersheet import parse_sheet
        df = parse_sheet(open(path, encoding="utf-8").read())
        if df.empty:
            return None
    else:
        df = pd.read_csv(path)
    if not {"question", "pont"} <= set(df.columns):
        return None
    note = df.get("megjegyzes", pd.Series([""] * len(df))).fillna("").astype(str)
    codes = [re.match(r"\s*([A-Z]{2}\d{1,3})\s*:\s*", s) for s in note]
    out = pd.DataFrame({
        "kód": [m.group(1) if m else "" for m in codes],
        "kérdés": df.question.astype(str),
        "pont": pd.to_numeric(df.pont, errors="coerce"),
        "indoklás": [s[m.end():] if m else s for s, m in zip(note, codes)],
    })
    out["nyelv_kod"] = [c[:2].lower() if c[:2] in ("HU", "EN") else detect_language(q)
                        for c, q in zip(out["kód"], out["kérdés"])]
    return out.dropna(subset=["pont"]).reset_index(drop=True)


def scores_figure(scores, title=True):
    """Vízszintes sávok: kézzel pontozott válaszonként a pont (1–5), magyar és angol csoportban."""
    import plotly.graph_objects as go

    s = scores.copy()
    s["_o"] = s.nyelv_kod.map({"hu": 0, "en": 1}).fillna(2)
    s["_k"] = s["kód"].replace("", "~")                                 # kód nélküliek a csoport végén
    s = s.sort_values(["_o", "_k"], kind="stable").reset_index(drop=True)
    labels = [f"{c or LANG_NAMES.get(l, l)} · {textwrap.shorten(q, 30, placeholder='…')}"
              for c, l, q in zip(s["kód"], s.nyelv_kod, s["kérdés"])]
    hover = [f"<b>{html.escape(textwrap.shorten(q, 110, placeholder='…'))}</b><br>{int(p)} pont<br>"
             + "<br>".join(textwrap.wrap(html.escape(textwrap.shorten(n, 330, placeholder='…')), 70))
             for q, p, n in zip(s["kérdés"], s.pont, s["indoklás"])]
    fig = go.Figure(go.Bar(
        x=s.pont, y=labels, orientation="h", marker=dict(color="#2a78d6", line=dict(width=0)),
        text=[f"{int(p)}" for p in s.pont], textposition="outside", cliponaxis=False,
        hovertext=hover, hoverinfo="text", width=0.62))
    mean = float(s.pont.mean())
    fig.add_vline(x=mean, line=dict(color=INK, width=1, dash="dot"))
    fig.add_annotation(x=mean, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"átlag {mean:.1f}".replace(".", ","), font=dict(size=11, color=INK))
    fig.update_layout(
        title=dict(text=f"A válaszok kézi pontozása (1–5)<br><sup>{len(s)} válasz; az egér alatt az indoklás</sup>",
                   x=0, font=dict(size=15)) if title else None,
        height=120 + 34 * len(s), margin=dict(l=10, r=30, t=80 if title else 30, b=36), bargap=0.3,
        xaxis=dict(range=[0, 5.4], tickvals=[1, 2, 3, 4, 5], gridcolor="rgba(128,128,128,0.18)", fixedrange=True,
                   zeroline=False),
        yaxis=dict(autorange="reversed", fixedrange=True, ticksuffix="  "),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        hoverlabel=dict(align="left", font_size=12), showlegend=False)
    return fig


def rank_text(rank):
    """A helyezés a táblázatba: '1.', '7.' vagy '–' (nincs az első 10-ben)."""
    return "–" if rank is None or pd.isna(rank) else f"{int(rank)}."


def eval_view(eval_df):
    """A felület táblázata: kérdés, nyelv, a jó cikk helye, várt cikk (a hit@k oszlopok a szövegben vannak)."""
    return pd.DataFrame({
        "kérdés": eval_df["kérdés"],
        "nyelv": [LANG_NAMES.get(l, l).lower() for l in (eval_df["nyelv_kod"] if "nyelv_kod" in eval_df
                                                           else [detect_language(str(q)) for q in eval_df["kérdés"]])],
        "a jó cikk helye": [rank_text(r) for r in eval_df["helyezés"]],
        "várt cikk": eval_df.get("várt cikk", [""] * len(eval_df)),
    })


__all__ = ["RANK_BINS", "rank_bin", "eval_overview", "overview_text", "eval_figure", "load_scores", "scores_figure",
           "rank_text", "eval_view"]
