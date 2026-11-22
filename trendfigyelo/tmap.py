"""Tématérkép: a cikkek 2D-s térképe a vektoraik alapján, a témák növekedése szerint színezve.

A 768 dimenziós vektorokat UMAP (ha nincs telepítve: t-SNE) vetíti síkba: a hasonló cikkek egymás mellé kerülnek.
A szín a cikk témájának növekedése (kék: csökkenő, szürke: változatlan, piros: növekvő részarány), a témák neve a
téma közepén áll. Egy hónapcsúszkával kiemelhető, hová estek egy adott hónap cikkei.
"""
import hashlib
import html
import json
import os
import re
import textwrap

import numpy as np

# Diverging skála a növekedés log2-jére: kék (csökken) – semleges szürke (változatlan) – piros (nő).
# A szürke középérték világos és sötét háttéren is látszik (a jelentésnek mindkét módja van).
MAP_COLORSCALE = [[0.0, "#256abf"], [0.25, "#6da7ec"], [0.5, "#a9a7a1"], [0.75, "#ec7a6c"], [1.0, "#c23b36"]]
LOG2_RANGE = 2.0                 # a skála vége: 4× növekedés ill. negyedére csökkenés (efölött telítődik)
INK = "#5f6670"                  # szövegszín (feliratok, tengelyek); világos és sötét háttéren is olvasható


def _fingerprint(emb, ids, method):
    h = hashlib.sha1(np.ascontiguousarray(emb[:, :8], dtype=np.float32).tobytes())
    h.update("\n".join(map(str, ids)).encode())
    return dict(method=method, n=int(len(emb)), sha1=h.hexdigest())


def _project(emb, method, seed):
    if method == "umap":
        import inspect
        import umap
        kw = dict(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=seed)
        if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:   # újabb umap: a megszokott elrendezés
            kw["compatibility_layout"] = True
        return umap.UMAP(**kw).fit_transform(emb)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = PCA(n_components=min(50, emb.shape[1], len(emb) - 1), random_state=seed).fit_transform(emb)
    return TSNE(n_components=2, perplexity=min(30, max(5, (len(emb) - 1) // 4)), init="pca",
                random_state=seed).fit_transform(x)


def map_coords(emb, ids, data_dir=None, method="auto", seed=42):
    """2D koordináták a cikkekhez (cikk × 2). Ha `data_dir` meg van adva, a Drive-on tárolja (terkep.npy +
    terkep.json), és csak akkor számol újra, ha a vektorok, a cikklista vagy a módszer változott."""
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "tsne"
    want = _fingerprint(np.asarray(emb), list(ids), method)
    if data_dir:
        path, meta_path = os.path.join(data_dir, "terkep.npy"), os.path.join(data_dir, "terkep.json")
        have = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
        if have == want and os.path.exists(path):
            print("Térkép-koordináták betöltve a Drive-ról")
            return np.load(path)
    print(f"Tématérkép számítása ({method.upper()}, {len(emb)} cikk)…")
    xy = np.asarray(_project(np.asarray(emb, dtype=np.float32), method, seed), dtype=float)
    xy = (xy - xy.mean(axis=0)) / xy.std(axis=0)          # egységes lépték, hogy a feliratok mérete stimmeljen
    if data_dir:
        np.save(path, xy)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(want, f, indent=1)
    return xy


def _label_name(name, n=30):
    """A térképen rövid név: az angol zárójeles rész nélkül (az egérmutatóra a teljes név látszik)."""
    short = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip() or name
    return textwrap.shorten(short, n, placeholder="…")


def _place_labels(anchors, sizes, iters=300):
    """Egyszerű taszítás: az egymást fedő feliratdobozokat függőlegesen széttolja.
    anchors: (k, 2) középpontok; sizes: (k, 2) szélesség, magasság ugyanabban a mértékegységben."""
    pos = anchors.astype(float).copy()
    k = len(pos)
    for _ in range(iters):
        moved = False
        for i in range(k):
            for j in range(i + 1, k):
                dx, dy = pos[j] - pos[i]
                ox = (sizes[i, 0] + sizes[j, 0]) / 2 - abs(dx)
                oy = (sizes[i, 1] + sizes[j, 1]) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    step = oy / 2 + 1e-3
                    s = 1.0 if dy >= 0 else -1.0
                    pos[i, 1] -= s * step
                    pos[j, 1] += s * step
                    moved = True
        if not moved:
            break
    return pos


def topic_map(coords, df, names, growth, category, n_highlight=3, height=680, width_hint=1040):
    """Interaktív Plotly-tématérkép.

    coords: map_coords() eredménye; df: id, title, month, topic oszlopokkal (a sorrend egyezik a vektorokéval);
    names: {téma: {"nev", "leiras"}}; growth: téma → növekedés (topic_trends); n_highlight: ennyi leggyorsabban
    növő téma neve kiemelt. Minden hónap külön réteg, a csúszka/lejátszás egy hónap cikkeit emeli ki."""
    import plotly.graph_objects as go

    xy = np.asarray(coords, dtype=float)
    topics = df.topic.astype(int).to_numpy()
    g = {int(t): float(v) for t, v in growth.items()}
    lg = np.clip(np.log2([g[t] for t in topics]), -LOG2_RANGE, LOG2_RANGE)
    months = sorted(df.month.unique())
    e = html.escape

    def hover_text(i):
        r = df.iloc[i]
        t = int(r.topic)
        return [e(textwrap.shorten(str(r.title), 95, placeholder="…")), str(r.month),
                e(textwrap.shorten(names[t]["nev"], 60)), f"{g[t]:.2f}×", str(r.id)]

    custom = [hover_text(i) for i in range(len(df))]
    month_arr = df.month.to_numpy()
    fig = go.Figure()
    for m in months:
        idx = np.flatnonzero(month_arr == m)
        fig.add_trace(go.Scattergl(
            x=xy[idx, 0].round(3), y=xy[idx, 1].round(3), mode="markers", name=m, showlegend=False,
            customdata=[custom[i] for i in idx],
            marker=dict(size=5, color=lg[idx].round(3), colorscale=MAP_COLORSCALE, cmin=-LOG2_RANGE, cmax=LOG2_RANGE,
                        opacity=0.72, line=dict(width=0),
                        showscale=(m == months[0]),
                        colorbar=dict(orientation="h", title=dict(text="a téma növekedése", side="top",
                                                                  font=dict(size=11)),
                                      tickvals=[-2, -1, 0, 1, 2], ticktext=["¼×", "½×", "1×", "2×", "4×+"],
                                      thickness=9, len=0.26, x=1, xanchor="right", y=1.0, yanchor="bottom",
                                      ypad=6, outlinewidth=0, tickfont=dict(size=10))),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · arXiv %{customdata[4]}<br>"
                          "téma: %{customdata[2]} (%{customdata[3]})<extra></extra>"))

    # Témanevek a téma közepén (medián: a kilógó cikkek nem húzzák el); a leggyorsabban növők félkövérek.
    order = sorted(g, key=g.get, reverse=True)
    strong = set(order[:n_highlight])
    tids = [t for t in order if (topics == t).any()]
    centers = np.array([np.median(xy[topics == t], axis=0) for t in tids])
    texts = [f"{_label_name(names[t]['nev'])} {g[t]:.1f}×" for t in tids]
    span = np.ptp(xy, axis=0) * 1.08
    px_x, px_y = span[0] / (width_hint - 20), span[1] / (height - 195)     # adategység / képpont
    sizes = np.array([[(len(s) * (7.4 if t in strong else 6.4) + 10) * px_x, 19 * px_y] for s, t in zip(texts, tids)])
    placed = _place_labels(centers, sizes)
    lo, hi = xy[:, 0].min() - 0.03 * span[0], xy[:, 0].max() + 0.03 * span[0]
    placed[:, 0] = np.clip(placed[:, 0], lo + sizes[:, 0] / 2, hi - sizes[:, 0] / 2)   # ne lógjon ki oldalt
    for t, s, (cx, cy), (lx, ly), (w, h) in zip(tids, texts, centers, placed, sizes):
        moved = bool(np.hypot((lx - cx) / w, (ly - cy) / h) > 0.6)
        # Plotly: nyíllal a szöveg az (ax, ay) ponton áll, a nyíl az (x, y) pontra mutat
        pos = dict(x=cx, y=cy, ax=lx, ay=ly, axref="x", ayref="y") if moved else dict(x=lx, y=ly)
        fig.add_annotation(
            **pos, showarrow=moved, arrowwidth=0.8, arrowcolor=INK, arrowhead=0, standoff=3,
            text=f"<b>{e(s)}</b>" if t in strong else e(s),
            font=dict(size=12 if t in strong else 10.5, color="#1d2127" if t in strong else INK),
            bgcolor="rgba(255,255,255,0.6)", borderpad=1, hovertext=e(names[t]["nev"]),
            name="kiemelt" if t in strong else "tema")

    # Hónapcsúszka: "mind" + hónaponként (a kiválasztott hónap erős, a többi halvány); lejátszás gombbal.
    def frame_opacity(sel):
        return [0.72 if sel is None else (0.95 if m == sel else 0.05) for m in months]

    def frame_size(sel):
        return [5 if sel is None or m != sel else 7 for m in months]

    steps = [None] + months
    fig.frames = [go.Frame(name=s or "mind",
                           data=[go.Scattergl(marker=dict(opacity=o, size=z)) for o, z in zip(frame_opacity(s), frame_size(s))],
                           traces=list(range(len(months)))) for s in steps]
    anim = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    fig.update_layout(
        sliders=[dict(active=0, x=0.08, len=0.9, y=0, yanchor="top", pad=dict(t=8, b=0),
                      currentvalue=dict(prefix="Kiemelt hónap: ", font=dict(size=12, color=INK)),
                      font=dict(size=10, color=INK), tickcolor=INK,
                      steps=[dict(label=s or "mind", method="animate", args=[[s or "mind"], anim]) for s in steps])],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top", pad=dict(t=12, r=6),
                          showactive=False, font=dict(size=11),
                          buttons=[dict(label="▶", method="animate",
                                        args=[months + ["mind"], dict(mode="immediate", fromcurrent=False,
                                                                      frame=dict(duration=900, redraw=True),
                                                                      transition=dict(duration=0))]),
                                   dict(label="■", method="animate", args=[[None], anim])])],
        title=dict(text=f"Tématérkép – arXiv {category}, {len(df)} cikk"
                        "<br><sup>minden pont egy cikk, a hasonló cikkek egymás mellé kerülnek · szín: a téma "
                        "növekedése (utolsó 3 hó / első 3 hó)</sup>", x=0, font=dict(size=16)),
        xaxis=dict(visible=False, range=[xy[:, 0].min() - 0.04 * span[0], xy[:, 0].max() + 0.04 * span[0]]),
        yaxis=dict(visible=False, range=[xy[:, 1].min() - 0.04 * span[1], xy[:, 1].max() + 0.04 * span[1]]),
        height=height, margin=dict(l=10, r=10, t=100, b=90), hovermode="closest", dragmode="pan",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        hoverlabel=dict(font_size=12, align="left"))
    return fig


# A jelentésben (fig.to_html(post_script=MAP_SCRIPT)): kattintásra megnyílik a cikk az arXivon; keskeny képernyőn
# csak a kiemelt témák neve látszik, és a csúszka a gombok mellé húzódik. A Plotly a {plot_id}-t a grafikon azonosítójára cseréli.
MAP_SCRIPT = """
var gd = document.getElementById('{plot_id}');
gd.on('plotly_click', function (ev) {
  var p = ev.points && ev.points[0];
  if (p && p.customdata && /^\\d{4}\\.\\d{4,5}$/.test(p.customdata[4]))
    window.open('https://arxiv.org/abs/' + p.customdata[4], '_blank', 'noopener');
});
(function () {
  var narrow = null;
  function fit() {
    var n = gd.clientWidth < 640;
    if (n === narrow) return;
    narrow = n;
    if (gd._fullTitle === undefined) gd._fullTitle = gd.layout.title.text;
    var upd = {'sliders[0].x': n ? 0.24 : 0.08, 'sliders[0].len': n ? 0.76 : 0.9,
               'title.text': n ? gd._fullTitle.split('<br>')[0] : gd._fullTitle};
    var xr = gd.layout.xaxis.range, w = xr[1] - xr[0];
    (gd.layout.annotations || []).forEach(function (a, i) {
      if (a.name === 'tema') upd['annotations[' + i + '].visible'] = !n;
      if (a.name === 'kiemelt') {
        var f = (((a.showarrow ? a.ax : a.x)) - xr[0]) / w;
        upd['annotations[' + i + '].xanchor'] = !n ? 'center' : f > 0.66 ? 'right' : f < 0.34 ? 'left' : 'center';
      }
    });
    Plotly.relayout(gd, upd);
  }
  fit();
  window.addEventListener('resize', fit);
})();
"""
