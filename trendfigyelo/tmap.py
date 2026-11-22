"""Tématérkép: a cikkek 2D-s térképe a vektoraik alapján, a témák növekedése szerint színezve.

A 768 dimenziós vektorokat UMAP vetíti síkba. A témacímkék kis súllyal részt vesznek a vetítésben (felügyelt UMAP),
ezért egy téma cikkei egy „szigetbe” kerülnek; a szigetek egymáshoz viszonyított helye a vektorokból adódik.
Címkék nélkül a hasonló témák cikkei egyetlen, nehezen olvasható felhőben keverednének.
A szín a cikk témájának növekedése (kék: csökkenő, szürke: változatlan, piros: növekvő részarány), a témák neve a
sziget fölött áll. Egy hónapcsúszkával kiemelhető, hová estek egy adott hónap cikkei.

A hónapcsúszka szándékosan nem Plotly-animáció (frames): a Colab és a Gradio régebbi Plotly-verziója (2.x)
animációnál eltünteti a Scattergl pontjait. A csúszka minden lépése egy egyszerű `restyle`, a lejátszást pedig a
HTML-jelentésben egy kis JavaScript végzi (MAP_SCRIPT).
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
TARGET_WEIGHT = 0.05             # a témacímkék súlya a felügyelt UMAP-ban (0: csak a vektorok; 0,3 fölött pöttyökké esik szét)
SHRINK = 0.45                    # t-SNE tartaléknál: a téma pontjai ennyire húzódnak a téma közepe felé


def _fingerprint(emb, ids, labels, method):
    h = hashlib.sha1(np.ascontiguousarray(emb[:, :8], dtype=np.float32).tobytes())
    h.update("\n".join(map(str, ids)).encode())
    if labels is not None:
        h.update(np.asarray(labels, dtype=np.int64).tobytes())
    return dict(method=method, supervised=labels is not None, n=int(len(emb)), sha1=h.hexdigest())


def _project(emb, labels, method, seed):
    if method == "umap":
        import inspect
        import umap
        kw = dict(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=seed)
        params = inspect.signature(umap.UMAP).parameters
        if "compatibility_layout" in params:                 # újabb umap: a megszokott elrendezés
            kw["compatibility_layout"] = True
        if labels is not None:
            kw["target_weight"] = TARGET_WEIGHT
        return umap.UMAP(**kw).fit_transform(emb, y=None if labels is None else np.asarray(labels))
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = PCA(n_components=min(50, emb.shape[1], len(emb) - 1), random_state=seed).fit_transform(emb)
    xy = TSNE(n_components=2, perplexity=min(30, max(5, (len(emb) - 1) // 4)), init="pca",
              random_state=seed).fit_transform(x)
    if labels is not None:                                    # a t-SNE nem felügyelhető: a témákat összehúzzuk
        lab = np.asarray(labels)
        for t in np.unique(lab):
            m = lab == t
            c = np.median(xy[m], axis=0)
            xy[m] = c + SHRINK * (xy[m] - c)
    return xy


def map_coords(emb, ids, data_dir=None, labels=None, method="auto", seed=42):
    """2D koordináták a cikkekhez (cikk × 2). `labels`: a cikkek témái (ajánlott; nélkülük egyetlen felhő lesz).
    Ha `data_dir` meg van adva, a Drive-on tárolja (terkep.npy + terkep.json), és csak akkor számol újra, ha a
    vektorok, a cikklista, a témák vagy a módszer változott."""
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "tsne"
    want = _fingerprint(np.asarray(emb), list(ids), labels, method)
    if data_dir:
        path, meta_path = os.path.join(data_dir, "terkep.npy"), os.path.join(data_dir, "terkep.json")
        have = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
        if have == want and os.path.exists(path):
            print("Térkép-koordináták betöltve a Drive-ról")
            return np.load(path)
    print(f"Tématérkép számítása ({method.upper()}, {len(emb)} cikk)…")
    xy = np.asarray(_project(np.asarray(emb, dtype=np.float32), labels, method, seed), dtype=float)
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


def topic_map(coords, df, names, growth, category, n_highlight=3, height=700, width_hint=1040, play_button=False):
    """Interaktív Plotly-tématérkép.

    coords: map_coords() eredménye; df: id, title, month, topic oszlopokkal (a sorrend egyezik a vektorokéval);
    names: {téma: {"nev", "leiras"}}; growth: téma → növekedés (topic_trends); n_highlight: ennyi leggyorsabban
    növő téma neve kiemelt. Minden hónap külön réteg; a csúszka egy hónap cikkeit emeli ki.
    play_button: lejátszás gomb (csak a HTML-jelentésben működik, a MAP_SCRIPT-tel)."""
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
            marker=dict(size=4.5, color=lg[idx].round(3), colorscale=MAP_COLORSCALE, cmin=-LOG2_RANGE, cmax=LOG2_RANGE,
                        opacity=0.75, line=dict(width=0),
                        showscale=(m == months[0]),
                        colorbar=dict(orientation="h", title=dict(text="a téma növekedése", side="top",
                                                                  font=dict(size=11)),
                                      tickvals=[-2, -1, 0, 1, 2], ticktext=["¼×", "½×", "1×", "2×", "4×+"],
                                      thickness=9, len=0.26, x=1, xanchor="right", y=1.0, yanchor="bottom",
                                      ypad=6, outlinewidth=0, tickfont=dict(size=10))),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · arXiv %{customdata[4]}<br>"
                          "téma: %{customdata[2]} (%{customdata[3]})<extra></extra>"))

    # Témanevek a sziget fölött (a téma pontjainak felső széle; a kilógó cikkek nem számítanak);
    # a leggyorsabban növők félkövérek. Az ütköző feliratokat függőlegesen széttoljuk.
    order = sorted(g, key=g.get, reverse=True)
    strong = set(order[:n_highlight])
    tids = [t for t in order if (topics == t).any()]
    texts = [f"{_label_name(names[t]['nev'])} {g[t]:.1f}×" for t in tids]
    span = np.ptp(xy, axis=0) * 1.12
    px_x, px_y = span[0] / (width_hint - 20), span[1] / (height - 200)     # adategység / képpont
    sizes = np.array([[(len(s) * (7.4 if t in strong else 6.4) + 10) * px_x, 18 * px_y] for s, t in zip(texts, tids)])
    tops = np.array([[np.median(xy[topics == t, 0]), np.percentile(xy[topics == t, 1], 90)] for t in tids])
    anchors = tops + np.c_[np.zeros(len(tids)), sizes[:, 1] * 0.55]
    placed = _place_labels(anchors, sizes)
    lo, hi = xy[:, 0].min() - 0.05 * span[0], xy[:, 0].max() + 0.05 * span[0]
    placed[:, 0] = np.clip(placed[:, 0], lo + sizes[:, 0] / 2, hi - sizes[:, 0] / 2)   # ne lógjon ki oldalt
    for t, s, (ax_, ay_), (lx, ly), (w, h) in zip(tids, texts, anchors, placed, sizes):
        moved = bool(np.hypot((lx - ax_) / w, (ly - ay_) / h) > 0.6)
        # Plotly: nyíllal a szöveg az (ax, ay) ponton áll, a nyíl az (x, y) pontra mutat
        pos = dict(x=ax_, y=ay_ - h * 0.45, ax=lx, ay=ly, axref="x", ayref="y") if moved else dict(x=lx, y=ly)
        fig.add_annotation(
            **pos, showarrow=moved, arrowwidth=0.8, arrowcolor=INK, arrowhead=0, standoff=2,
            text=f"<b>{e(s)}</b>" if t in strong else e(s),
            font=dict(size=12 if t in strong else 10.5, color="#1d2127" if t in strong else INK),
            bgcolor="rgba(255,255,255,0.6)", borderpad=1, hovertext=e(names[t]["nev"]),
            name="kiemelt" if t in strong else "tema")

    # Hónapcsúszka: "mind" + hónaponként (a kiválasztott hónap erős, a többi halvány). Restyle, nem animáció.
    def step_args(sel):
        return [{"marker.opacity": [0.75 if sel is None else (0.95 if m == sel else 0.06) for m in months],
                 "marker.size": [4.5 if sel is None or m != sel else 7 for m in months]}]

    steps = [None] + months
    slider_x = 0.1 if play_button else 0.0
    fig.update_layout(
        sliders=[dict(active=0, x=slider_x, len=1 - slider_x, y=0, yanchor="top", pad=dict(t=8, b=0),
                      currentvalue=dict(prefix="Kiemelt hónap: ", font=dict(size=12, color=INK)),
                      font=dict(size=10, color=INK), tickcolor=INK,
                      steps=[dict(label=s or "mind", method="restyle", args=step_args(s)) for s in steps])],
        title=dict(text=f"Tématérkép – arXiv {category}, {len(df)} cikk"
                        "<br><sup>minden pont egy cikk, egy sziget egy téma; a hasonló témák egymás közelében · "
                        "szín: a téma növekedése (utolsó 3 hó / első 3 hó)</sup>", x=0, font=dict(size=16)),
        xaxis=dict(visible=False, range=[lo, hi]),
        yaxis=dict(visible=False, range=[xy[:, 1].min() - 0.04 * span[1], xy[:, 1].max() + 0.08 * span[1]]),
        height=height, margin=dict(l=10, r=10, t=100, b=90), hovermode="closest", dragmode="pan",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        hoverlabel=dict(font_size=12, align="left"))
    if play_button:
        fig.update_layout(updatemenus=[dict(
            type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top", pad=dict(t=12, r=6),
            showactive=False, font=dict(size=11), buttons=[dict(label="▶ lejátszás", method="skip", args=[None])])])
    return fig


# A jelentésben (fig.to_html(post_script=MAP_SCRIPT)): kattintásra megnyílik a cikk az arXivon; a „lejátszás” gomb
# végiglépteti a csúszkát; keskeny képernyőn csak a kiemelt témák neve látszik. A {plot_id}-t a Plotly cseréli.
MAP_SCRIPT = """
var gd = document.getElementById('{plot_id}');
gd.on('plotly_click', function (ev) {
  var p = ev.points && ev.points[0];
  if (p && p.customdata && /^\\\\d{4}\\\\.\\\\d{4,5}$/.test(p.customdata[4]))
    window.open('https://arxiv.org/abs/' + p.customdata[4], '_blank', 'noopener');
});
(function () {
  var timer = null;
  function show(i) {
    var st = gd.layout.sliders[0].steps;
    Plotly.restyle(gd, st[i].args[0]);
    Plotly.relayout(gd, {'sliders[0].active': i});
  }
  function stop() { clearInterval(timer); timer = null; Plotly.relayout(gd, {'updatemenus[0].buttons[0].label': '▶ lejátszás'}); }
  gd.on('plotly_buttonclicked', function () {
    if (timer) { stop(); return; }
    var n = gd.layout.sliders[0].steps.length, i = 0;
    Plotly.relayout(gd, {'updatemenus[0].buttons[0].label': '■ állj'});
    timer = setInterval(function () {
      i += 1;
      if (i >= n) { show(0); stop(); return; }
      show(i);
    }, 900);
  });
  gd.on('plotly_sliderchange', function (ev) { if (timer && ev && ev.interaction) stop(); });   // kézi csúsztatás: megáll
})();
(function () {
  var narrow = null;
  function fit() {
    var n = gd.clientWidth < 640;
    if (n === narrow) return;
    narrow = n;
    if (gd._fullTitle === undefined) gd._fullTitle = gd.layout.title.text;
    var upd = {'title.text': n ? gd._fullTitle.split('<br>')[0] : gd._fullTitle};
    var xr = gd.layout.xaxis.range, w = xr[1] - xr[0];
    (gd.layout.annotations || []).forEach(function (a, i) {
      if (a.name === 'tema') upd['annotations[' + i + '].visible'] = !n;
      if (a.name === 'kiemelt') {
        var f = (((a.showarrow ? a.ax : a.x)) - xr[0]) / w;
        upd['annotations[' + i + '].xanchor'] = !n ? 'center' : f > 0.66 ? 'right' : f < 0.34 ? 'left' : 'center';
      }
    });
    if (gd.layout.updatemenus && gd.layout.updatemenus.length) {
      upd['sliders[0].x'] = n ? 0.3 : 0.1;
      upd['sliders[0].len'] = n ? 0.7 : 0.9;
    }
    Plotly.relayout(gd, upd);
  }
  fit();
  window.addEventListener('resize', fit);
})();
"""
