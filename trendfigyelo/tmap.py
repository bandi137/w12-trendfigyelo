"""Tématérkép: a cikkek 2D-s térképe a vektoraik alapján, a témák növekedése szerint színezve.

A cikkek vektorait (az e5-base modellnél 768 dimenziósak) UMAP vetíti síkba (felügyelet nélkül): a hasonló cikkek egymás mellé kerülnek, és a
cikkek egyetlen összefüggő felhőt alkotnak, ahogy a valóságban is egymásba folynak a témák. Hogy a témák mégis
kivehetők legyenek, megjelenítéskor minden cikk a témája közepe felé húzódik (PULL: a távolság ekkora hányadával).
Így a témák elmosódott foltokként látszanak, a szélük átfedhet, és minden cikk a valódi szomszédai felé esik.
(A UMAP témacímkés felügyelete ehhez nem alkalmas: már a legkisebb súly is éles, elkülönült szigeteket ad.)
A szín a cikk témájának növekedése (kék: csökkenő, szürke: változatlan, piros: növekvő részarány), a témák neve a
folt fölött áll. Egy hónapcsúszkával kiemelhető, hová estek egy adott hónap cikkei.

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
PULL = 0.5                       # a cikk és a témája közepe közti távolság ekkora hányadával húzódik befelé
                                 # (0: a nyers UMAP-felhő; 0,7 fölött már éles, elkülönült foltok)
UMAP_PARAMS = dict(n_neighbors=30, min_dist=0.25, metric="cosine")


DEFAULT_SEED = 42


def _fingerprint(emb, ids, method, seed=DEFAULT_SEED):
    """A tárolt térkép ellenőrző adatai: minden, ami a koordinátákat meghatározza (vektorok, cikklista,
    módszer, paraméterek, véletlen kezdőérték)."""
    h = hashlib.sha1(np.ascontiguousarray(emb, dtype=np.float32).tobytes())     # a teljes vektor (nem csak az eleje)
    h.update("\n".join(map(str, ids)).encode())
    return dict(method=method, params=UMAP_PARAMS if method == "umap" else {}, seed=int(seed), n=int(len(emb)),
                sha1_full=h.hexdigest())


def _project(emb, method, seed):
    if method == "umap":
        import inspect
        import umap
        kw = dict(UMAP_PARAMS, random_state=seed)
        if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:   # újabb umap: a megszokott elrendezés
            kw["compatibility_layout"] = True
        return umap.UMAP(**kw).fit_transform(emb)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = PCA(n_components=min(50, emb.shape[1], len(emb) - 1), random_state=seed).fit_transform(emb)
    return TSNE(n_components=2, perplexity=min(30, max(5, (len(emb) - 1) // 4)), init="pca",
                random_state=seed).fit_transform(x)


def map_coords(emb, ids, data_dir=None, labels=None, method="auto", seed=DEFAULT_SEED):
    """2D koordináták a cikkekhez (cikk × 2), felügyelet nélküli UMAP-pal (ha nincs telepítve: t-SNE).
    Ha `data_dir` meg van adva, a Drive-on tárolja (terkep.npy + terkep.json), és csak akkor számol újra, ha a
    vektorok, a cikklista, a módszer, a paraméterek vagy a véletlen kezdőérték (seed) változott; a témák változása nem számít (a témák csak a megjelenítést
    alakítják). `labels`: nem használt, csak a régebbi notebook-cellák miatt fogadja el."""
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "tsne"
    want = _fingerprint(np.asarray(emb), list(ids), method, seed)
    if data_dir:
        path, meta_path = os.path.join(data_dir, "terkep.npy"), os.path.join(data_dir, "terkep.json")
        have = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
        if have is not None and "seed" not in have:      # a november 25. előtti tárolás mindig az alapértékkel készült
            have = dict(have, seed=DEFAULT_SEED)
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


def _pull(xy, topics, pull):
    """Minden pontot a témája közepe (medián) felé húz a távolság `pull` hányadával: 0 = változatlan felhő."""
    xy = np.array(xy, dtype=float)
    if pull <= 0:
        return xy
    for t in np.unique(topics):
        m = topics == t
        c = np.median(xy[m], axis=0)
        xy[m] = c + (1 - pull) * (xy[m] - c)
    return xy


def _label_name(name, n=30):
    """A térképen rövid név: az angol zárójeles rész nélkül (az egérmutatóra a teljes név látszik)."""
    short = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip() or name
    return textwrap.shorten(short, n, placeholder="…")


def _spread_islands(xy, topics, sep=0.8, max_expand=3.0, iters=500):
    """A szigeteket megjelenítéskor széthúzza, hogy ne torlódjanak egymásra, és a nevüknek is jusson hely.
    1) A szigetek középpontjainak távolságát egyenletesen megnöveli (a szigetek mérete nem változik), így az
       elrendezés változatlan marad; 2) a még átfedő szigeteket (körök: a pontok 75 %-a esik bennük) a
       középpontjaikat összekötő egyenes mentén széttolja. sep: a középpontok legkisebb távolsága a két sugár
       összegének arányában (1 alatt a foltok szélei átfedhetnek, ahogy a valóságban is egymásba folynak)."""
    xy = np.array(xy, dtype=float)
    tids = np.unique(topics)
    masks = [topics == t for t in tids]
    c0 = np.array([np.median(xy[m], axis=0) for m in masks])
    r = np.array([np.percentile(np.linalg.norm(xy[m] - c, axis=1), 75) for m, c in zip(masks, c0)])
    d = np.linalg.norm(c0[:, None] - c0[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)                                    # a legközelebbi szomszéd szigetek
    need = sep * (r + r[nn]) / np.maximum(d[np.arange(len(tids)), nn], 1e-9)
    k = float(np.clip(np.percentile(need, 60), 1.0, max_expand))
    mid = c0.mean(axis=0)
    c = mid + k * (c0 - mid)
    for _ in range(iters):
        moved = False
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                v = c[j] - c[i]
                dist = np.hypot(*v)
                o = sep * (r[i] + r[j]) - dist
                if o > 1e-9:
                    u = v / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    c[i] -= u * o / 2
                    c[j] += u * o / 2
                    moved = True
        if not moved:
            break
    for m, dv in zip(masks, c - c0):
        xy[m] += dv
    return xy


def _layout(xy, topics, pull, spread):
    """Megjelenítési elrendezés. A foltok helyét (a széthúzást) mindig a rögzített PULL összehúzással számolja ki,
    és az így kapott eltolást alkalmazza a `pull` szerint összehúzott pontokra. Korábban a széthúzás a választott
    összehúzás után futott, és mindig ugyanakkora távolságra tolta szét a foltokat: a kép csak átméreteződött, így a
    „Témák összehúzása” csúszka 0 és kb. 0,6 között hatástalan volt. Így a foltok helye nem függ a csúszkától, a
    csúszka csak azt állítja, mennyire tömörek a foltok."""
    xy = np.asarray(xy, dtype=float)
    out = _pull(xy, topics, pull)
    ref = _pull(xy, topics, PULL)
    if spread:
        moved = _spread_islands(ref, topics)
        for t in np.unique(topics):                          # témánként egy eltolás (a széthúzás csak eltol)
            m = topics == t
            out[m] += np.median(moved[m] - ref[m], axis=0)
        ref = moved
    return _landscape_fixed(out, _landscape_basis(ref))


def _landscape_basis(xy):
    """A fekvő tájoláshoz tartozó forgatás (a rögzített elrendezésből, hogy a csúszka ne forgassa el a térképet)."""
    z = xy - np.median(xy, axis=0)
    _, vecs = np.linalg.eigh(np.cov(z.T))
    rot = vecs[:, ::-1].copy()
    if np.median((z @ rot)[:, 0] ** 3) < 0:
        rot[:, 0] *= -1
    return np.median(xy, axis=0), rot


def _landscape_fixed(xy, basis):
    center, rot = basis
    return (xy - center) @ rot


def _landscape(xy):
    """Elforgatja a térképet, hogy a hosszabbik kiterjedése vízszintes legyen (a UMAP tengelyeinek nincs jelentése,
    így a forgatás semmit nem változtat; a képernyőn viszont jobban kitölti a helyet)."""
    z = xy - np.median(xy, axis=0)
    _, vecs = np.linalg.eigh(np.cov(z.T))
    rot = z @ vecs[:, ::-1]                                    # 1. főirány vízszintesen
    if np.median(rot[:, 0] ** 3) < 0:                          # determinisztikus irány (ne tükröződjön futásonként)
        rot[:, 0] *= -1
    return rot


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


QUESTION_NEIGHBOURS = 10     # a kérdés helye ennyi leghasonlóbb cikk helyének súlyozott átlaga
QUESTION_TEMP = 0.02         # a súlyozás élessége (koszinusz-hasonlóságban): kisebb → a legközelebbi cikk dominál


def question_position(xy, scores, k=QUESTION_NEIGHBOURS, temp=QUESTION_TEMP):
    """A kérdés helye a térképen: a hozzá leghasonlóbb k cikk (megjelenítési) helyének súlyozott átlaga.
    Így a kérdés ugyanazokon a megjelenítési lépéseken megy át, mint a cikkek, és nem kell hozzá a UMAP-modell."""
    scores = np.asarray(scores, dtype=float)
    idx = np.argsort(-scores)[:k]
    w = np.exp((scores[idx] - scores[idx].max()) / temp)
    return (w[:, None] * np.asarray(xy)[idx]).sum(axis=0) / w.sum()


def map_question(rag, entry):
    """Egy kérdés a térképhez a RAG naplójából (rag.log eleme) vagy egy kérdésszövegből: a kérdés szövege, a
    hasonlósága minden cikkhez, és a modellnek átadott források azonosítói."""
    if isinstance(entry, str):
        entry = {"question": entry, "sources": ", ".join(rag.retrieve(entry).id)}
    sources = [i.strip() for i in str(entry.get("sources", "")).split(",") if i.strip()]
    return {"text": entry["question"], "scores": rag.emb @ rag.embed_query(entry["question"]), "sources": sources}


MAX_POINTS = 6000      # ennél több pontnál a böngésző (főleg mobilon) lassul, a térkép pedig összefolyik


def display_sample(months, max_points=MAX_POINTS, seed=0):
    """A képre kerülő cikkek sorszámai (növekvő sorrendben): ha több a cikk max_points-nál, hónaponként
    arányos, determinisztikus véletlen minta, így a havi arányok és a csúszka kiemelése hű marad."""
    months = np.asarray(months)
    n = len(months)
    if max_points is None or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    keep = []
    for m in sorted(set(months.tolist())):
        idx = np.flatnonzero(months == m)
        k = max(1, int(round(len(idx) * max_points / n)))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def topic_map(coords, df, names, growth, category, n_highlight=3, height=None, width_hint=1040, play_button=False,
              pull=PULL, spread=True, questions=None, active_question=0, max_points=MAX_POINTS, seed=0):
    """Interaktív Plotly-tématérkép.

    coords: map_coords() eredménye; df: id, title, month, topic oszlopokkal (a sorrend egyezik a vektorokéval);
    names: {téma: {"nev", "leiras"}}; growth: téma → növekedés (topic_trends); n_highlight: ennyi leggyorsabban
    növő téma neve kiemelt. Minden hónap külön réteg; a csúszka egy hónap cikkeit emeli ki.
    play_button: lejátszás gomb (csak a HTML-jelentésben működik, a MAP_SCRIPT-tel).
    pull: mennyire húzódjanak a cikkek a témájuk közepe felé (0: nyers felhő; lásd PULL).
    spread: a foltok széthúzása, hogy ne torlódjanak és a nevük elférjen (csak a megjelenítést érinti).
    height: a kép magassága; ha nincs megadva, a térkép arányából adódik (a két tengely léptéke azonos).
    questions: kérdések a térképen (map_question() elemei): csillag a kérdés helyén, bekarikázva a modellnek
    átadott források, vonallal a kérdéshez kötve, számozott címkékkel. Több kérdésnél legördülő menüből
    választható, melyik látszik; active_question: kezdetben melyik (sorszám vagy "last").
    max_points: legfeljebb ennyi pont kerül a képre (hónaponként arányos véletlen minta, rögzített seed-del);
    a térkép elrendezése, a témák helye és a nevek mindig az összes cikkből számolódnak, a kérdés forrásai
    mindig látszanak. None: minden cikk."""
    import plotly.graph_objects as go

    xy = np.asarray(coords, dtype=float)
    topics = df.topic.astype(int).to_numpy()
    g = {int(t): float(v) for t, v in growth.items()}
    lg = np.clip(np.log2([g[t] for t in topics]), -LOG2_RANGE, LOG2_RANGE)
    months = sorted(df.month.unique())
    e = html.escape

    order = sorted(g, key=g.get, reverse=True)
    strong = set(order[:n_highlight])
    tids = [t for t in order if (topics == t).any()]
    texts = [f"{_label_name(names[t]['nev'])} {g[t]:.1f}×" for t in tids]
    label_px = {t: (len(s) * (7.4 if t in strong else 6.4) + 10, 18) for s, t in zip(texts, tids)}
    xy = _layout(xy, topics, pull, spread)
    span = np.ptp(xy, axis=0) * np.array([1.1, 1.14])
    scale = span[0] / (width_hint - 20)                           # adategység / képpont, mindkét tengelyen
    if height is None:
        height = int(np.clip(span[1] / scale + 190, 560, 900))

    def hover_text(i):
        r = df.iloc[i]
        t = int(r.topic)
        return [e(textwrap.shorten(str(r.title), 95, placeholder="…")), str(r.month),
                e(textwrap.shorten(names[t]["nev"], 60)), f"{g[t]:.2f}×", str(r.id)]

    shown = display_sample(df.month.to_numpy(), max_points, seed)
    month_arr = df.month.to_numpy()
    fig = go.Figure()
    for m in months:
        idx = shown[month_arr[shown] == m]
        fig.add_trace(go.Scattergl(
            x=xy[idx, 0].round(3), y=xy[idx, 1].round(3), mode="markers", name=m, showlegend=False,
            customdata=[hover_text(i) for i in idx],
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
    # a leggyorsabban növők félkövérek. A még ütköző feliratokat függőlegesen széttoljuk.
    sizes = np.array([[label_px[t][0] * scale, label_px[t][1] * scale] for t in tids])
    tops = np.array([[np.median(xy[topics == t, 0]), np.percentile(xy[topics == t, 1], 90)] for t in tids])
    anchors = tops + np.c_[np.zeros(len(tids)), sizes[:, 1] * 0.55]
    placed = _place_labels(anchors, sizes)
    lo, hi = xy[:, 0].min() - 0.045 * span[0], xy[:, 0].max() + 0.045 * span[0]
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
                 "marker.size": [4.5 if sel is None or m != sel else 7 for m in months]},
                list(range(len(months)))]                     # csak a hónap-rétegek (a kérdés-rétegek nem)

    steps = [None] + months
    y0, y1 = xy[:, 1].min() - 0.03 * span[1], xy[:, 1].max() + 0.05 * span[1]
    slider_x = 0.1 if play_button else 0.0
    fig.update_layout(
        sliders=[dict(active=0, x=slider_x, len=1 - slider_x, y=0, yanchor="top", pad=dict(t=8, b=0),
                      currentvalue=dict(prefix="Kiemelt hónap: ", font=dict(size=12, color=INK)),
                      font=dict(size=10, color=INK), tickcolor=INK,
                      steps=[dict(label=s or "mind", method="restyle", args=step_args(s)) for s in steps])],
        title=dict(text=f"Tématérkép – arXiv {category}, {len(df)} cikk"
                        + (f" (a képen {len(shown)} pont, véletlen minta)" if len(shown) < len(df) else "") +
                        "<br><sup>minden pont egy cikk, a hasonló cikkek egymás mellett; a cikkek a témájuk közepe felé húzva · "
                        "szín: a téma növekedése (utolsó 3 hó / első 3 hó részaránya, simítva)</sup>", x=0, font=dict(size=16)),
        xaxis=dict(visible=False, range=[lo, hi]),
        yaxis=dict(visible=False, range=[y0, y1],
                   scaleanchor="x", scaleratio=1),                    # a szigetek alakja minden szélességen megmarad
        height=height, margin=dict(l=10, r=10, t=100, b=90), hovermode="closest", dragmode="pan",
        meta=dict(aspect=round(float(np.ptp(xy[:, 1]) * 1.08 / (hi - lo)), 4), mb=90,
                  xr=[round(float(lo), 4), round(float(hi), 4)], yr=[round(float(y0), 4), round(float(y1), 4)]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        hoverlabel=dict(font_size=12, align="left"))
    menus = []
    if play_button:
        menus.append(dict(
            type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top", pad=dict(t=12, r=6),
            showactive=False, font=dict(size=11), buttons=[dict(label="▶ lejátszás", method="skip", args=[None])]))
    if questions:
        _add_questions(fig, xy, df, hover_text, questions, menus,
                       active=len(questions) - 1 if active_question == "last" else int(active_question))
        fig.update_layout(title_text=fig.layout.title.text.replace(
            "</sup>", " · ★ a kérdés, ○ a modellnek átadott források</sup>"))
    if menus:
        fig.update_layout(updatemenus=menus)
    return fig


QUESTION_COLORS = dict(star="#f5b301", ink="#1d2127")


def _add_questions(fig, xy, df, hover_text, questions, menus, active=0):
    """Kérdésenként három réteg és számozott címkék: vonalak a kérdéstől a forrásokig, bekarikázott források,
    legfelül csillag a kérdés helyén, és forrásonként egy „1”, „2”, … címke a kérdés köré legyezve, vékony vonallal a saját
    pontjához kötve. A források gyakran egy kis foltban, a csillag alatt vannak (a kérdés helye a leghasonlóbb cikkekből
    adódik), ezért külön karikaként alig látszanak; a címkék így is megkülönböztetik őket.
    A rétegek SVG-rétegek (nem WebGL): mindig a pontok fölött vannak, és minden Plotly-verzió kirajzolja az üres
    karikát. Egy kérdésnél mind látszik; többnél az `active` sorszámú, és egy legördülő menü vált köztük."""
    import plotly.graph_objects as go
    e = html.escape
    pos_of = {str(i): n for n, i in enumerate(df.id)}
    n_base = len(fig.data)
    labels = {}                                               # kérdés sorszáma → a címke-annotációk indexei
    for qi, q in enumerate(questions):
        qx, qy = question_position(xy, q["scores"])
        src = [pos_of[i] for i in q["sources"] if i in pos_of]
        show = qi == active
        text = e(textwrap.shorten(str(q["text"]), 120, placeholder="…"))
        lx, ly = [], []
        for i in src:
            lx += [qx, xy[i, 0], None]
            ly += [qy, xy[i, 1], None]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name="q-vonal", showlegend=False, hoverinfo="skip",
                                 visible=show, line=dict(color=QUESTION_COLORS["ink"], width=1.2)))
        fig.add_trace(go.Scatter(
            x=xy[src, 0].round(3), y=xy[src, 1].round(3), mode="markers", name="q-forras", showlegend=False,
            visible=show, customdata=[[n + 1] + hover_text(i) for n, i in enumerate(src)],
            marker=dict(symbol="circle-open", size=14, color=QUESTION_COLORS["ink"], line=dict(width=2.2)),
            hovertemplate="<b>%{customdata[0]}. forrás:</b> %{customdata[1]}<br>%{customdata[2]} · arXiv "
                          "%{customdata[5]}<br>téma: %{customdata[3]}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[round(float(qx), 3)], y=[round(float(qy), 3)], mode="markers", name="q-csillag", showlegend=False,
            visible=show, customdata=[[text]],
            marker=dict(symbol="star", size=20, color=QUESTION_COLORS["star"],
                        line=dict(width=1.5, color=QUESTION_COLORS["ink"])),
            hovertemplate="<b>Kérdés:</b> %{customdata[0]}<extra></extra>"))
        labels[qi] = []
        for n, i in enumerate(src):                           # címkék legyezve a kérdés köré (képpontban)
            ang = np.deg2rad(-90 + 360 * n / max(len(src), 1) + 20)
            labels[qi].append(len(fig.layout.annotations))
            fig.add_annotation(
                x=float(xy[i, 0]), y=float(xy[i, 1]), ax=float(58 * np.cos(ang)), ay=float(58 * np.sin(ang)),
                text=f"<b>{n + 1}</b>", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=QUESTION_COLORS["ink"],
                standoff=7, font=dict(size=11, color=QUESTION_COLORS["ink"]), bgcolor="rgba(255,255,255,0.92)",
                bordercolor=QUESTION_COLORS["ink"], borderwidth=1, borderpad=2, visible=show, name="q-cimke",
                hovertext=e(f"{n + 1}. forrás: {textwrap.shorten(str(df.title.iloc[i]), 90, placeholder='…')}"))
    if len(questions) > 1:
        n_q = 3 * len(questions)

        def args(sel):
            vis = [True] * n_base + [sel is not None and k // 3 == sel for k in range(n_q)]
            ann = {f"annotations[{j}].visible": qi == sel for qi, idx in labels.items() for j in idx}
            return [{"visible": vis}, ann]
        buttons = [dict(label=f"{k + 1}. {textwrap.shorten(str(q['text']), 55, placeholder='…')}", method="update",
                        args=args(k)) for k, q in enumerate(questions)]
        buttons.append(dict(label="(egyik kérdés sem)", method="update", args=args(None)))
        menus.append(dict(type="dropdown", direction="down", x=0, y=1, xanchor="left", yanchor="top", active=active,
                          showactive=True, font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                          pad=dict(t=2, l=2), buttons=buttons))


# A jelentésben (fig.to_html(post_script=MAP_SCRIPT)): kattintásra megnyílik a cikk az arXivon; a „lejátszás” gomb
# végiglépteti a csúszkát; keskeny képernyőn csak a kiemelt témák neve látszik; sötét módban sötét feliratdobozok
# világos szöveggel. A {plot_id}-t a Plotly cseréli (egyszerű szövegcsere). Nyers (r""") szöveg: a JavaScript
# pontosan azt kapja, ami itt áll (korábban a \\d duplán escape-elődött, és a hivatkozás nem működött).
ARXIV_ID_JS = r"/^\d{4}\.\d{4,5}$/"
MAP_SCRIPT = r"""
var gd = document.getElementById('{plot_id}');
gd.on('plotly_click', function (ev) {
  var p = ev.points && ev.points[0];
  if (p && p.customdata && ARXIV_ID.test(p.customdata[4]))
    window.open('https://arxiv.org/abs/' + p.customdata[4], '_blank', 'noopener');
});
(function () {
  var timer = null;
  function show(i) {
    var st = gd.layout.sliders[0].steps;
    Plotly.restyle(gd, st[i].args[0], st[i].args[1]);
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
  var meta = gd.layout.meta || {}, aspect = meta.aspect || 0.7;
  var fullTitle = gd.layout.title.text, fullHeight = gd.layout.height;
  var dark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  // A teljes elrendezést minden méretváltozáskor újraszámolja, a szélességet is: a Plotly saját átméretezése
  // ehhez képest később fut, ezért korábban keskenyre váltáskor a régi szélesség maradt, és a térkép üres lett.
  function fit() {
    var width = gd.parentElement ? gd.parentElement.clientWidth : gd.clientWidth;
    if (!width) return;
    var n = width < 640, d = !!(dark && dark.matches);
    var xr = meta.xr || gd.layout.xaxis.range, w = xr[1] - xr[0];
    var upd = {'title.text': width < 900 ? fullTitle.split('<br>')[0] : fullTitle, width: width,
               height: n ? Math.min(fullHeight, Math.round((width - 20) * aspect + 170)) : fullHeight,
               'font.color': d ? '#b4bac2' : '#5f6670',
               'sliders[0].font.color': d ? '#b4bac2' : '#5f6670',
               'sliders[0].currentvalue.font.color': d ? '#b4bac2' : '#5f6670'};
    if (meta.xr) { upd['xaxis.range'] = meta.xr.slice(); upd['yaxis.range'] = meta.yr.slice(); }
    (gd.layout.annotations || []).forEach(function (a, i) {
      var k = 'annotations[' + i + ']', strong = a.name === 'kiemelt';
      if (a.name === 'q-cimke') {                             // forráscímke: csak a színe igazodik a módhoz
        upd[k + '.bgcolor'] = d ? 'rgba(24,26,31,0.92)' : 'rgba(255,255,255,0.92)';
        upd[k + '.font.color'] = upd[k + '.bordercolor'] = upd[k + '.arrowcolor'] = d ? '#f2f4f6' : '#1d2127';
        return;
      }
      if (a.name === 'tema') upd[k + '.visible'] = !n;
      // a felirat ne lógjon ki oldalt: a szélekhez közel a szél felé igazodik (becsült szövegszélességgel)
      var px = ((a.showarrow ? a.ax : a.x) - xr[0]) / w * (width - 20) + 10;
      var half = String(a.text).replace(/<[^>]+>/g, '').length * (strong ? 3.8 : 3.3) + 4;
      upd[k + '.xanchor'] = px - half < 0 ? 'left' : px + half > width ? 'right' : 'center';
      upd[k + '.bgcolor'] = d ? 'rgba(24,26,31,0.78)' : 'rgba(255,255,255,0.6)';
      upd[k + '.font.color'] = strong ? (d ? '#f2f4f6' : '#1d2127') : (d ? '#c9ced5' : '#5f6670');
      upd[k + '.arrowcolor'] = d ? '#9aa1aa' : '#5f6670';
    });
    (gd.layout.updatemenus || []).forEach(function (m, k) {
      if (m.type === 'buttons') {                             // a csúszka a lejátszás gomb mellől induljon
        var sx = Math.min(0.35, 115 / width);
        upd['sliders[0].x'] = sx;
        upd['sliders[0].len'] = 1 - sx;
      } else if (m.type === 'dropdown') {                     // kérdésválasztó: keskenyen a csúszka alatt
        var extra = n ? 48 : 0, mb = (meta.mb || 90) + extra;
        upd.height += extra;
        upd['margin.b'] = mb;
        var plotH = upd.height - (gd.layout.margin.t || 100) - mb;
        upd['updatemenus[' + k + '].y'] = n ? -(mb - 40) / plotH : 1;
        upd['updatemenus[' + k + '].bgcolor'] = d ? 'rgba(24,26,31,0.92)' : 'rgba(255,255,255,0.88)';
        upd['updatemenus[' + k + '].font.color'] = d ? '#e6e9ed' : '#1d2127';
        upd['updatemenus[' + k + '].bordercolor'] = d ? '#5a616b' : '#c8ccd2';
      }
    });
    gd.style.height = upd.height + 'px';                     // a to_html rögzített div-magassága is igazodjon
    if (gd.parentElement && gd.parentElement.style.height) gd.parentElement.style.height = upd.height + 'px';
    Plotly.relayout(gd, upd);
    var ink = d ? '#f2f4f6' : '#1d2127', lines = [], rings = [], stars = [];
    gd.data.forEach(function (t, i) {
      if (t.name === 'q-vonal') lines.push(i);
      if (t.name === 'q-forras') rings.push(i);
      if (t.name === 'q-csillag') stars.push(i);
    });
    if (lines.length) Plotly.restyle(gd, {'line.color': ink}, lines);
    if (rings.length) Plotly.restyle(gd, {'marker.color': ink}, rings);
    if (stars.length) Plotly.restyle(gd, {'marker.line.color': ink}, stars);
  }
  var t = null;
  function later() { clearTimeout(t); t = setTimeout(fit, 120); }
  fit();
  window.addEventListener('resize', later);
  if (dark && dark.addEventListener) dark.addEventListener('change', fit);
})();
""".replace("ARXIV_ID.test", ARXIV_ID_JS + ".test")
