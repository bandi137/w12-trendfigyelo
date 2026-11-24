"""Témák feltárása, havi trendek, LLM-es témaelnevezés, táblázat és grafikon."""
import json
import os
import re
import textwrap

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from .llm import generate
from .prompts import NAMING_PROMPT, SUMMARY_PROMPT

# Az absztraktokban mindenütt előforduló szavak, amelyek semmit nem mondanak a témáról
DOMAIN_STOP_WORDS = {
    "model", "models", "language", "languages", "large", "llm", "llms", "paper", "propose", "proposed",
    "approach", "approaches", "method", "methods", "results", "result", "show", "shows", "based", "using",
    "task", "tasks", "performance", "work", "study", "new", "novel", "existing", "data", "framework",
    "experiments", "demonstrate", "significantly", "improve", "improves", "state", "art", "introduce",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def find_topics(emb, texts, n_topics=25, seed=42, n_keywords=6, min_df=5, max_df=0.3):
    """KMeans-témák + kulcsszavak class-based TF-IDF-fel (a BERTopic képlete).

    Egy szó csak akkor lehet kulcsszó, ha legalább `min_df` cikkben előfordul (így nem kerülnek be
    egyetlen cikk ritka szavai), és legfeljebb a cikkek `max_df` hányadában (így kiesnek az általános szavak).
    """
    labels = KMeans(n_clusters=n_topics, n_init=10, random_state=seed).fit_predict(emb)
    cv = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=min_df, max_df=max_df,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
    X = cv.fit_transform(texts)                                       # cikk × szó
    M = sp.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(n_topics, len(labels)))
    C = np.asarray((M @ X).todense(), dtype=float)                    # téma × szó előfordulás
    tf = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + C.sum() / n_topics / np.maximum(C.sum(axis=0), 1))
    score = tf * idf
    terms = np.array(cv.get_feature_names_out())
    keywords = {t: ", ".join(terms[np.argsort(-score[t])[:n_keywords]]) for t in range(n_topics)}
    return labels, keywords


def topic_trends(months, labels, window=3):
    """Havi részarány témánként + növekedés: utolsó `window` hónap átlaga / első `window` hónap átlaga."""
    share = pd.crosstab(months, labels, normalize="index").sort_index()
    first, last = share.iloc[:window].mean(), share.iloc[-window:].mean()
    growth = ((last + 0.005) / (first + 0.005)).sort_values(ascending=False)
    return share, growth


# --- Témaelnevezés --------------------------------------------------------------------------------

def build_naming_prompt(topic_ids, keywords, examples, cat):
    blocks = []
    for t in topic_ids:
        titles = "\n".join(f"  - {title}" for title in examples[t])
        blocks.append(f"Téma {t}\nKulcsszavak: {keywords[t]}\nCikkcímek:\n{titles}")
    return NAMING_PROMPT.format(cat=cat, topics="\n\n".join(blocks))


def parse_names(text, topic_ids, keywords):
    """A modell JSON-válaszának feldolgozása. Ha a válasz nem szótár, egy téma hiányzik, vagy a név/leírás nem szöveg
    (pl. szám, lista), az első kulcsszó lesz a név, a leírás üres – kivétel helyett."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    def text_of(v):
        return v.strip() if isinstance(v, str) else ""

    out = {}
    for t in topic_ids:
        item = data.get(str(t))
        item = item if isinstance(item, dict) else {}
        out[t] = {"nev": text_of(item.get("nev")) or keywords[t].split(", ")[0].strip(),
                  "leiras": text_of(item.get("leiras"))}
    return out


def name_topics(llm, model, df, keywords, topic_ids, category, data_dir, n_examples=5):
    """Egyetlen LLM-hívás az összes témára. Az eredmény a Drive-ra mentődik (temanevek.json), és amíg
    a kulcsszavak nem változnak, nem kérdezi újra a modellt."""
    path = os.path.join(data_dir, "temanevek.json")
    kw_now = {str(t): keywords[t] for t in topic_ids}
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    if cache.get("keywords") == kw_now:
        print("Témanevek betöltve a Drive-ról")
        return {int(k): v for k, v in cache["names"].items()}
    examples = {t: df[df.topic == t].sort_values("published", ascending=False).title.head(n_examples).tolist()
                for t in topic_ids}
    prompt = build_naming_prompt(topic_ids, keywords, examples, category)
    names = parse_names(generate(llm, model, prompt, json_output=True, temperature=0.2), topic_ids, keywords)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"keywords": kw_now, "names": {str(t): v for t, v in names.items()}}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(data_dir, "prompt_temanevek.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)  # a dokumentációhoz
    return names


# --- Megjelenítés ---------------------------------------------------------------------------------

def trend_table(df, topic_ids, names, keywords, growth):
    return pd.DataFrame({
        "topic":       topic_ids,
        "név":         [names[t]["nev"] for t in topic_ids],
        "leírás":      [names[t]["leiras"] for t in topic_ids],
        "kulcsszavak": [keywords[t] for t in topic_ids],
        "cikkek":      [int((df.topic == t).sum()) for t in topic_ids],
        "növekedés":   [round(float(growth[t]), 2) for t in topic_ids],
    })


def style_table(table, n=8):
    return (table.head(n).style.hide(axis="index")
            .format({"növekedés": "{:.2f}"})
            .set_properties(**{"text-align": "left", "vertical-align": "top", "white-space": "normal"})
            .set_properties(subset=["név"], **{"font-weight": "bold", "min-width": "180px"})
            .set_properties(subset=["leírás"], **{"min-width": "320px"})
            .set_properties(subset=["kulcsszavak"], **{"opacity": "0.7", "font-size": "0.9em", "max-width": "220px"})
            .bar(subset=["növekedés"], color="#9ecae1", vmin=0)
            .set_table_styles([{"selector": "th", "props": "text-align: left;"}]))


def plot_trends(share, growth, names, topic_ids, category, path=None, n=6, window=3):
    """A leggyorsabban növő n téma havi részaránya (a webes jelentés grafikonjának álló változata).

    A legjobb 3 téma színes és vastag (mögötte halványan a nyers havi érték), a többi halványszürke; a nevek a
    vonalak végén állnak; az első és az utolsó `window` hónap sávval jelölve (a növekedés ezek átlagának hányadosa)."""
    import matplotlib.pyplot as plt
    from .report import GREY, N_HIGHLIGHT, PALETTE, short_name, spread_labels
    top = list(topic_ids[:n])
    raw = share[top] * 100
    smooth = raw.rolling(2, min_periods=1).mean()               # 2 havi mozgóátlag a zaj csökkentésére
    x = list(range(len(smooth)))
    ymax = float(max(smooth.max().max(), raw[top[:N_HIGHLIGHT]].max().max()))
    fig, ax = plt.subplots(figsize=(12, 5.5 + 0.12 * max(0, len(top) - 8)))
    for lo, hi, label in ((-0.5, window - 0.5, "bázis"), (len(x) - window - 0.5, len(x) - 0.5, "összehasonlítás")):
        ax.axvspan(lo, hi, color="grey", alpha=0.10, lw=0)
        ax.text((lo + hi) / 2, ymax * 1.12, f"{label} ({window} hó)", ha="center", va="top", fontsize=9, color="#666")
    for rank in range(len(top) - 1, -1, -1):                    # a kiemeltek kerüljenek legfelülre
        t = top[rank]
        strong = rank < N_HIGHLIGHT
        color = PALETTE[rank] if strong else GREY
        if strong:
            ax.plot(x, raw[t], "o", ms=3.5, color=color, alpha=0.35, lw=0)
        ax.plot(x, smooth[t], color=color, lw=2.8 if strong else 1.3, alpha=1 if strong else 0.75)
    ys = spread_labels([float(smooth[t].iloc[-1]) for t in top], gap=0.055 * ymax, top=ymax * 1.05)
    for rank, t in enumerate(top):
        strong = rank < N_HIGHLIGHT
        ax.annotate(f"{short_name(names[t]['nev'])}  {growth[t]:.1f}×", (x[-1], ys[rank]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10 if strong else 9,
                    fontweight="bold" if strong else "normal", color=PALETTE[rank] if strong else "#666")
    ax.set_title(f"A leggyorsabban növő témák – arXiv {category}", fontsize=14, loc="left", pad=22)
    ax.text(0, 1.015, f"vonal: 2 havi mozgóátlag · pontok: havi nyers érték · a szám a növekedés "
                      f"(utolsó {window} hó / első {window} hó)", transform=ax.transAxes, fontsize=9, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(list(smooth.index), rotation=45, ha="right")
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel("A havi cikkek aránya (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.72)                              # hely a feliratoknak
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def summarize_trends(llm, model, names, growth, topic_ids, category, n=8):
    blocks = "\n".join(f"- {names[t]['nev']} ({growth[t]:.1f}×): {names[t]['leiras']}" for t in topic_ids[:n])
    return generate(llm, model, SUMMARY_PROMPT.format(cat=category, topics=blocks))
