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
    """A modell JSON-válaszának feldolgozása. Ha egy téma hiányzik vagy hibás a válasz, az első kulcsszó lesz a név."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    out = {}
    for t in topic_ids:
        item = data.get(str(t)) or {}
        out[t] = {"nev": (item.get("nev") or keywords[t].split(", ")[0]).strip(),
                  "leiras": (item.get("leiras") or "").strip()}
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


def plot_trends(share, growth, names, topic_ids, category, path=None, n=6):
    """A leggyorsabban növő n téma havi részaránya, a jelmagyarázatban a témanevekkel."""
    import matplotlib.pyplot as plt
    top = topic_ids[:n]
    smooth = (share[top] * 100).rolling(2, min_periods=1).mean()   # 2 havi mozgóátlag a zaj csökkentésére
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for t in top:
        label = textwrap.fill(names[t]["nev"], 38) + f"  ({growth[t]:.1f}×)"
        ax.plot(smooth.index, smooth[t], marker="o", lw=2.2, label=label)
    ax.set_title(f"A leggyorsabban növő témák – arXiv {category}", fontsize=14, loc="left")
    ax.set_ylabel("A havi cikkek aránya (%)")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9.5, labelspacing=1.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def summarize_trends(llm, model, names, growth, topic_ids, category, n=8):
    blocks = "\n".join(f"- {names[t]['nev']} ({growth[t]:.1f}×): {names[t]['leiras']}" for t in topic_ids[:n])
    return generate(llm, model, SUMMARY_PROMPT.format(cat=category, topics=blocks))
