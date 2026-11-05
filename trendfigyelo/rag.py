"""Visszakeresés + válaszgenerálás forrásmegjelöléssel."""
import re

import numpy as np
import pandas as pd

from .llm import generate
from .prompts import RAG_PROMPT


def retrieve(df, emb, query_vec, k=5, month_from=None):
    """A kérdéshez legközelebbi k cikk (koszinusz-hasonlóság; a vektorok normáltak)."""
    scores = emb @ query_vec
    if month_from:
        scores = np.where(df.month.values >= month_from, scores, -1)
    idx = np.argsort(-scores)[:k]
    hits = df.iloc[idx].copy()
    hits["score"] = scores[idx].round(3)
    return hits


def build_prompt(question, hits, template=RAG_PROMPT):
    context = "\n\n".join(f"[{r.id}] {r.title} ({r.month})\n{r.abstract}" for r in hits.itertuples())
    return template.format(context=context, question=question)


def check_citations(answer, hits):
    """Csak olyan azonosítóra hivatkozott-e, amit tényleg megkapott?"""
    cited = set(re.findall(r"\d{4}\.\d{4,5}", answer))
    return {"cited": sorted(cited), "invalid": sorted(cited - set(hits.id))}


class RAG:
    def __init__(self, df, emb, embed_query, llm, model, top_k=5):
        self.df, self.emb, self.embed_query = df, emb, embed_query
        self.llm, self.model, self.top_k = llm, model, top_k
        self.log = []  # minden kérdés-válasz ide kerül -> dokumentációhoz

    def retrieve(self, question, k=None, month_from=None):
        return retrieve(self.df, self.emb, self.embed_query(question), k or self.top_k, month_from)

    def ask(self, question, k=None, month_from=None, show=True):
        hits = self.retrieve(question, k, month_from)
        answer = generate(self.llm, self.model, build_prompt(question, hits))
        cit = check_citations(answer, hits)
        self.log.append({"question": question, "answer": answer, "sources": ", ".join(hits.id), **cit})
        if show:
            print(answer, "\n\nForrások:")
            for r in hits.itertuples():
                print(f"  [{r.id}] {r.score:.3f}  {r.title}")
            if cit["invalid"]:
                print("⚠ Nem kapott forrásra hivatkozott:", cit["invalid"])
        return answer

    def save_log(self, path):
        pd.DataFrame(self.log).to_csv(path, index=False)
        print(len(self.log), "válasz mentve:", path)
