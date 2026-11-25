"""Visszakeresés + válaszgenerálás forrásmegjelöléssel."""
import json
import re

import numpy as np
import pandas as pd

from .llm import generate
from .prompts import LANGUAGE_RULES, RAG_PROMPT
from .verify import claims_markdown, verify_claims


def retrieve(df, emb, query_vec, k=5, month_from=None):
    """A kérdéshez legközelebbi k cikk (koszinusz-hasonlóság; a vektorok normáltak).
    month_from ("ÉÉÉÉ-HH"): csak az ettől a hónaptól kezdődő cikkek közül választ; ha ott kevesebb cikk van,
    kevesebb találatot ad, üres időszaknál egyet sem (korábban a hiányt régebbi cikkekkel töltötte fel)."""
    pos = np.arange(len(df))
    if month_from:
        pos = np.flatnonzero(df.month.values >= month_from)
    scores = emb[pos] @ query_vec
    order = pos[np.argsort(-scores)[:k]]
    hits = df.iloc[order].copy()
    hits["score"] = (emb[order] @ query_vec).round(3)
    return hits


HU_CHARS = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")
HU_WORDS = {"az", "és", "hogy", "milyen", "hogyan", "miért", "melyik", "melyek", "mit", "mik", "mi", "vannak",
            "van", "nem", "egy", "vagy", "lehet", "szerint", "kapcsolatban", "mennyire", "hol", "mikor", "kik",
            "kell", "mire", "miben", "között", "alapján", "mostanában", "mely", "miről", "ezek", "ilyen"}


def detect_language(question):
    """'hu', ha a kérdés magyar (ékezetes magyar betű vagy tipikus magyar szó van benne), különben 'en'."""
    if any(ch in HU_CHARS for ch in question):
        return "hu"
    words = set(re.findall(r"[a-z]+", question.lower()))
    return "hu" if words & HU_WORDS else "en"


def resolve_language(question, lang=None):
    """A válasz nyelve: 'hu' vagy 'en', ha kézzel meg van adva; különben (None, 'auto') a kérdésből felismerve.
    A felismerés tévedhet (pl. „MI”, „van der Waals” angol kérdésben), ezért a felületen kézzel is választható."""
    return lang if lang in ("hu", "en") else detect_language(question)


def build_prompt(question, hits, template=RAG_PROMPT, lang=None):
    context = "\n\n".join(f"[{r.id}] {r.title} ({r.month})\n{r.abstract}" for r in hits.itertuples())
    return template.format(context=context, question=question, language_rule=LANGUAGE_RULES[resolve_language(question, lang)])


def check_citations(answer, hits):
    """Hivatkozás-ellenőrzés: csak olyan azonosítóra hivatkozott-e, amit tényleg megkapott (invalid), és hivatkozott-e
    egyáltalán valamelyik átadott forrásra (uncited). Azt nem vizsgálja, hogy az állításokat a forrás alátámasztja-e."""
    cited = set(re.findall(r"\d{4}\.\d{4,5}", answer))
    invalid = sorted(cited - set(hits.id))
    return {"cited": sorted(cited), "invalid": invalid, "uncited": bool(len(hits)) and not (cited & set(hits.id))}


class RAG:
    def __init__(self, df, emb, embed_query, llm, model, top_k=5, verify=True):
        """verify: a hivatkozások tartalmi ellenőrzése egy második modellhívással (verify.py)."""
        self.df, self.emb, self.embed_query = df, emb, embed_query
        self.llm, self.model, self.top_k, self.verify = llm, model, top_k, verify
        self.log = []  # minden kérdés-válasz ide kerül -> dokumentációhoz

    def retrieve(self, question, k=None, month_from=None):
        return retrieve(self.df, self.emb, self.embed_query(question), k or self.top_k, month_from)

    def ask(self, question, k=None, month_from=None, show=True, lang=None, verify=None):
        """lang: 'hu' / 'en' a válasz nyelve; None vagy 'auto': a kérdésből felismerve.
        verify: a hivatkozások tartalmi ellenőrzése (None: a RAG beállítása szerint)."""
        hits = self.retrieve(question, k, month_from)
        language = resolve_language(question, lang)
        if hits.empty:                                   # üres időszak: nincs mit átadni a modellnek
            answer = (f"A megadott időszakban ({month_from} után) nincs cikk az adatbázisban, ezért nem tudok "
                      "forrással alátámasztott választ adni." if language == "hu" else
                      f"There are no papers in the database from {month_from} onward, so I cannot give a sourced answer.")
        else:
            answer = generate(self.llm, self.model, build_prompt(question, hits, lang=lang))
        cit = check_citations(answer, hits)
        claims, verify_error = [], ""
        if (self.verify if verify is None else verify) and not hits.empty:
            claims, verify_error = verify_claims(self.llm, self.model, answer, hits)
        self.log.append({"question": question, "answer": answer, "sources": ", ".join(hits.id), "lang": language, **cit,
                         "claims": claims, "verify_error": verify_error})
        if show:
            print(answer, "\n\nForrások:")
            for r in hits.itertuples():
                print(f"  [{r.id}] {r.score:.3f}  {r.title}")
            if cit["invalid"]:
                print("⚠ Nem kapott forrásra hivatkozott:", cit["invalid"])
            if cit["uncited"]:
                print("⚠ A válasz egyetlen átadott forrásra sem hivatkozik.")
            check = claims_markdown(claims, verify_error)
            if check:
                print(check.replace("**", "").replace("_", "").strip())
        return answer

    def save_log(self, path):
        df = pd.DataFrame(self.log)
        if "claims" in df:                                   # JSON-ként, hogy visszaolvasva is értelmezhető legyen
            df["claims"] = [json.dumps(c, ensure_ascii=False) for c in df["claims"]]
        df.to_csv(path, index=False)
        print(len(self.log), "válasz mentve:", path)
