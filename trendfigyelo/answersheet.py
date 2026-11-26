"""Válaszlap a kézi pontozáshoz: a rendszer sorban felteszi a kiválasztott tesztkérdéseket, és a kérdéseket,
a válaszokat (kattintható arXiv-hivatkozásokkal), a modellnek átadott forrásokat, a várt cikk helyét és a
hivatkozás-ellenőrzés ítéleteit egyetlen Markdown-fájlba írja. A pontozó a fájlban tölti ki a „Pont” és
„Megjegyzés” sorokat, és visszatölti `eval/qa_pontozas.md` néven; a parse_sheet ebből állítja elő a
pontozási táblázatot (ugyanazt, amit a qa_pontozas.csv adott).
"""
import re

import pandas as pd

from .evaluate import first_rank
from .rag import detect_language
from .verify import ICONS, summary_counts

ID_RE = re.compile(r"(?<![\w/.])(\d{4}\.\d{4,5})(?![\w])")
SCALE = ("5 = pontos és teljes, minden állítás a forrásokban van, helyes hivatkozásokkal; "
         "4 = lényegében jó, kisebb hiány vagy pontatlanság; 3 = részben jó vagy hiányos; "
         "2 = jelentős hiba vagy alá nem támasztott állítás; 1 = hibás vagy nem a kérdésre válaszol.")

# A pontozás szempontjai a válaszlap elején (hogy több pontozó, ill. több alkalom is azonos mércével dolgozzon).
RUBRIC = """**Mit pontozunk?** A választ a modellnek átadott források (az absztraktok) alapján: azt, amit a modell a kapott cikkekből kihozott. Nem a kérdés „igazi” válaszát és nem a keresést (azt a kiértékelés hit@k-ja méri). A szempontok fontossági sorrendben:

1. **Forráshűség:** minden állítás benne van a hivatkozott cikk absztraktjában, vagy közvetlenül következik belőle (a legfontosabb).
2. **Hivatkozások:** a hivatkozás arra a cikkre mutat, amelyikből az állítás származik.
3. **Válasz a kérdésre:** a kérdés lényegére felel, és amennyire a források engedik, teljes.
4. **Érthetőség és nyelv:** a kérdés nyelvén, érthetően (csak kis súllyal).

| Pont | Mikor |
|---|---|
| **5** | Minden állítás alátámasztott, minden hivatkozás helyes, a kérdés lényegét megválaszolja (amennyire a források engedik). |
| **4** | Lényegében jó, valótlan állítás nincs; de egy kisebb pontatlanság, kissé általánosabb megfogalmazás, vagy kimarad egy fontos részlet, ami a forrásokban benne volt. |
| **3** | Részben jó: a válasz egy része helyes, de hiányos, **vagy** van benne egy alá nem támasztott / túlzó állítás, **vagy** egy hivatkozás rossz cikkre mutat. |
| **2** | Jelentős hiba: több alá nem támasztott állítás, a fő állítás nem a hivatkozott forrásból jön, vagy csak érinti a kérdést. |
| **1** | Hibás, kitalált tartalom, vagy nem a kérdésre válaszol; a hivatkozások nagyrészt rosszak. |

**Határesetek:**
- *A modell jelzi, hogy a források nem elegendőek* (gyakran ★ nélkül, mert a várt cikk nem került át): ha a jelzés igaz, és amit mond, az alátámasztott, **3 pont** (helyesen viselkedett, de nem válaszolt); a megjegyzésbe: „forráshiány”.
- *Igaz, de a forrásokban nem szereplő (általános tudásból vett) állítás:* alá nem támasztottnak számít, legfeljebb **2–3 pont**.
- *Rossz nyelvű válasz* (magyar kérdésre angolul vagy fordítva): egy pont levonás.
- *A hivatkozás-ellenőrzés ítéletei* (✅/⚠️/❌) csak segítség; a te ítéleted számít. Ha nem értesz egyet vele, írd a megjegyzésbe.

**Kitöltés:** a „Pont” sorba egyetlen egész szám kerüljön (1–5; fél pont nincs, a „4,5”-ből 4 lesz). A megjegyzés 1–2 mondat: mi hiányzik vagy mi hibás (több sorba is írható, a következő „---” vonalig tart). Érdemes előbb a választ, aztán a hivatkozott absztraktokat elolvasni, és az állításokat egyenként összevetni; kérdésenként kb. 5–8 perc."""


def arxiv_url(paper_id):
    return f"https://arxiv.org/abs/{paper_id}"


def link_ids(text):
    """A szövegben álló arXiv-azonosítók Markdown-linkké alakítása."""
    return ID_RE.sub(lambda m: f"[{m.group(1)}]({arxiv_url(m.group(1))})", str(text))


def code_questions(questions):
    """Kódok a fájlbeli sorrend szerint, nyelvenként: HU01, HU02…, EN01… (futásonként ugyanazok)."""
    counters, out = {}, []
    for q in questions:
        lang = detect_language(q["q"])
        counters[lang] = counters.get(lang, 0) + 1
        out.append({**q, "lang": lang, "code": f"{lang.upper()}{counters[lang]:02d}"})
    return out


def select_questions(questions, n_hu=6, n_en=6):
    """Nyelvenként n kérdés, a fájlban egyenletesen elosztva (a régi és az új kérdések közül is); ha n legalább
    annyi, ahány kérdés van, mind. Visszaad: kódolt kérdések a fájlbeli sorrendben."""
    coded = code_questions(questions)
    picked = []
    for lang, n in (("hu", n_hu), ("en", n_en)):
        group = [q for q in coded if q["lang"] == lang]
        n = max(0, int(n or 0))
        if n >= len(group):
            picked += group
        elif n:
            step = len(group) / n
            picked += [group[int(i * step + step / 2)] for i in range(n)]
    order = {id(q): i for i, q in enumerate(coded)}
    return sorted(picked, key=lambda q: order[id(q)])


def run_sheet(rag, items, k=None, verify=True, progress=None):
    """A kérdések feltevése sorban. A rag naplójába nem kerülnek (ne jelenjenek meg a közzétételi listában).
    progress(i, n, kód): hívható visszajelzés. Egy kérdés hibája nem állítja meg a többit."""
    out = []
    for i, q in enumerate(items, 1):
        if progress:
            progress(i, len(items), q["code"])
        n_log = len(rag.log)
        row = {**q, "answer": "", "hits": [], "claims": [], "verify_error": "", "error": "", "rank": None}
        try:
            row["rank"] = first_rank(rag.retrieve(q["q"], k=10).id.tolist(), q["expected"])
            rag.ask(q["q"], k=k or rag.top_k, show=False, verify=verify)
            entry = rag.log[-1]
            ids = [s.strip() for s in entry["sources"].split(",") if s.strip()]
            meta = rag.df.drop_duplicates("id").set_index("id").reindex(ids)
            row.update(answer=entry["answer"], claims=entry.get("claims") or [],
                       verify_error=entry.get("verify_error", ""),
                       hits=[{"id": pid, "title": str(meta.loc[pid, "title"]), "month": str(meta.loc[pid, "month"])}
                             for pid in ids])
        except Exception as exc:                                       # pl. kvóta: a többi kérdés még lefuthat
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            del rag.log[n_log:]
        out.append(row)
    return out


def _title(df, pid):
    hit = df.loc[df.id == pid]
    return (str(hit.title.iloc[0]), str(hit.month.iloc[0])) if len(hit) else ("(nincs az adatban)", "")


def to_markdown(rows, meta, df=None):
    """A válaszlap Markdown-szövege. meta: generated, category, n_papers, months, model, top_k."""
    lines = [
        f"# Válaszlap a kézi pontozáshoz – {meta.get('generated', '')}", "",
        f"arXiv {meta.get('category', '')}, {meta.get('n_papers', '')} cikk ({meta.get('months', '')}) · "
        f"nyelvi modell: `{meta.get('model', '')}` · a modell {meta.get('top_k', '')} cikket kapott · "
        f"{len(rows)} kérdés", "",
        "**Teendő:** minden kérdésnél olvasd el a választ és a hivatkozott cikkek absztraktját (a linkeken), és "
        "töltsd ki a **Pont** sort (1–5) és a **Megjegyzés** sort. A kitöltött fájlt töltsd fel a repóba "
        "`eval/qa_pontozas.md` néven; a felület és a jelentés ebből rajzolja a pontozás ábráját.", "",
        RUBRIC, "",
        "Jelek: ★ = a tesztkérdés várt cikke; ✅ alátámasztja · ⚠️ részben · ❌ nem támasztja alá "
        "(a hivatkozás-ellenőrzés, egy második modellhívás ítélete: jelzés, nem bizonyíték).", "",
    ]
    for r in rows:
        lines += ["---", "", f"## {r['code']} · {r['q']}", ""]
        exp = []
        for pid in r["expected"]:
            title, month = _title(df, pid) if df is not None else ("", "")
            exp.append(f"[{pid}]({arxiv_url(pid)})" + (f" – {title}" if title else "") + (f" ({month})" if month else ""))
        rank = r.get("rank")
        given = len(r["hits"]) or r.get("k") or 5
        where = ("a keresés nem hozta az első 10 találat között" if rank is None else
                 f"a keresésben a(z) {rank}. helyen" + (" (a modell megkapta)" if rank <= given
                                                        else " (a modell nem kapta meg)"))
        lines += [f"**Várt cikk:** {'; '.join(exp)} · {where}", ""]
        if r.get("error"):
            lines += [f"> ⚠ A kérdés nem futott le: {r['error']}", ""]
        else:
            lines += ["### Válasz", "", link_ids(r["answer"]).strip(), "", "### A modellnek átadott források", ""]
            lines += [f"{i}. [{h['id']}]({arxiv_url(h['id'])}){' ★' if h['id'] in r['expected'] else ''} – "
                      f"{h['title']} ({h['month']})" for i, h in enumerate(r["hits"], 1)] or ["_nincs_"]
            lines.append("")
            if r.get("verify_error"):
                lines += [f"> ⚠ A hivatkozás-ellenőrzés nem futott le: {r['verify_error']}", ""]
            elif r.get("claims"):
                cnt = summary_counts(r["claims"])
                lines += [f"### Hivatkozások tartalmi ellenőrzése ({ICONS['igen']} {cnt['igen']} · "
                          f"{ICONS['részben']} {cnt['részben']} · {ICONS['nem']} {cnt['nem']})", ""]
                lines += [f"- {ICONS.get(c.get('itelet'), '·')} „{c['mondat']}” "
                          f"[{', '.join(c['ids'])}]" + (f" – {c['indoklas']}" if c.get("indoklas") else "")
                          for c in r["claims"]]
                lines.append("")
        lines += ["### Értékelés", "", "**Pont (1–5):** ", "", "**Megjegyzés:** ", ""]
    return "\n".join(lines)


HEADING = re.compile(r"^## ([A-Z]{2}\d{2,3}) · (.+)$", re.M)


def parse_sheet(text):
    """A kitöltött válaszlapból a pontozás táblázata (question, pont, megjegyzes – „HU01: …” alakban), a
    load_scores-szal azonos oszlopokkal. A pont nélküli kérdések kimaradnak."""
    rows = []
    heads = list(HEADING.finditer(text))
    for i, h in enumerate(heads):
        block = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        m = re.search(r"\*\*Pont \(1–5\):\*\*[ \t]*([1-5])\b", block)
        if not m:
            continue
        note = re.search(r"\*\*Megjegyzés:\*\*[ \t]*(.*?)(?=\n---|\n## |\Z)", block, re.S)
        rows.append({"question": h.group(2).strip(), "pont": int(m.group(1)),
                     "megjegyzes": f"{h.group(1)}: " + (" ".join(note.group(1).split()) if note else "")})
    return pd.DataFrame(rows, columns=["question", "pont", "megjegyzes"])
