"""A hivatkozások tartalmi ellenőrzése: alátámasztja-e a hivatkozott cikk absztraktja az állítást.

A válasz hivatkozást tartalmazó mondatait (állításait) egy második modellhívás a hivatkozott cikkek
absztraktjával veti össze, és mindegyikről ítéletet ad: alátámasztja / részben / nem támasztja alá.
Korlát: az ellenőrző ugyanaz a nyelvi modell, és csak az absztraktot látja, a teljes cikket nem; az ítélet
jelzés a felhasználónak, nem bizonyíték.
"""
import json
import re

from .llm import generate
from .prompts import VERIFY_PROMPT

ID_RE = re.compile(r"\d{4}\.\d{4,5}")
VERDICTS = {"igen": "alátámasztja", "részben": "részben", "nem": "nem támasztja alá"}
ICONS = {"igen": "✅", "részben": "⚠️", "nem": "❌"}


def split_claims(answer):
    """Az állítások: a válasz hivatkozást tartalmazó mondatai, a bennük hivatkozott azonosítókkal.
    A mondat utáni hivatkozást ("… állítás. [2501.01234]") az előző mondathoz számítja; a hivatkozás nélküli
    mondatokat (bevezetés, összegzés) kihagyja."""
    claims = []
    for line in str(answer).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()   # felsorolásjel
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[^\s\[(])", line)
        merged = []
        for p in parts:
            if merged and re.fullmatch(r"[\s\[\](),;.]*(?:\d{4}\.\d{4,5}[\s\[\](),;.]*)+", p):
                merged[-1] += " " + p                                       # csak hivatkozás: az előző mondaté
            else:
                merged.append(p)
        for sentence in merged:
            ids = list(dict.fromkeys(ID_RE.findall(sentence)))
            if ids:
                text = re.sub(r"\s*[\[(][^\])]*\d{4}\.\d{4,5}[^\])]*[\])]", "", sentence).strip()
                claims.append({"mondat": text or sentence.strip(), "ids": ids})
    return claims


def build_verify_prompt(claims, hits):
    used = {i for c in claims for i in c["ids"]}
    sources = "\n\n".join(f"[{r.id}] {r.title}\n{r.abstract}" for r in hits.itertuples() if r.id in used)
    lines = "\n".join(f'{n}. "{c["mondat"]}" – hivatkozás: {", ".join(c["ids"])}' for n, c in enumerate(claims, 1))
    return VERIFY_PROMPT.format(sources=sources or "(nincs: a hivatkozott azonosítók egyike sem volt a kapott cikkek között)",
                                claims=lines)


def parse_verdicts(text, claims):
    """A modell JSON-válasza → állításonként {'mondat', 'ids', 'itelet' ('igen'/'részben'/'nem'/None), 'indoklas'}.
    Ami hiányzik vagy nem értelmezhető, annak ítélete None (nem ellenőrzött) – kivétel helyett."""
    m = re.search(r"[\[{].*[\]}]", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        data = []
    if isinstance(data, dict):                                   # {"allitasok": [...]} vagy {"1": {...}}
        inner = next((v for v in data.values() if isinstance(v, list)), None)
        data = inner if inner is not None else [dict(v, n=k) for k, v in data.items() if isinstance(v, dict)]
    by_n = {}
    for el in data if isinstance(data, list) else []:
        if not isinstance(el, dict):
            continue
        low = {str(k).strip().lower(): v for k, v in el.items()}
        n = re.search(r"\d+", str(low.get("n", low.get("sorszam", low.get("sorszám", "")))))
        if n:
            by_n[int(n.group(0))] = low
    out = []
    for i, c in enumerate(claims, 1):
        el = by_n.get(i, {})
        v = str(el.get("itelet", el.get("ítélet", el.get("verdict", "")))).strip().lower()
        v = "igen" if v.startswith(("igen", "alátámaszt", "yes", "support")) else \
            "részben" if v.startswith(("rész", "partial")) else "nem" if v.startswith(("nem", "no")) else None
        why = el.get("indoklas", el.get("indoklás", el.get("reason", "")))
        out.append({**c, "itelet": v, "indoklas": why.strip() if isinstance(why, str) else ""})
    return out


def verify_claims(llm, model, answer, hits):
    """Az állítások ellenőrzése egy modellhívással. Visszaad: (állítások listája, hibaüzenet vagy "")."""
    claims = split_claims(answer)
    if not claims or hits is None or not len(hits):
        return [], ""
    try:
        raw = generate(llm, model, build_verify_prompt(claims, hits), json_output=True, temperature=0)
    except Exception as exc:                                     # az ellenőrzés hibája ne vigye el a választ
        return [dict(c, itelet=None, indoklas="") for c in claims], f"az ellenőrzés nem futott le ({exc})"
    return parse_verdicts(raw, claims), ""


def summary_counts(claims):
    """{'igen': n, 'részben': n, 'nem': n, None: n} – a hiányzók is."""
    out = {"igen": 0, "részben": 0, "nem": 0, None: 0}
    for c in claims or []:
        out[c.get("itelet") if c.get("itelet") in VERDICTS else None] += 1
    return out


def claims_markdown(claims, error=""):
    """A felület és a notebook szövege: összegzés, és a nem (teljesen) alátámasztott állítások indoklással."""
    if error:
        return f"\n\n> ⚠ A hivatkozások tartalmi ellenőrzése nem futott le: {error}"
    if not claims:
        return ""
    cnt = summary_counts(claims)
    head = (f"**Hivatkozások tartalmi ellenőrzése** ({len(claims)} hivatkozott állítás): "
            f"{ICONS['igen']} {cnt['igen']} alátámasztott · {ICONS['részben']} {cnt['részben']} részben · "
            f"{ICONS['nem']} {cnt['nem']} nem alátámasztott" + (f" · {cnt[None]} nem értékelt" if cnt[None] else ""))
    flagged = [c for c in claims if c.get("itelet") in ("részben", "nem")]
    rows = "\n".join(f"- {ICONS[c['itelet']]} „{c['mondat']}” [{', '.join(c['ids'])}]"
                     + (f" – {c['indoklas']}" if c.get("indoklas") else "") for c in flagged)
    note = ("\n\n_Az ellenőrzést egy második modellhívás végzi a hivatkozott cikkek absztraktja alapján; "
            "jelzés, nem bizonyíték._")
    return f"\n\n{head}" + (f"\n\n{rows}" if rows else "") + note


def as_list(value):
    """A naplóból (CSV-ből visszaolvasva szöveg is lehet) az állítások listája."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []
