"""Adatgyűjtés az arXiv API-ból, havi bontásban, hónaponkénti Drive-os tárolással.

A Drive-on:
  honapok/ÉÉÉÉ-HH.csv  – egy lezárt hónap cikkei (egy lezárt hónap tartalma már nem változik)
  letoltes.json        – hónaponként: mikor töltöttük le, hány cikk, milyen beállítással
  papers.csv           – az aktuális időablak összes cikke egyben (ezt használja minden más)
"""
import json
import os
from datetime import date

import pandas as pd

REFRESH_MODES = ("auto", "mindig", "soha")
MONTH_DIR = "honapok"
META_FILE = "letoltes.json"


def month_ranges(n, today=None):
    """Az utolsó n lezárt hónap: (kezdet, vég, 'ÉÉÉÉ-HH') hármasok az arXiv submittedDate szűrőhöz."""
    today = (today or date.today()).replace(day=1)
    out = []
    for i in range(n, 0, -1):
        y, m = today.year, today.month - i
        while m <= 0:
            m += 12
            y -= 1
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((f"{y}{m:02d}010000", f"{ny}{nm:02d}010000", f"{y}-{m:02d}"))
    return out


def download_month(category, start, end, label, per_month):
    """Egy hónap cikkei az arXiv API-ból."""
    import arxiv  # csak itt kell, a tesztekhez nem
    client = arxiv.Client(page_size=200, delay_seconds=3, num_retries=5)
    search = arxiv.Search(query=f"cat:{category} AND submittedDate:[{start} TO {end}]",
                          max_results=per_month, sort_by=arxiv.SortCriterion.SubmittedDate)
    rows = [{"id": r.get_short_id().split("v")[0], "title": " ".join(r.title.split()),
             "abstract": " ".join(r.summary.split()), "published": r.published.date().isoformat(),
             "month": label, "authors": ", ".join(a.name for a in r.authors[:5]), "url": r.entry_id}
            for r in client.results(search)]
    return pd.DataFrame(rows, columns=["id", "title", "abstract", "published", "month", "authors", "url"])


def _read_meta(data_dir):
    path = os.path.join(data_dir, META_FILE)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def _write_meta(data_dir, meta):
    with open(os.path.join(data_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(meta.items())), f, ensure_ascii=False, indent=1)


def _adopt_legacy(data_dir, category, per_month, meta):
    """A régi (egyben letöltött) papers.csv hónapokra bontása, hogy ne kelljen újra letölteni."""
    legacy = os.path.join(data_dir, "papers.csv")
    mdir = os.path.join(data_dir, MONTH_DIR)
    if not os.path.exists(legacy) or os.path.isdir(mdir):
        return
    os.makedirs(mdir, exist_ok=True)
    df = pd.read_csv(legacy, dtype={"id": str})
    when = date.fromtimestamp(os.path.getmtime(legacy)).isoformat()
    for label, part in df.groupby("month"):
        part.to_csv(os.path.join(mdir, f"{label}.csv"), index=False)
        meta[label] = dict(letoltve=when, cikkek=len(part), per_month=per_month, category=category)
    _write_meta(data_dir, meta)
    print(f"A korábbi papers.csv hónapokra bontva ({df.month.nunique()} hónap, letöltve: {when}).")


def load_or_download(data_dir, category, months, per_month, refresh="auto", today=None, downloader=None):
    """Az utolsó `months` lezárt hónap cikkei.

    refresh="auto"   – csak a hiányzó (vagy más beállítással letöltött) hónapokat tölti le az arXivról
    refresh="mindig" – minden hónapot újra letölt
    refresh="soha"   – a Drive-on lévő papers.csv-t használja, akkor is, ha az időablak már továbblépett
    """
    if refresh not in REFRESH_MODES:
        raise ValueError(f"FRISSITES értéke ez legyen: {', '.join(REFRESH_MODES)} (most: {refresh!r})")
    downloader = downloader or download_month
    path = os.path.join(data_dir, "papers.csv")

    if refresh == "soha" and os.path.exists(path):
        df = pd.read_csv(path, dtype={"id": str})
        print("Betöltve a Drive-ról (frissítés kikapcsolva):", len(df), "cikk,", df.month.min(), "–", df.month.max())
        return df

    meta = _read_meta(data_dir)
    if refresh != "mindig":
        _adopt_legacy(data_dir, category, per_month, meta)
    mdir = os.path.join(data_dir, MONTH_DIR)
    os.makedirs(mdir, exist_ok=True)

    parts, loaded, fetched = [], [], []
    for start, end, label in month_ranges(months, today):
        mpath = os.path.join(mdir, f"{label}.csv")
        m = meta.get(label, {})
        cached = (refresh != "mindig" and os.path.exists(mpath)
                  and m.get("category") == category and m.get("per_month") == per_month)
        if cached:
            parts.append(pd.read_csv(mpath, dtype={"id": str}))
            loaded.append(label)
            continue
        part = downloader(category, start, end, label, per_month)
        if part.empty:
            raise RuntimeError(f"{label}: az arXiv 0 cikket adott vissza – próbáld újra később.")
        part.to_csv(mpath, index=False)
        meta[label] = dict(letoltve=(today or date.today()).isoformat(), cikkek=len(part),
                           per_month=per_month, category=category)
        _write_meta(data_dir, meta)       # hónaponként mentjük, hogy megszakadás után ne vesszen el
        parts.append(part)
        fetched.append(label)
        print(label, len(part), "cikk letöltve")

    df = pd.concat(parts, ignore_index=True).drop_duplicates("id").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"{len(df)} cikk, {df.month.min()} – {df.month.max()}. "
          f"Drive-ról: {len(loaded)} hónap, most letöltve: {len(fetched)} hónap"
          + (f" ({', '.join(fetched)})" if fetched else "") + ".")
    return df


def download_dates(data_dir, month_labels):
    """A megadott hónapok letöltési dátuma(i) a jelentéshez: '2026-11-05' vagy '2026-11-05 – 2026-12-01'."""
    meta = _read_meta(data_dir)
    days = sorted({meta[m]["letoltve"] for m in month_labels if m in meta})
    if not days:
        return ""
    return days[0] if len(days) == 1 else f"{days[0]} – {days[-1]}"
