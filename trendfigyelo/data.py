"""Adatgyűjtés az arXiv API-ból, havi bontásban, Drive-os gyorsítótárral."""
import os
from datetime import date

import pandas as pd


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


def download_arxiv(category, months, per_month):
    import arxiv  # csak itt kell, a tesztekhez nem
    client = arxiv.Client(page_size=200, delay_seconds=3, num_retries=5)
    rows = []
    for start, end, label in month_ranges(months):
        search = arxiv.Search(query=f"cat:{category} AND submittedDate:[{start} TO {end}]",
                              max_results=per_month, sort_by=arxiv.SortCriterion.SubmittedDate)
        n0 = len(rows)
        for r in client.results(search):
            rows.append({"id": r.get_short_id().split("v")[0], "title": " ".join(r.title.split()),
                         "abstract": " ".join(r.summary.split()), "published": r.published.date().isoformat(),
                         "month": label, "authors": ", ".join(a.name for a in r.authors[:5]),
                         "url": r.entry_id})
        print(label, len(rows) - n0, "cikk")
    return pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)


def load_or_download(data_dir, category, months, per_month):
    """Ha már van papers.csv a Drive-on, azt tölti be; különben letölti és elmenti."""
    path = os.path.join(data_dir, "papers.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"id": str})
        print("Betöltve a Drive-ról:", len(df), "cikk")
        return df
    df = download_arxiv(category, months, per_month)
    df.to_csv(path, index=False)
    print("Mentve:", path, len(df), "cikk")
    return df
