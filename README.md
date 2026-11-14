# w12-trendfigyelo

<a href="https://colab.research.google.com/github/bandi137/w12-trendfigyelo/blob/main/W12_trendfigyelo_rag.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**W12 – Kutatási trendfigyelő RAG-gal.** Az arXiv egy kategóriájának (alapból `cs.CL`) elmúlt 12 havi
absztraktjaiból kérdésekre válaszol forrásmegjelöléssel, és kimutatja a felfutó kutatási témákat.

**Legfrissebb jelentés:** https://bandi137.github.io/w12-trendfigyelo/ · **Hol tartunk:** lásd a [folytatási jegyzetet](W12_folytatas.md).

## Készítők

Sümegi András ([bandi137](https://github.com/bandi137)), Süle Szilvia ([SuleSz](https://github.com/SuleSz))

## Felépítés

| Fájl | Tartalom |
| --- | --- |
| `W12_trendfigyelo_rag.ipynb` | Colab notebook: beállítások, hívások, eredmények |
| `trendfigyelo/data.py` | arXiv-letöltés hónaponként, a hónapok külön tárolva a Drive-on (`FRISSITES`: auto / mindig / soha) |
| `trendfigyelo/embed.py` | Vektorizálás (`intfloat/multilingual-e5-base`) |
| `trendfigyelo/rag.py` | Visszakeresés, válaszgenerálás, hivatkozás-ellenőrzés |
| `trendfigyelo/trends.py` | Témák (KMeans + c-TF-IDF kulcsszavak), havi trendek, LLM-es témanevek, táblázat, grafikon |
| `trendfigyelo/prompts.py` | Az összes prompt |
| `trendfigyelo/evaluate.py` | Kiértékelés (hit@1/5/10), tesztkérdések beolvasása |
| `trendfigyelo/app.py` | Gradio webes felület: Kérdezz, Trendek, Kiértékelés, Közzététel |
| `trendfigyelo/report.py` | Önálló HTML-jelentés (interaktív Plotly-grafikon) |
| `trendfigyelo/publish.py` | Közzététel a GitHub Pages-re egyetlen commitban (GitHub API, token) |
| `docs/` | A GitHub Pages weboldal; közzétételi útmutató: `docs/KOZZETETEL.md` |
| `eval/kerdesek.csv` | Tesztkérdések a várt cikkekkel (útmutató: `eval/README.md`) |
| `tests/` | Offline tesztek kitalált adaton és szimulált LLM-mel |

## Használat Colabban

1. Nyisd meg a notebookot a fenti gombbal, és állítsd be: *Runtime → Change runtime type → T4 GPU*.
2. A 🔑 *Secrets* fülön vedd fel a `GEMINI_API_KEY` titkot.
3. *Runtime → Run all*. Az adatok és a vektorok a Google Drive `W12_trendfigyelo/` mappájába mentődnek.
   Minden lezárt hónap egyszer töltődik le (`honapok/ÉÉÉÉ-HH.csv`); új hónapnál csak az. A vektorok akkor
   számolódnak újra (mind), ha a cikklista vagy a modell változott.

**Webes felület:** a notebook 7. szakasza elindítja (nyilvános `….gradio.live` link a Colab-munkamenet végéig).
A *Közzététel* gombhoz kell egy `GITHUB_TOKEN` a *Secrets* fülön – útmutató: [`docs/KOZZETETEL.md`](docs/KOZZETETEL.md).

**Kódfrissítés után** futtasd újra a notebook első celláját, majd az alatta lévőket (*Runtime → Run after*):
az első cella lehúzza és újratölti a legfrissebb kódot. A notebook szerkezetének változásakor nyisd meg újra a GitHubról.

## Tesztek

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```
