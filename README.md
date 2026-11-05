# w12-trendfigyelo

<a href="https://colab.research.google.com/github/bandi137/w12-trendfigyelo/blob/main/W12_trendfigyelo_rag.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**W12 – Kutatási trendfigyelő RAG-gal.** Az arXiv egy kategóriájának (alapból `cs.CL`) elmúlt 12 havi
absztraktjaiból kérdésekre válaszol forrásmegjelöléssel, és kimutatja a felfutó kutatási témákat.

## Felépítés

| Fájl | Tartalom |
| --- | --- |
| `W12_trendfigyelo_rag.ipynb` | Colab notebook: beállítások, hívások, eredmények |
| `trendfigyelo/data.py` | arXiv-letöltés havi bontásban, Drive-os gyorsítótárral |
| `trendfigyelo/embed.py` | Vektorizálás (`intfloat/multilingual-e5-base`) |
| `trendfigyelo/rag.py` | Visszakeresés, válaszgenerálás, hivatkozás-ellenőrzés |
| `trendfigyelo/trends.py` | Témák (KMeans + c-TF-IDF kulcsszavak), havi trendek, LLM-es témanevek, táblázat, grafikon |
| `trendfigyelo/prompts.py` | Az összes prompt |
| `trendfigyelo/evaluate.py` | Kiértékelés (hit@1/5/10) |
| `tests/` | Offline tesztek kitalált adaton és szimulált LLM-mel |

## Használat Colabban

1. Nyisd meg a notebookot a fenti gombbal, és állítsd be: *Runtime → Change runtime type → T4 GPU*.
2. A 🔑 *Secrets* fülön vedd fel a `GEMINI_API_KEY` titkot.
3. *Runtime → Run all*. Az adatok és a vektorok a Google Drive `W12_trendfigyelo/` mappájába mentődnek.

**Kódfrissítés után** elég a notebook első celláját újrafuttatni: lehúzza a legfrissebb kódot a GitHubról,
és az `autoreload` betölti. A notebook szerkezetének változásakor nyisd meg újra a GitHubról.

## Tesztek

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```
