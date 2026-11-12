# W12 – Kutatási trendfigyelő RAG-gal: folytatási jegyzet (frissítve: 2026-11-12)

A teljes fejlesztési összefoglaló (feladat, felépítés, eredmények, napló, tanulságok, teendők) itt van, ezt kell először elolvasni:
https://claude.ai/artifact/2GZDCWTAp3tby8K8kGiTzK  (Claude Doc; docs-eszközökkel olvasható, id: 0a452541-054d-47b6-9a34-60547ec7389c)

## Legfontosabb tények
- Csapat: Sümegi András, Szilvia. Leadás: 2026. november 16. (hétfő) 00:00 (az eredeti 2026-11-11 helyett).
- Az oktató dokumentációt kér: (1) feladat azonosítója és címe, (2) beadók + ki mit csinált, (3) 1–3 bekezdés a megoldás körülményeiről, (4) a megoldás: promptok, git-repó, képernyőképek, (5) tanulságok.
- Repó: github.com/bandi137/w12-trendfigyelo (a Claude GitHub-alkalmazás telepítve, push működik; új beszélgetésben újra csatolni kell). Aktuális commit: 1739461.
- Futtatás: Google Colab (T4). A notebook első cellája git pull-lal frissíti a kódot, majd kézzel törli és újraimportálja a trendfigyelo modulokat (az IPython autoreload a Colabban nem működik: "No module named 'imp'"). Kódfrissítés után: első cella + Runtime → Run after. Adatok a Drive-on: MyDrive/W12_trendfigyelo/cs_CL/.
- Stack: arXiv cs.CL, 12 hónap × 400 cikk (4800); intfloat/multilingual-e5-base; gemini-3.5-flash-lite (GEMINI_API_KEY a Colab Secrets-ben); KMeans 25 téma + c-TF-IDF kulcsszavak; LLM-es témanév + leírás.
- Kód: trendfigyelo/ (data, embed, llm, rag, trends, prompts, evaluate), tests/test_core.py (8 offline teszt).

## Állapot
- Az első valódi futás (a59865f) működött; hibája a zajos kulcsszavak voltak (pl. "tajik, sealed, odrl") — javítva b136390-ben.
- b136390 Colabban az autoreload miatt elakadt; javítva 1739461-ben. 1739461 valódi adaton még nem futott.

## Következő lépések
Új verzió futtatása Colabban → 15–20 tesztkérdés (EVAL_QUESTIONS) → kézi pontozás → képernyőképek → munkamegosztás rögzítése → dokumentáció megírása.
