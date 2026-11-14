# W12 – Kutatási trendfigyelő RAG-gal: folytatási jegyzet (frissítve: 2026-11-13, délután)

A teljes fejlesztési összefoglaló (feladat, felépítés, eredmények, napló, tanulságok, teendők) itt van, ezt kell először elolvasni:
https://claude.ai/artifact/2GZDCWTAp3tby8K8kGiTzK  (Claude Doc; docs-eszközökkel olvasható, id: 0a452541-054d-47b6-9a34-60547ec7389c)

## Legfontosabb tények
- Csapat: Sümegi András (GitHub: bandi137), Süle Szilvia (GitHub: SuleSz). Leadás: 2026. november 16. (hétfő) 00:00 (az eredeti 2026-11-11 helyett).
- Az oktató dokumentációt kér: (1) feladat azonosítója és címe, (2) beadók + ki mit csinált, (3) 1–3 bekezdés a megoldás körülményeiről, (4) a megoldás: promptok, git-repó, képernyőképek, (5) tanulságok.
- Repó: github.com/bandi137/w12-trendfigyelo (a Claude GitHub-alkalmazás telepítve, push működik; új beszélgetésben újra csatolni kell). Legutóbbi kódváltozás (2026-11-13): Gradio webes felület + statikus HTML-jelentés közzététele a GitHub Pages-re (https://bandi137.github.io/w12-trendfigyelo/).
- Futtatás: Google Colab (T4). A notebook első cellája git pull-lal frissíti a kódot, majd kézzel törli és újraimportálja a trendfigyelo modulokat (az IPython autoreload a Colabban nem működik: "No module named 'imp'"). Kódfrissítés után: első cella + Runtime → Run after. Adatok a Drive-on: MyDrive/W12_trendfigyelo/cs_CL/ (hónaponként: honapok/ÉÉÉÉ-HH.csv, letöltési dátumok: letoltes.json; a Beállításokban FRISSITES = "auto" | "mindig" | "soha"; a vektorok változáskor mind újraszámolódnak, embeddings.json alapján). Mindenki a saját Gemini API-kulcsát használja (Colab Secrets: GEMINI_API_KEY).
- Stack: arXiv cs.CL, 12 hónap × 400 cikk (4800); intfloat/multilingual-e5-base; gemini-3.5-flash-lite; KMeans 25 téma + c-TF-IDF kulcsszavak; LLM-es témanév + leírás.
- Kód: trendfigyelo/ (data, embed, llm, rag, trends, prompts, evaluate, app, report, publish), tests/ (23 offline teszt).
- Webes felület: a notebook 7. szakasza indítja (Gradio, `….gradio.live` link a munkamenet végéig). Fülek: Kérdezz, Trendek, Kiértékelés, Közzététel. A Közzététel gomb egy commitban feltölti a `docs/index.html`-t és a `docs/archiv/ÉÉÉÉ-HH-NN.html`-t; ehhez `GITHUB_TOKEN` kell a Colab Secrets-ben (Sümegi András: finomhangolt token, Contents RW; Süle Szilvia: klasszikus token, `public_repo`, mert közreműködőként a finomhangolt nem működik). Útmutató: `docs/KOZZETETEL.md`.
- Tesztkérdések: `eval/kerdesek.csv` (oszlopok: vart_cikkek, keszitette, kerdes; útmutató: `eval/README.md`). A GitHub webes felületén szerkeszthető; a notebook 6. szakasza beolvassa, és figyelmeztet az elírt azonosítókra.

## Munkamegosztás
- Ágak gazdái (a projektjavaslat óta): Sümegi András – trendág (témák, trendek), Süle Szilvia – RAG-ág (kérdés–válasz, visszakeresés).

### Napi munka (2026-11-13-tól)
- Sümegi András: minden kód- és notebook-módosítás Claude Code-dal, ő tölti fel a repóba. A notebookot és a `trendfigyelo/` kódot egyszerre csak ő szerkeszti.
- Süle Szilvia (ingyenes Claude, nincs Claude Code): futtatás és ellenőrzés Colabban, tesztkérdések az `eval/kerdesek.csv`-ben, kézi pontozás, képernyőképek, a dokumentáció szövege. A saját fájljait a GitHub webes felületén tölti fel közreműködőként.

## Állapot
- Az első valódi futás (a59865f) működött; hibája a zajos kulcsszavak voltak (pl. "tajik, sealed, odrl") — javítva b136390-ben.
- b136390 Colabban az autoreload miatt elakadt; javítva 1739461-ben. Az 1739461-es verzió mindkettőnknél működik Colabban.

## Következő lépések
GITHUB_TOKEN beállítása mindkettőnknél → első közzététel a felületről → 15–20 tesztkérdés az `eval/kerdesek.csv`-be → kiértékelés → kézi pontozás → képernyőképek → munkamegosztás rögzítése a dokumentációban → dokumentáció megírása.
