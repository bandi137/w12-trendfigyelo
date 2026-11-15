# Teendők – Süle Szilvia (2026. november 14–15.)

**Leadás: hétfő (november 16.) 00:00.** Ahhoz, hogy András vasárnap este még be tudja építeni az eredményeket a
dokumentációba, az 1–5. lépés legyen kész **vasárnap délig**. A 6. lépés (a dokumentáció szövege) mehet utána.

| # | Feladat | Kb. idő |
| --- | --- | --- |
| 1 | A friss notebook megnyitása és lefuttatása Colabban | 30 perc (ebből 20 perc várakozás) |
| 2 | 15–20 tesztkérdés írása a várt cikkekkel az `eval/kerdesek.csv`-be | 2–3 óra |
| 3 | A kiértékelés futtatása | 10 perc |
| 4 | 5–10 RAG-válasz kézi pontozása 1–5 skálán | 1 óra |
| 5 | Képernyőképek | 30 perc |
| 6 | A dokumentáció szövegének megírása | egyeztetés szerint |

A kódot (`trendfigyelo/` mappa, notebook) továbbra is csak András módosítja. Te a GitHub webes felületén a
saját fájljaidat szerkeszted: `eval/kerdesek.csv` és a `screenshots/` mappa. Ha valami hibát ad, a hibaüzenetet
szó szerint, a cella számával küldd el Andrásnak.

---

## 1. A friss notebook megnyitása és lefuttatása

**Miért:** reggel két változás került a kódba. Az arXiv-adatok mostantól hónaponként tárolódnak a Drive-on, és a
Beállításokba bekerült egy `FRISSITES` kapcsoló. Egy már megnyitott Colab-notebook nem frissül magától, ezért
újra kell nyitni a GitHubról.

1. Zárd be a régi notebook lapját.
2. Nyisd meg a repó főoldalát (github.com/bandi137/w12-trendfigyelo), és kattints az **Open in Colab** gombra.
3. *Runtime → Change runtime type → T4 GPU* (ha nem az van beállítva).
4. *Runtime → Run all*. A Drive-hozzáférést engedélyezd.
5. Ellenőrizd a kimeneteket:
   - **0. szakasz (első kódcella):** a sor így kezdődik: `Kódverzió: <azonosító> – …`. Az azonosítót András megadja, vagy a GitHub főoldalán a legfrissebb commit mellett látod.
   - **1. szakasz (Beállítások):** van benne egy `FRISSITES = "auto"` sor. Ezt **ne** írd át.
   - **2. szakasz (Adatgyűjtés):** az első futáskor ezt írja ki: *„A korábbi papers.csv hónapokra bontva
     (12 hónap …)”*, majd *„4800 cikk … Drive-ról: 12 hónap, most letöltve: 0 hónap.”* Ez azt jelenti, hogy
     **nem** töltött le semmit újra, csak átrendezte a meglévő adatot.
   - **3. szakasz (Vektorizálás):** az első futáskor *„Vektorizálás (változott a cikklista): 4800 cikk
     újraszámolása…”*. Ez egyszer pár percig tart, utána a Drive-ról töltődik
     (*„Vektorok betöltve a Drive-ról”*).
6. Futtasd még egyszer a 2. és a 3. szakaszt. Most már mindkettőnek pár másodperc alatt le kell futnia.

> Ha a Drive-odon még nincs adat (soha nem futtattad végig), a 2. szakasz 15–20 percig tölt le. Ez rendben van.

## 2. Tesztkérdések írása (`eval/kerdesek.csv`)

**Miért:** a kiértékelés azt méri, hogy a keresés (a RAG „R” betűje) megtalálja-e a jó cikkeket. Ehhez kellenek
kérdések, és minden kérdéshez egy vagy több cikk, amelyről **mi magunk döntöttük el**, hogy válaszol a
kérdésre. A mérőszám a hit@k: hányszor van a várt cikk az első 1, 5, illetve 10 találat között. Ez a te
ágad (RAG) fő számszerű eredménye a dokumentációban.

**Kérdésenként:**

1. A notebook **6. szakaszában** keresd meg a „Segéd: várt cikkek keresése” cellát.
2. Írd át benne a `KERDES` és a `KULCSSZO` értékét, majd futtasd a cellát. Két listát kapsz: a keresés első
   10 találatát és a kulcsszóra illő címeket.
3. Olvasd el a címeket, és ha kell, az absztraktokat is. Csak azt az azonosítót válaszd ki (pl. `2602.22144`),
   amelyik **tényleg** válaszol a kérdésre.
   - Ne csak a keresés első találatát másold be, különben a mérés önmagát igazolja.
   - Jó, ha a kulcsszavas listából is választasz. Ha egy jó cikket a keresés nem talál meg, az is érdekes
     eredmény.
4. Jegyezd fel a sort ebben a formában: `azonosító(k),SuleSz,kérdés`.

**Hogy legyen változatos (összesen 15–20 kérdés):**

- kb. fele magyar, fele angol kérdés;
- könnyű kérdések (egy jól körülhatárolt téma, pl. hallucináció, gépi fordítás) és nehezebbek (szűk vagy több
  témát összekötő kérdés);
- egy kérdéshez 1–3 várt cikk; ha több is van, pontosvesszővel válaszd el őket: `2512.19070; 2602.22144`.

**A fájl szerkesztése a GitHubon:**

1. Nyisd meg: github.com/bandi137/w12-trendfigyelo/blob/main/eval/kerdesek.csv
2. Kattints a ceruza ikonra (*Edit this file*).
3. Az első sor (`vart_cikkek,keszitette,kerdes`) maradjon, alá írd a kérdéseket, soronként egyet.
4. *Commit changes…* → az üzenet pl. `Tesztkérdések (Szilvia)` → *Commit directly to the main branch* →
   *Commit changes*.

Példa:

```csv
vart_cikkek,keszitette,kerdes
2512.19070; 2602.22144,SuleSz,Milyen tanítás nélküli módszerek csökkentik a képi nyelvi modellek hallucinációit?
2610.36980,SuleSz,How are LLM agents evaluated on long-horizon tasks?
```

(Ezek csak a formátumot mutatják; a valódi azonosítókat te keresed ki.) A kérdésben lehet vessző, mert mindig
az utolsó oszlop. A részletes leírás az `eval/README.md`-ben van.

Érdemes 5 kérdésenként menteni, és közben egyszer lefuttatni a 3. lépést. Így hamar kiderül, ha valami nem
stimmel a formátummal.

## 3. A kiértékelés futtatása

1. A notebookban futtasd újra az **első cellát** (ez lehúzza a friss `kerdesek.csv`-t), majd a **6. szakasz**
   második kódcelláját (`questions = tf.load_questions(...)`).
2. Ha figyelmeztetést kapsz (*„ezek az azonosítók nincsenek az adatok között (elírás?)”*), javítsd az
   azonosítót a GitHubon, és ismételd meg az 1. pontot.
3. Az eredmény egy táblázat (kérdésenként: megvan-e a várt cikk az első 1/5/10 találatban) és az összesítés
   (hit@1, hit@5, hit@10). Az eredmény a Drive-ra is elmentődik: `W12_trendfigyelo/cs_CL/eval.csv`.

A webes felület **Kiértékelés** fülén ugyanez egy gombbal is megy (7. szakasz).

**Hogyan értelmezd:** a hit@5 = 0,80 azt jelenti, hogy a kérdések 80%-ánál a várt cikk benne volt abban az
5 cikkben, amelyet a nyelvi modell megkap. Ha egy várt cikk nincs benne, a modell nem is tud rá
hivatkozni. Ezért ez a szám a válaszok minőségének felső korlátja. Jegyezd fel azokat a kérdéseket, ahol
a keresés nem találta meg a cikket, mert jó példák a tanulságokhoz.

## 4. RAG-válaszok kézi pontozása

**Miért:** a hit@k csak a keresést méri. A válasz minőségét (pontos-e, nem talál-e ki semmit, jól hivatkozik-e)
embernek kell megítélnie.

1. Indítsd el a webes felületet (notebook 7. szakasz), és a **Kérdezz** fülön tegyél fel 5–10 kérdést. Lehetnek
   a tesztkérdéseid közül is.
2. **Utána** futtasd újra a 6. szakasz utolsó cellát (`rag.save_log(...)`). Ez menti a kérdéseket és a válaszokat a
   Drive-ra: `W12_trendfigyelo/cs_CL/qa_naplo.csv`. Ha ezt a felületen feltett kérdések előtt futtatod, azok
   kimaradnak a naplóból.
3. Nyisd meg a `qa_naplo.csv`-t a Drive-on Google Táblázatokkal. Oszlopok: `question`, `answer`, `sources`
   (a modellnek átadott cikkek), `invalid` (olyan hivatkozás, amelyet a modell nem kapott meg).
4. Adj hozzá két oszlopot: `pont` (1–5) és `megjegyzes`. A pontozás:

| Pont | Jelentés |
| --- | --- |
| 5 | Pontos, a kérdésre válaszol, minden állítás mögött ott a megfelelő forrás |
| 4 | Pontos, kisebb hiány (pl. egy fontos szempont kimaradt, vagy egy hivatkozás pontatlan) |
| 3 | Részben jó: van benne általánosság vagy forrással nem alátámasztott állítás |
| 2 | Inkább rossz: félreérti a kérdést vagy a forrásokat |
| 1 | Használhatatlan, vagy kitalált tartalom (hallucináció) |

   Az ellenőrzéshez nyisd meg a hivatkozott cikkeket (`arxiv.org/abs/<azonosító>`), és nézd meg, hogy az
   absztrakt tényleg azt mondja-e, amit a válasz állít.
5. Mentsd el a táblázatot, és töltsd le CSV-ként (*Fájl → Letöltés → CSV*). Töltsd fel a repóba
   `eval/qa_pontozas.csv` néven (*Add file → Upload files*).

A korábbi futáson feltűnt két dolog. Ezekre figyelj, és írd a `megjegyzes` oszlopba, ha előfordul:
- angol kérdésre is magyar válasz jött;
- a modell egyszer egy számjeggyel elrontott egy azonosítót, és ezt a hivatkozás-ellenőrző jelezte (`invalid`).

## 5. Képernyőképek

Mentsd őket a repó `screenshots/` mappájába (*Add file → Upload files*), beszédes fájlnévvel, pl.
`03_kerdezz_hallucinacio.png`.

1. A webes felület **Kérdezz** füle egy jó válasszal és a forráslistával.
2. Egy angol kérdés és a válasza.
3. Ha előfordul: a figyelmeztetés *„⚠ A modell nem kapott forrásra is hivatkozott: …”*.
4. A **Trendek** fül: a grafikon és a táblázat (előtte: *Témák számítása*, majd *Összefoglaló írása*).
5. A **Kiértékelés** fül az eredménnyel.
6. A közzétett weboldal: bandi137.github.io/w12-trendfigyelo
7. A notebook 2–3. szakaszának kimenete (hónapok a Drive-ról, vektorok betöltve).

## 6. A dokumentáció szövege

Az oktató öt pontot kér: (1) a feladat azonosítója és címe, (2) ki mit csinált, (3) 1–3 bekezdés arról, mi
sikerült és hogyan, (4) a megoldás: promptok, repó, képernyőképek, (5) tanulságok. A fejlesztés részleteit
(napló, tanulságjelöltek, modellválasztás) András fejlesztési összefoglalója tartalmazza; ezt átküldi, vagy
egyeztetjük, ki melyik részt írja. A te részed biztosan: a tesztkérdések, a kiértékelés és a kézi pontozás
eredménye és tanulságai (a 2–4. lépésből).

---

### Ha elakadsz

| Jelenség | Teendő |
| --- | --- |
| `No module named …` vagy furcsa importhiba | *Runtime → Restart session*, majd *Run all* |
| `429` / kvótahiba a Gemininél | A kód magától újrapróbálja; ha tartósan fennáll, várj pár percet |
| A 6. szakasz nem látja az új kérdéseidet | Az első cellát is újra kell futtatni (az húzza le a friss fájlt) |
| A felület linkje (`….gradio.live`) nem működik | A 7. szakasz celláját futtasd újra; a link csak a Colab-munkamenet végéig él |
| Bármi más | A hibaüzenetet szó szerint, a cella számával küldd el Andrásnak |
