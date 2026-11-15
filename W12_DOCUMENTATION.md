# W12 – Kutatási trendfigyelő RAG-gal

**Tárgy:** Alkalmazott mesterségesintelligencia-modellek (BME) · **Leadás:** 2026. november 16.

| | |
|---|---|
| **Git-repó** | https://github.com/bandi137/w12-trendfigyelo |
| **Közzétett jelentés (statikus HTML, GitHub Pages)** | https://bandi137.github.io/w12-trendfigyelo/ |
| **Webes felület (Gradio)** | GRADIO_LINK_IDE – jelszóval védett, felhasználónév `w12`; a link csak a futó Colab-munkamenet idejére él, minden indításnál új. Ha nem működik, a notebook 7. szakaszát futtatva új link készül. |
| **Notebook** | [megnyitás Colabban](https://colab.research.google.com/github/bandi137/w12-trendfigyelo/blob/main/W12_trendfigyelo_rag.ipynb) |

## 1. A feladat

**Feladatazonosító:** W12
**Cím:** Kutatási trendfigyelő RAG-gal
**Készítők:** Sümegi András ([bandi137](https://github.com/bandi137)) és Süle Szilvia ([SuleSz](https://github.com/SuleSz))

A prototípus az arXiv `cs.CL` kategóriájának cikkcímeiből és absztraktjaiból magyar és angol kérdésekre válaszol, forrásmegjelöléssel. Emellett témákba csoportosítja a cikkeket, és megmutatja a vizsgált mintában növekvő témák részarányát. A megoldás elkészítése mellett azt is vizsgáltuk, hogyan használhatók az MI-eszközök a fejlesztés, a hibakeresés és a dokumentálás támogatására.

## 2. Munkamegosztás

| Készítő | Fő feladatok |
|---|---|
| **Sümegi András** | A kód és a notebook fejlesztésének összefogása (Claude Code-dal); adatgyűjtés, témacsoportosítás és trendág; a webes felület és a közzététel integrációja; GitHub-repó és GitHub Pages. |
| **Süle Szilvia** | A RAG-ág kipróbálása és tesztelése; a 20 tesztkérdés és a várt cikkek összeállítása; a kiértékelés futtatása, 8 válasz kézi pontozása; tesztfuttatások, válaszok és képernyőképek rögzítése; a november 15-i közzététel; a kódhibák felderítése ([eval/errors.md](eval/errors.md)). |

A megoldásról és az elakadásokról folyamatosan egyeztettünk. A GitHub biztosította a közös kód és az eredményfájlok átadását; a fejlesztési szakaszok közötti folytatást a Claude részösszefoglalói és a személyes megbeszélések segítették.

## 3. A megoldás körülményei

A kiadott feladatok közül ezt találtuk a legérdekesebbnek, és más szakterületeken is láttunk felhasználási lehetőséget. A korábbi órai bemutatóink, a kétsugaras oszcilloszkóp és a macskaszőr-szimulátor után ennél a feladatnál a keresési és értékelési folyamat megismerése került előtérbe. Az eredeti adatforrástervben OpenAlex szerepelt; a megvalósult változat az arXiv adataival dolgozik.

Az első elképzelésünk egy állandóan elérhető webalkalmazás volt. A rendelkezésre álló tárhelyen nem volt GPU, és a kísérleti fázisban a működtetéshez kapcsolódó erőforrás- és tokenköltségeket is korlátozni akartuk. A saját gépen futó Python és az Ollamával használt helyi nyelvi modell továbbra is érdekes irány számunkra, de ennek kiépítését a házifeladat keretein túlmutató feladatnak tekintettük. Végül Google Colab-prototípust készítettünk: a notebook böngészőből használható, a vektorizáláshoz GPU rendelhető hozzá, a válaszokat pedig Gemini API-hívások állítják elő.

A közös notebook és a GitHub mindkettőnk számára hozzáférhetővé tette a megoldást. A Colab-munkamenet megszűnése miatt a letöltött adatokat és a számított vektorokat Google Drive-on tároljuk. A Gradio-felület a futó munkamenethez kapcsolódik, a statikus HTML-jelentés pedig GitHub Pages-en is megtekinthető. A végső futásokban egységesen a `gemini-3.5-flash-lite` modellt használtuk.

## 4. A megoldás

### 4.1. Felépítés

```
arXiv API (cs.CL, 12 lezárt hónap × 400 cikk) ──► honapok/ÉÉÉÉ-HH.csv, papers.csv (Google Drive)
        │
        ▼
multilingual-e5-base vektorok (4800 × 768) ──► embeddings.npy
        │                                  │
        ▼                                  ▼
RAG-ág: visszakeresés (top 5)        Trendág: KMeans 25 téma + c-TF-IDF kulcsszavak
→ Gemini válasz forrásokkal          → havi részarány, növekedés
→ hivatkozás-ellenőrzés              → Gemini témanév + leírás + összefoglaló
        │                                  │
        └──────────► Gradio-felület ◄──────┘
                     (Kérdezz · Trendek · Kiértékelés · Közzététel)
                            │
                            ▼
                     statikus HTML-jelentés → GitHub Pages
```

| Elem | Megoldás |
|---|---|
| Futtatókörnyezet | Google Colab (T4 GPU); a notebook első cellája a GitHubról frissíti a kódot |
| Adat | arXiv `cs.CL`, 2025-11 – 2026-10, havonta a 400 legutoljára beküldött cikk (4800 absztrakt); hónaponként tárolva a Drive-on, `FRISSITES` kapcsolóval (auto / mindig / soha) |
| Vektorizálás | `intfloat/multilingual-e5-base` (768 dimenzió), cím + absztrakt; a vektorok a Drive-on, és ha a cikklista vagy a modell változik, mind újraszámolódnak |
| Nyelvi modell | `gemini-3.5-flash-lite` a Gemini API-n (Colab Secrets: `GEMINI_API_KEY`), újrapróbálás kvótahibánál |
| Visszakeresés | koszinusz-hasonlóság, a kérdéshez legközelebbi 5 cikk; opcionális hónapszűrés |
| Válasz | a kérdés nyelvén (a kód állapítja meg: magyar / angol), minden állítás után a forrás azonosítója; a hivatkozás-ellenőrző jelzi, ha a modell meg nem kapott azonosítóra hivatkozik |
| Témák | KMeans, 25 téma; kulcsszavak class-based TF-IDF-fel (csak legalább 5 cikkben előforduló szavak) |
| Növekedés | lásd 4.5 |
| Felület | Gradio (nyilvános `gradio.live` link, jelszóval védve: Colab Secrets `UI_PASSWORD`) |
| Közzététel | a jelentés egyetlen commitban kerül a repó `docs/` mappájába a GitHub REST API-val (Colab Secrets: `GITHUB_TOKEN`); archívum dátum szerint |
| Tesztek | 28 offline teszt (`tests/`), kitalált adaton, szimulált nyelvi modellel és letöltővel; hálózat nélkül futnak |

### 4.2. A megoldás és az ellenőrzés dokumentumai

- **Git-repozitórium:** https://github.com/bandi137/w12-trendfigyelo
- **Notebook:** [W12_trendfigyelo_rag.ipynb](W12_trendfigyelo_rag.ipynb) ([megnyitás Colabban](https://colab.research.google.com/github/bandi137/w12-trendfigyelo/blob/main/W12_trendfigyelo_rag.ipynb))
- **Python-csomag:** [trendfigyelo/](trendfigyelo/) – `data` (letöltés), `embed`, `rag`, `trends`, `prompts`, `llm`, `evaluate`, `app` (Gradio), `report` (HTML + grafikon), `publish`
- **Közzétett jelentés:** https://bandi137.github.io/w12-trendfigyelo/ (archívum: [2026-11-13](docs/archiv/2026-11-13.html), [2026-11-15](docs/archiv/2026-11-15.html))
- **Képernyőképek:** [screenshots/](screenshots/) – a trendek ([trends.png](screenshots/trends.png)), a kiértékelés ([evaluation.png](screenshots/evaluation.png)), egy angol válasz ([EN06_hard_answer.png](screenshots/EN06_hard_answer.png)), egy problémás magyar példa ([HU10_hard_answer.png](screenshots/HU10_hard_answer.png), [HU10_hard_answer_error.png](screenshots/HU10_hard_answer_error.png)), a betöltött tesztkérdések ([20_test_questions_loaded.png](screenshots/20_test_questions_loaded.png)), a közzététel ([publish.png](screenshots/publish.png)) és az első futás ([Colab v01](screenshots/W12_trendfigyelo_rag.ipynb%20-%20Colab%20v01%20elso%20futas.png)).
- **Tesztkérdések és eredmények:** [eval/kerdesek.csv](eval/kerdesek.csv), [eval/eval.csv](eval/eval.csv), [eval/qa_naplo.csv](eval/qa_naplo.csv), [eval/qa_pontozas.csv](eval/qa_pontozas.csv), teljes példaválaszok: [test_questions/](test_questions/)
- **Feltárt kódhibák:** [eval/errors.md](eval/errors.md)
- **Közzétételi útmutató:** [docs/KOZZETETEL.md](docs/KOZZETETEL.md)

### 4.3. Promptok

A promptok a [trendfigyelo/prompts.py](trendfigyelo/prompts.py) fájlban találhatók, innen idézhetők:

| Prompt | Feladata |
|---|---|
| `RAG_PROMPT` és `LANGUAGE_RULES` | Válaszadás kizárólag a kapott cikkek alapján, forrásazonosítókkal, a kérdés nyelvén; a bizonyítékhiány jelzése; legfeljebb 200 szavas válasz. A válasz nyelvét a kód állapítja meg, és a célnyelven írt külön sor kéri (a csak magyarul megfogalmazott kérés mellett a modell angol kérdésre is magyarul válaszolt). |
| `NAMING_PROMPT` | Rövid magyar témanevek (angol szakkifejezéssel) és egymondatos leírások a kulcsszavak és a példaként megadott cikkcímek alapján, JSON-ban. |
| `SUMMARY_PROMPT` | A növekvő témák 4–5 mondatos, közérthető összefoglalása. |

### 4.4. A vizsgált állomány és a futás adatai

A vizsgált állomány 4800 cikket tartalmaz, a `month` mező szerint 2025 novemberétől 2026 októberéig, havonta 400 bejegyzéssel. A címeket és absztraktokat az `intfloat/multilingual-e5-base` modell alakítja vektorokká. A válaszgenerálás alapbeállításban a kérdéshez legközelebbi öt cikket kapja meg; a trendág KMeans-csoportosítást és c-TF-IDF-kulcsszavakat használ.

A közzétett jelentés (2026-11-15 18:38) futásának adatai:

| | |
|---|---|
| Adat letöltve | 2026-11-12 (a jelentés fejlécében is szerepel) |
| Kódverzió a közzétételkor | [`172b47e`](https://github.com/bandi137/w12-trendfigyelo/commit/172b47e) (a közzétételi commit: [`4f6cc83`](https://github.com/bandi137/w12-trendfigyelo/commit/4f6cc83)) |
| Vektormodell | `intfloat/multilingual-e5-base` |
| Nyelvi modell | `gemini-3.5-flash-lite` |
| Beállítások | 12 hónap × 400 cikk, 25 téma, 5 átadott cikk, növekedési ablak 3 hónap |
| A cikkállomány ellenőrzőösszege | nem rögzítettük; lásd a 6. pontot |

**A mintavétel korlátja.** A rendszer havonta legfeljebb 400 cikket vesz át, az arXiv API-tól beküldési dátum szerint csökkenő sorrendben, tehát a hónap *utolsó* beküldéseit. A feltöltött állományban ezek a hónapok utolsó 3–11 napját fedik le: például szeptemberben szeptember 28–30. szerepel. Ez befolyásolhatja a témák részarányát, ezért a jelentés eredményei *a vizsgált havi mintában* növekvő témákról szólnak, nem a kategória egészéről.

### 4.5. A növekedési mutató

A témák növekedése az utolsó 3 hónap átlagos részarányának és az első 3 hónap átlagos részarányának hányadosa, mindkét oldalon 0,005 (0,5 százalékpont) simítással:

```
növekedés = (átlag(részarány, utolsó 3 hónap) + 0,005) / (átlag(részarány, első 3 hónap) + 0,005)
```

A simítás célja, hogy az eleinte 0 részarányú témák ne kapjanak végtelen, néhány cikkből álló témák pedig ne kapjanak irreálisan nagy növekedést. Következménye, hogy a kijelzett érték az egyszerű hányadosnál kisebb: például 1 %-ról 10 %-ra emelkedés kijelzett értéke 7×, nem 10×. A jelentés és a grafikon szövege jelenleg egyszerű hányadosként írja le a mutatót; ezt pontosítani kell (6. pont). A grafikon vonalainak 2 havi mozgóátlaga ettől külön, csak a megjelenítést szolgáló simítás.

### 4.6. Kiértékelés

A keresést 20, fele részben magyar, fele részben angol kérdésen értékeltük. Minden kérdéshez előre kijelölt várt cikk tartozik. A hit@k azt méri, hogy a kijelölt cikkek közül legalább egy szerepel-e az első k találatban.

| Mérőszám | Találatot adó kérdések | Eredmény | magyar kérdések | angol kérdések |
|---|---:|---:|---:|---:|
| hit@1 | 12/20 | 60% | 4/10 | 8/10 |
| hit@5 | 16/20 | 80% | 6/10 | 10/10 |
| hit@10 | 16/20 | 80% | 6/10 | 10/10 |

A mérés forrásai: [kerdesek.csv](eval/kerdesek.csv) és [eval.csv](eval/eval.csv). Nyolc kiválasztott kérdés válaszát külön is értékeltük 1–5 skálán (pontok: 5, 5, 5, 5, 4, 3, 3, 3; átlag 4,1); a válasznapló és a pontozás itt található: [qa_naplo.csv](eval/qa_naplo.csv), [qa_pontozas.csv](eval/qa_pontozas.csv).

## 5. Tanulságok

### 5.1. A feladat határainak kijelölése technikai döntés

Kezdetben a feladatot túldimenzionáltuk. A folyamatosan működő webalkalmazás, a saját gépes futtatás és a helyi nyelvi modell mind lehetséges irány volt, de további üzemeltetési feladatokat is hozott volna. A Colab választásával a rendelkezésre álló időt a RAG-folyamat megismerésére és a prototípus ellenőrzésére tudtuk fordítani.

A Google Colab prototípusfejlesztésre és tanulásra bevált. Egy hasonló feladatot legközelebb is egy végigfuttatható, körülhatárolt prototípussal kezdenénk. Az állandó szolgáltatás kialakítását külön fejlesztési szakaszként kezelnénk, saját erőforrás- és költségtervvel.

### 5.2. A futtatókörnyezet az alkalmazás része

A programozási feladatot nem éreztük különösen bonyolultnak; az MI által javasolt megoldások többnyire már az első futtatáskor működtek. Az elakadások jelentős része a környezethez és a hozzáférésekhez kapcsolódott: nem elérhető modellnevet állítottunk be (a `gemini-2.5-flash` néhány hónap alatt elérhetetlenné vált az új felhasználóknak), nem volt megadva az API-kulcs, illetve kódfrissítés után nem definiált változó okozott hibát. A Colab Pythonja sem azonos egy helyi Pythonnal: az IPython autoreload a hiányzó `imp` modul miatt el sem indult, ezt helyi teszt nem fogta volna meg. Esti használatkor átmeneti, szervertúlterhelésnek tulajdonított fennakadás is előfordult, amely néhány perc várakozás után megszűnt.

A Gemini ezekhez használható hibakeresési tanácsokat adott. A tapasztalatból azt visszük tovább, hogy az induló ellenőrzésnek a modell elérhetőségére, a kulcsokra, a jogosultságokra és a munkamenet állapotára is ki kell terjednie. A frissítést követő teljes, sorrendhelyes próba különösen fontos notebookos munkánál.

### 5.3. A munkamenet és a tartós adatok külön kezelést igényelnek

A Colab-munkamenet megszűnhet, miközben a cikkállományra és a számított vektorokra a következő alkalommal is szükség van. Ez indokolta a Drive-os mentést, a hónaponkénti adattárolást és a vektorok gyorsítótárazását. A projekt újraindításához így felhasználhatók a már letöltött adatok és az elkészült vektorok.

Egy következő projektben már a tervezéskor rögzítenénk, mi számítható újra, minek az előállítása időigényes, és mely eredményeket kell változatlanul megőrizni. A futtatókörnyezet, a kód, az adat és a mérési eredmény helyét külön is dokumentálnánk.

### 5.4. A fiókok és erőforrások áttekintése időt takarít meg

Mindketten több Google-fiókkal rendelkezünk. Zavart okozott a különböző fiókokhoz tartozó tárhelyek áttekintése, valamint az, hogy melyik fiókkal milyen MI-modellhez férünk hozzá. A futásokhoz végül egységesen a `gemini-3.5-flash-lite` modellt használtuk. Csak később derült ki, hogy Szilvia egy másik fiókjával Pro modellhez is hozzáférhetett volna.

Legközelebb az induláskor elkészítenénk egy rövid erőforrástérképet: melyik fiókhoz tartozik a notebook, az adattárhely, az API-hozzáférés és a használható modell. Ez segítené a fejlesztési erőforrások tudatos összpontosítását és az azonos beállításokkal végzett tesztelést.

### 5.5. Az együttműködéshez közös kód és közös helyzetkép is kell

A GitHub megkönnyítette a közös munkát és a fájlok átadását. A fejlesztési fázisok összehangolását és a folytatást a Claude részösszefoglalói nagyban segítették. Ezekhez szóbeli egyeztetés is társult: két szomszédos íróasztalnál dolgoztunk, így az elakadásokat közvetlenül is meg tudtuk beszélni.

Ezt a munkamódszert megtartanánk. Egy fejlesztési szakasz végén érdemes röviden rögzíteni az aktuális állapotot, a kipróbált megoldást, a nyitott hibákat és a következő lépést. Így a folytatásnak ellenőrizhető kiindulópontja van, és könnyebb átadni a feladatot egymásnak vagy egy MI-asszisztensnek.

### 5.6. A visszakeresést és a válaszgenerálást külön kell értékelni

A tesztek az előre várt hibafajtákra adtak konkrét példákat. A képi bizonyíték és a hétköznapi tudás ütközéséről szóló magyar kérdésnél a megfelelő CDH-Bench cikk nem került az öt átadott forrás közé. A modell jelezte a forráshiányt. Más esetben, az angol–maráthi fordítás kérdésénél a megfelelő cikk az első találat volt, és a válasz tartalma is megfelelt az absztraktnak, de a modell `2601.09725` helyett `2501.09725` azonosítóra hivatkozott. Ezt a hivatkozás-ellenőrző jelezte. Ugyanez a hibatípus a november 5-i első futásban is előfordult (`2602.21262` → `2601.21262`): a hivatkozásokat géppel kell ellenőrizni, szemmel nem tűnik fel.

Az első példa a visszakeresés korlátját, a második a generált hivatkozás hibáját mutatja. A hit@k ezért a keresést jellemzi; a válasz minőségének megítéléséhez az állításokat és a hivatkozásokat is össze kell vetni a forrásokkal. Az adatbázisban szereplő cikk csak akkor segítheti a válaszgenerálást, ha bekerül az átadott kontextusba.

### 5.7. A megbízhatóság követelménye a felhasználástól függ

Az ilyen hibákra számítottunk. A prototípus tesztelése konkrétan megmutatta, milyen pontokon lenne szükség további ellenőrzésre. A forráshiány korrekt jelzése fontos viselkedés, ugyanakkor a felhasználó eredeti kérdése ilyenkor megválaszolatlan marad.

Egy továbbfejlesztett vagy érzékenyebb alkalmazásnál mindenképpen többszintű ellenőrzést tartanánk szükségesnek. Ennek részletes megtervezése még további feladat. A mostani megoldásban a visszakeresési mérés, a forrásazonosítók ellenőrzése és a kiválasztott válaszok tartalmi összevetése ad támpontot a korlátok megismeréséhez.

A trendek értelmezésénél is fontos a mérés határa: havi 400 cikk mintáját vizsgáljuk. A feltöltött állományban ezek a hónapok utolsó 3–11 napját fedik le, ezért a megfigyelt részarányváltozást a vizsgált mintára kell vonatkoztatni. A növekedési mutató simítást is alkalmaz (4.5). A húszkérdéses tesztben a magyar és angol kérdések eltérő témájúak, így eredményeik különbségéből a nyelv önálló hatása nem állapítható meg.

### 5.8. A nyelvi modell által írt összefoglaló többet állít, mint amit a mérés alátámaszt

A jelenlegi közzétett összefoglaló a témák gyakoriságából arra is következtet, hogy „a technológia nemcsak olcsóbbá, de okosabbá is válik”, és egy téma „robbanásszerű, tízszeres növekedését” emeli ki. A mérés csak részarányok változását mutatja a vizsgált mintában; ok-okozati vagy minőségi állítást nem támaszt alá. A tízszeres növekedésű téma („Keresőoptimalizálás és értékelés”, 318 cikk, kulcsszavai: answer, accuracy, evidence, points, does, judge) ráadásul inkább gyűjtőtémának látszik, mint egy jól körülhatárolt kutatási iránynak. A kulcsszavas témacímkék korábban is félrevezetőek voltak: az első futás leggyorsabban növő témájának kulcsszavai („tajik, sealed, odrl”) egyetlen cikkekből származó ritka szavak voltak, ezt a legalább 5 cikkben előforduló szavak szűrése javította. Az összefoglaló promptját úgy kell szigorítani, hogy csak a mért részarányokról beszéljen (6. pont).

### 5.9. Ugyanaz a projekt eltérő tudást adott kettőnknek

**András:** az MI-vel támogatott fejlesztésben és a Colab használatában már volt tapasztalatom. Számomra elsősorban a RAG és a Gradio-alkalmazás jelentett újdonságot. A projektben egy közös folyamat részeként tudtam kipróbálni az adatgyűjtést, a vektorizálást, a visszakeresést, a válaszgenerálást és az értékelést.

**Szilvia:** korábban főleg ChatGPT-t használtam, a Claude-dal most kezdtem dolgozni. Több programozási tapasztalatom volt Matlabban, a Python-ismereteim a feladat során gazdagodtak. Újdonság volt számomra a GitHubbal összekapcsolt Claude és a Colab használata is.

A következő közös fejlesztésben megtartanánk a Colabot a prototípus kipróbálásához, a GitHubot az együttműködéshez, valamint a részösszefoglalókat a folytatás megkönnyítésére. A fiókok, hozzáférések és fejlesztési erőforrások rendezését viszont már a munka elején elvégeznénk.

## 6. Ismert hibák, szükséges módosítások és továbbfejlesztési lehetőségek

A leadás előtt a kódot szándékosan nem változtattuk meg (kivéve a felület jelszavas védelmét), hogy a repó megegyezzen azzal, amivel a közzétett eredmények készültek.

### Szükséges módosítások

| Hol | Mit | Miért |
|---|---|---|
| `report.py`, notebook 5. szakasz | A növekedési mutató leírása a jelentésben és a grafikonon írja ki a simítást (4.5), és „a vizsgált havi mintában” növekvő témákról beszéljen. | A jelenlegi szöveg egyszerű hányadost ígér. |
| `prompts.py` – `SUMMARY_PROMPT` | Csak a mért részarány-változásokról írjon; ok-okozati, minőségi („olcsóbb, okosabb”) és a kategória egészére vonatkozó állításokat ne tegyen; a simított mutató korlátját említse. | 5.8. |
| `report.py`, notebook | A jelentés fejlécébe és egy `futas.json`-ba kerüljön a futás rekordja: kódverzió, modellek, beállítások, a `papers.csv` és az `embeddings.npy` SHA-256 ellenőrzőösszege. | Az eredmények utólag is azonosíthatók legyenek; a mostani futáshoz ezt kézzel rekonstruáltuk (4.4), ellenőrzőösszeg nélkül. |
| `rag.py` – hónapszűrés | Előbb ténylegesen szűrje a jelölteket, és azokból válasszon. | Kevés cikknél régebbi cikkeket is visszaad −1 ponttal ([errors.md](eval/errors.md)). |
| `rag.py` – nyelvfelismerés | Kézzel választható válasznyelv a felületen; az automatikus felismerés maradjon alapértelmezés. | Angol kérdésben az „MI”, a „van der Waals” vagy a „résumé” is magyar választ állít be ([errors.md](eval/errors.md)). |
| `trends.py` – témanevek | A JSON értékeinek típusát is ellenőrizze, különben a tartalék elnevezést használja. | Érvényes, de rossz típusú JSON-nál kivétel ([errors.md](eval/errors.md)). |
| `embed.py` – vektor-ujjlenyomat | Az ellenőrzőösszeg a címet és az absztraktot is tartalmazza. | Változatlan azonosító mellett módosult absztrakthoz a régi vektort használja ([errors.md](eval/errors.md)). |

### Továbbfejlesztési lehetőségek

- **Reprezentatív mintavétel.** A hónap teljes cikkállományának letöltése, vagy a hónapon belül egyenletesen (pl. hetente) vett minta a „hónap utolsó napjai” torzítás helyett (4.4).
- **A közzététel védelme.** A felület jelszava (Colab Secrets `UI_PASSWORD`) a mostani leadásban elkészült; ezen túl a Közzététel gombhoz külön megerősítés vagy külön jelszó, a `GITHUB_TOKEN` minimális jogosultsággal, és a közzétett jelentés automatikus ellenőrzése (pl. a témák száma, a dátum) a feltöltés előtt.
- **Minőségi teszt erősebb modellel.** A technikai tesztekhez a gyors és olcsó `gemini-3.5-flash-lite` megfelelő; a RAG-válaszok minőségi teszteléséhez erősebb, érvelő modell (pl. `gemini-3.1-pro-preview`, fizetős API) adna tisztább képet. A modellnév egy helyen állítható.
- **Kiértékelés bővítése.** Több tesztkérdés, nyelvenként azonos témákkal, hogy a magyar és az angol kérdések eltérése (hit@5: 6/10, illetve 10/10) a nyelv hatásaként értelmezhető legyen; a magyar kérdések angolra fordítása a keresés előtt (query translation) mint kipróbálható javítás.
- **Hivatkozás-ellenőrzés javítással.** A tévesen írt azonosítót (egy számjegy eltérés) a rendszer a legközelebbi átadott azonosítóra javíthatná, és jelölhetné a javítást.
- **Témacímkék minősége.** Gyűjtőtémák felismerése (pl. nagy méret és általános kulcsszavak), a témák számának adatvezérelt megválasztása, BERTopic-szerű hierarchikus témák.
- **Összehasonlítás kész eszközzel** (pl. NotebookLM) ugyanazokon a kérdéseken.
- **Állandó szolgáltatás.** Ütemezett havi futás, saját gépen vagy szerveren futó változat helyi nyelvi modellel (Ollama); ennek költség- és erőforrástervét külön szakaszban készítenénk el (5.1).

## 7. Fejlesztési napló (kivonat)

| Dátum | Esemény |
|---|---|
| 2026-09-26 | Projektjavaslat: 10 hetes terv Dockerrel, vektoradatbázissal, webes felülettel; András a trendág, Szilvia a RAG-ág gazdája. |
| 2026-11-04 | Egyszerűsítés egyetlen Colab-notebookra. A `gemini-2.5-flash` 404-et adott, csere `gemini-3.5-flash-lite`-ra. |
| 2026-11-05 | Első valódi futás: 4800 cikk, két mintaválasz, trendtábla, grafikon. Kód modulokba szervezve, c-TF-IDF kulcsszavak, LLM-es témanevek, offline tesztek. GitHub-repó. |
| 2026-11-06 | Colab-hiba (autoreload, `imp`) javítva kézi modul-újratöltéssel. |
| 2026-11-12 | Leadási határidő módosult november 16-ra. Adat letöltve (a közzétett futás alapja). |
| 2026-11-13 | Gradio webes felület; statikus HTML-jelentés és közzététel a GitHub Pages-re; első közzététel. |
| 2026-11-14 | Hónaponkénti adattárolás, `FRISSITES` kapcsoló, letöltési dátum a jelentésben; megjegyzés a modellválasztásról. |
| 2026-11-15 | Grafikon újratervezve; a megjelenített témák száma újraszámolás nélkül állítható; a válasz a kérdés nyelvén; Szilvia 20 tesztkérdése, kiértékelés, 8 pontozott válasz, képernyőképek, kódhibák listája; közzététel 18:38; jelszó a felületen. |
