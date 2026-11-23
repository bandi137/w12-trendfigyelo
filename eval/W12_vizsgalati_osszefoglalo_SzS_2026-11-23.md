# W12 – A trendfigyelő ellenőrzésének összefoglalója

**Süle Szilvia · 2026. november 23.**

**Vizsgált kódverzió:** [`f2ef8e5`](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/README.md)  
**Weboldal:** https://bandi137.github.io/w12-trendfigyelo/

Átnéztem a projekt Python-kódját, a Colab-notebookot, a dokumentációt és a közzétett jelentést. Az ellenőrzés során tesztek, modellválaszok és böngészős próbák készültek. Mind a 32 meglévő teszt sikeresen lefutott. Több olyan hibát is sikerült reprodukálni, amelyet ezek a tesztek még nem fednek le.

## A havi 400 cikkes korlát

A havonta legfeljebb 400 cikk letöltése tudatos tesztbeállítás: a fejlesztés alatt korlátozza a feldolgozandó adatmennyiséget és a futási időt. Ezt nem tekintem programhibának.

Az adatgyűjtés a hónap végén beküldött cikkeket veszi át, ezért a mostani trendek erre a mintára vonatkoznak. A jelentésben ezt röviden jelezni kellene. A teljes havi állomány feldolgozását vagy az időben egyenletes mintavételt a tesztfázis utáni fejlesztések közé sorolnám.

## Javítandó hibák és kockázatok

- **Jelszó nélkül is elindulhat a nyilvános kezelőfelület.** Ha nincs `UI_PASSWORD`, a program csak figyelmeztet. Beállított GitHub-token mellett a link birtokosa közzétehet, és az API-keretet is használhatja. A nyilvános indításhoz kötelezővé tenném a hitelesítést. Ez feltételes kockázat; a ténylegesen használt felület jelszó nélkül nem működik. [Érintett kód](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/app.py#L261)

- **A hónapszűrés régebbi cikkeket is visszaad.** Ha a kiválasztott időszakban kevesebb cikk van, mint a kért találatszám, a program a hiányzó helyeket korábbi cikkekkel tölti fel, −1 pontszámmal. Üres időszaknál is ad találatot. Először ténylegesen szűrni kell a cikkeket, és csak a megmaradtak közül választani. [Érintett kód](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/rag.py#L11)

- **A térképes cikkhivatkozás nem működik.** A pontokra kattintáskor az arXiv-azonosítót ellenőrző JavaScript-mintában túl sok escape-karakter szerepel, ezért egy szabályos azonosítót is elutasít. A hibát a kódban és az élő oldalon is ellenőriztem. [Érintett sor](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/tmap.py#L287)

- **Mobilméretben használhatatlanná szűkül a trendgrafikon.** A rögzített, 270 pixeles jobb margó szinte elfogyasztja a rajzterületet. A feliratoknak keskeny képernyőn más elrendezés kellene. A tématérkép átméretezés után egy próbában üres maradt, újratöltésre megjelent; sötét módban több felirat nehezen olvasható. [Grafikonbeállítás](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/report.py#L124)

- **A gyorsítótár nem minden tartalmi változást érzékel.** Ha ugyanahhoz a cikkazonosítóhoz módosított cím vagy absztrakt tartozik, a régi vektor maradhat használatban. A térkép ellenőrzőösszege pedig csak a vektorok első nyolc komponensét veszi figyelembe. A gyorsítótárak ellenőrzésébe a teljes releváns tartalomnak és beállításnak be kellene kerülnie. [Vektorok](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/embed.py#L23) · [Térkép](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/tmap.py#L34)

- **Hibás adattípusú modellválasz leállítja a témaelnevezést.** A JSON lehet szabályos, miközben a várt szótár helyett szöveget, vagy a név helyett számot tartalmaz. Ilyenkor a tartalék elnevezés helyett kivétel keletkezik. Az értékek típusát is ellenőrizni kellene. [Érintett kód](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/trends.py#L64)

- **Tévedhet az automatikus nyelvfelismerés.** Angol kérdésben az „MI”, a „van der Waals” vagy a „résumé” magyar válasznyelvet állíthat be. A felületen legyen kézzel választható magyar vagy angol válasznyelv is. [Érintett kód](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/rag.py#L28)

- **A növekedési mutató leírása nem egyezik a számítással.** A képlet mindkét átlagos részarányhoz 0,005-öt, azaz 0,5 százalékpontot ad. Emiatt például 1%-ról 10%-ra növekedésnél 7× jelenik meg. A simítás indokolt lehet, de a jelentésnek és a grafikon feliratának közölnie kellene. [Számítás](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/trends.py#L46)

- **A hivatkozás-ellenőrzés csak az azonosítókra terjed ki.** Azt vizsgálja, hogy az idézett cikk szerepelt-e a modellnek átadott források között. A hivatkozás nélküli választ és a forrással alá nem támasztott állítást nem jelzi. Ezekhez külön ellenőrzés szükséges. [Érintett kód](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/rag.py#L41)

- **A közzétételhez kevés szerkesztési lehetőség tartozik.** A jelentésbe az egész kérdés–válasz-napló bekerülhet, az aznapi új publikálás pedig felülírja az azonos dátumú archív oldalt. Szükség lenne a közlendő válaszok kiválasztására, a feltöltendő változat előnézetére és egyedi futásazonosítóra. [Jelentés és közzététel](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/trendfigyelo/app.py#L122)

A hónapszűrés, a nyelvfelismerés, a témanevek feldolgozása és a vektorgyorsítótár hibája már az [`eval/errors.md`](https://github.com/bandi137/w12-trendfigyelo/blob/f2ef8e525461351ace12085bda53e6bf0924550e/eval/errors.md) fájlban is szerepel.

## Javasolt továbbfejlesztések

- **A témák minőségének ellenőrzése.** Témánként néhány reprezentatív cikket és kézzel javítható nevet jelenítenék meg. A „Keresőoptimalizálás és értékelés” elnevezést például érdemes felülvizsgálni: a hozzá tartozó általános kulcsszavak önmagukban nem igazolják ezt a címkét. A különböző futások témáinak összevetéséhez tartós témaazonosítók is kellenének.

- **Óvatosabb összefoglaló.** A prompt a mintában mért részarányváltozásokról kérjen szöveget. A cikkek gyakoriságából önmagában nem következik, hogy egy technológia jobb, olcsóbb vagy megbízhatóbb lett.

- **Pontosabb visszakeresés.** Kipróbálnám a kulcsszavas és a szemantikus keresés összekapcsolását, majd a találatok külön újrarangsorolását. Kevés vagy gyenge forrás esetén a rendszer jelezze, hogy a kérdésre nincs elegendő bizonyíték.

- **Bővebb kiértékelés.** A meglévő 20 keresési teszt és 8 kézzel pontozott válasz mellé azonos tartalmú magyar–angol kérdéspárokat, megválaszolhatatlan kérdéseket és a most feltárt hibákra célzott teszteket vennék fel. A visszakeresés és a generált válasz minőségét továbbra is külön mérném.

- **Megismételhető futások és automatizált ellenőrzés.** Rögzíteném a csomagverziókat, a beállításokat és az adatok ellenőrzőösszegét. A tesztek fussanak automatikusan a GitHubon, a grafikonokat pedig böngészőben, mobilméretben is ellenőrizzük. Hosszabb távon csak az új vagy módosított cikkeket kellene újravektorizálni.

Elsőként a nyilvános felület védelmét, a hónapszűrést, a térképes hivatkozást és a mobilgrafikont javítanám. Ezek után következhet a gyorsítótár és a modellválaszok feldolgozásának rendbetétele, majd a keresés és a trendelemzés minőségének javítása.

