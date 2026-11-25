# W12 – Ellenőrzés és nyitott hibák

**Gazdája:** Süle Szilvia · **Indult:** 2026. november 24. · **Leadási határidő:** 2026. december 5. (szombat) 00:00

Ez a fájl a két korábbi ellenőrzési fájlt váltja fel. A régiek a repó történetében változatlanul megmaradnak, rögzített linkkel:
[W12_vizsgalati_osszefoglalo_SzS_2026-11-23.md](https://github.com/bandi137/w12-trendfigyelo/blob/0598356/eval/W12_vizsgalati_osszefoglalo_SzS_2026-11-23.md) ·
[errors.md](https://github.com/bandi137/w12-trendfigyelo/blob/0598356/eval/errors.md).
Az azóta kijavított hibák listája a [dokumentáció 6. pontjában](../W12_DOCUMENTATION.md#6-ismert-hibák-szükséges-módosítások-és-továbbfejlesztési-lehetőségek) van („Javítva”).

A fájl élő: a megállapítások dátummal kerülnek bele, a fájl neve nem változik. Így a dokumentáció mindig ugyanide hivatkozhat.

---

## 1. Tesztelési feladat (2026. november 25-től)

A november 23-i ellenőrzés óta változott részek kipróbálása **valódi adaton**. Az *Eredmény* oszlopba: **OK**, vagy **hiba** és egy rövid leírás (ha lehet, képernyőképpel a `screenshots/` mappába). A fájl a GitHub webes felületén szerkeszthető (ceruza ikon).

### 1.1. Előkészítés

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| E1 | Ha a 2. pont szerint **te** töltöd le az új adatot: nyisd meg a notebookot frissen a GitHubról (README: „Open in Colab”), majd **Runtime → Run all**. Ha András tölti le, előbb a megosztott mappát kell beállítani (2. pont), és csak utána futtatni. | Az 1. szakaszban `PER_MONTH = 3000`. A 2. szakasz hónaponként kiírja: „… 10/30 nap, N cikk”, végül „ÉÉÉÉ-HH N cikk letöltve”; a 12 hónap kb. 25–40 perc, összesen kb. 20–25 ezer cikk. A vektorizálás és a térkép egyszer teljesen újraszámolódik (néhány perc). Második futáskor minden a Drive-ról töltődik. | |
| E2 | Új kódcellában: `!cd /content/repo && python -m pytest -q` | `52 passed` (a figyelmeztetések nem számítanak). | |
| E3 | Nézd meg a jelentés fejlécében az „adat letöltve” dátumot, és a 6. szakasz (kiértékelés) figyelmeztetéseit. | Az új letöltés napja. Ha egy tesztkérdés várt cikke nincs az új adatban, a kiértékelés figyelmeztet; ezeket a kérdéseket írd ide (az `eval/kerdesek.csv` várt azonosítóját át kell nézni). | |

### 1.2. A felület védelme

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| V1 | A Colab 🔑 Secrets fülön kapcsold ki a notebook hozzáférését a `UI_PASSWORD`-höz, és futtasd a 7. szakaszt. | A cella hibaüzenettel leáll („A nyilvános felület jelszó nélkül nem indul el…”), nincs `gradio.live` link. | |
| V2 | Kapcsold vissza, és futtasd újra. | Bejelentkező ablak; `w12` + a jelszó után a felület betölt. | |

### 1.3. Kérdezz fül

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| K1 | Tegyél fel egy kérdést. | Kattintás után a gomb „⏳ Kérdezek…” lesz és letiltódik, a válasz helyén várakozó üzenet; a végén a gomb visszaáll. A fülön **nincs** térkép. | |
| K2 | „Csak ettől a hónaptól”: **2026-10**, „Hány cikket kapjon a modell”: **10**. | Minden forrás 2026-10-es. | |
| K3 | Angol kérdés „MI” vagy „van der Waals” szóval, a válasz nyelve **automatikus**, majd ugyanez **angol** beállítással. | Automatikusnál lehet magyar a válasz (ez az ismert korlát); „angol” választásnál angol. | |
| K4 | Egy témán kívüli kérdés (pl. „Mi Franciaország fővárosa?”). | Ha a válasz egyetlen forrásra sem hivatkozik, figyelmeztetés jelenik meg: „A válasz egyetlen átadott forrásra sem hivatkozik…”. | |

### 1.4. Tématérkép fül

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| T1 | A felület indulása után, a Kérdezz fülön feltett kérdés **előtt** nyisd meg a fület (a notebook 4. szakaszának kérdései nem számítanak). | „A kérdés és a források a tématérképen” jelölőnégyzet üres és szürke, nem választható. | |
| T2 | „Témák összehúzása”: **0**, majd **0,8** (a csúszka elengedése után a térkép újrarajzolódik). | 0-nál szétterülő, egymásba folyó felhő; 0,8-nál tömör foltok. A foltok helye és a térkép iránya nem változik. | |
| T3 | Tegyél fel két kérdést a Kérdezz fülön, majd gyere vissza. | A jelölőnégyzet aktív és bejelölt; a térképen a **legutóbbi** kérdés: ★, karikák, vonalak és „1”–„5” címkék. | |
| T4 | Vidd az egeret a címkékre és a karikákra. | Az „1.”–„5.” források címe és sorrendje megegyezik a Kérdezz fül forráslistájával. | |
| T5 | A térkép bal felső menüjében válts a másik kérdésre, majd „(egyik kérdés sem)”. | A csillag, a karikák és a címkék a kiválasztott kérdéshez váltanak, illetve eltűnnek. | |
| T6 | Vedd ki a pipát a jelölőnégyzetből. | A kérdések eltűnnek a térképről. | |
| T7 | A térkép alatti hónapcsúszkával válassz egy hónapot. | Az adott hónap pontjai kiemelődnek; a kérdés jelölései megmaradnak. | |

### 1.5. Trendek fül

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| R1 | „Témák számítása”, majd „Összefoglaló írása”. | Mindkettőnél „⏳ …” gomb és várakozó üzenet, a végén visszaáll. | |
| R2 | Olvasd el az összefoglalót. | Ismert hiba (nyitott): minőségi, ok-okozati állításokat tehet („olcsóbbá, okosabbá”). Ha ilyet látsz, másold ide. | |

### 1.6. A közzétett jelentés (a Közzététel fül után, https://bandi137.github.io/w12-trendfigyelo/)

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| J1 | Kattints egy pontra a tématérképen. | Új lapon megnyílik a cikk az arXivon. | |
| J2 | A térkép bal felső menüje és a ▶ lejátszás. | A menü a kérdések között vált; a lejátszás végiglépteti a hónapokat („■ állj” megállítja). | |
| J3 | Telefonon (vagy keskeny ablakban). | A trendgrafikon alatt jelmagyarázat van, nem lóg ki; a térkép kitölti a szélességet, alatta nincs üres sáv; a kérdésválasztó a csúszka alatt van. | |
| J4 | Sötét módban (telefon vagy böngésző beállítás). | A térkép feliratai, a karikák, a címkék és a menü olvashatók. | |
| J5 | Méretezd át az ablakot széles → keskeny → széles. | A térkép egyszer sem marad üres. | |
| J6 | Olvasd el a Trendek és a Tématérkép bevezetőjét, és a térkép címét. | A Trendeknél: „Havonta legfeljebb 3000 cikk szerepel, a hónap minden napjáról…”. A térképnél a tengelyek magyarázata és a vektorok dimenziója (768); a címben „… cikk (a képen kb. 6000 pont, véletlen minta)”. | |

### 1.7. Dokumentáció

| # | Mit csinálj | Mit kell látnod | Eredmény |
|---|---|---|---|
| D1 | Olvasd át a [W12_DOCUMENTATION.md](../W12_DOCUMENTATION.md) 4.7. és 6. pontját. | Megegyezik azzal, amit a felületen és a jelentésben látsz. Ami nem, ide. | |

---

## 2. Az adatállomány: új, közös letöltés

**Előzmény.** A közzétett jelentések nem ugyanabból a letöltésből készültek, mert mindenki a saját Drive-ján tárolja az adatot:

| Jelentés | Ki tette közzé | Adat letöltve | A leggyorsabban növő téma |
|---|---|---|---|
| 2026-11-15, 2026-11-23 | Szilvia | 2026-11-12 | „Keresőoptimalizálás és értékelés”, 10,1× |
| 2026-11-24 | András | 2026-11-04 | „Bizonyítékalapú értékelés”, 9,25× |

Mindkettő a régi, havi 400 cikkes minta volt, amely csak a hónap utolsó napjait fedte le („400 cikk”-hiba). A kiértékelés (hit@1 0,60 · hit@5 0,80 · hit@10 0,80) mindkettőnél azonos.

**Döntés (2026. november 24.).** Az új kód napi bontásban tölt le, havonta legfeljebb 3000 cikket a hónap minden napjáról (a cs.CL-ben ez szinte a teljes havi anyag). A régi hónapok ezért mindkettőnknél újratöltődnének. Javaslat:

1. **Egyikünk tölt le** (E1) a saját Drive-jára, és a `MyDrive/W12_trendfigyelo` mappát megosztja a másikkal (szerkesztési joggal).
2. A másik a megosztott mappára parancsikont tesz a saját *Saját meghajtó*-jába (*Hozzáadás a Saját meghajtóhoz* / *Add shortcut to Drive*), ugyanazzal a névvel (`W12_trendfigyelo`); a saját régi mappáját előtte átnevezi (pl. `W12_trendfigyelo_regi`). Így a notebook útvonala változatlan, és mindketten ugyanazt az adatot használjuk.
3. A végleges jelentés és a dokumentáció számai ebből az egy letöltésből készülnek; a dátumát ide írjuk: **…**

---

## 3. Még nyitott a korábbi ellenőrzésből

A [dokumentáció 6. pontja](../W12_DOCUMENTATION.md#6-ismert-hibák-szükséges-módosítások-és-továbbfejlesztési-lehetőségek) szerint (2026. november 24.):

- **Növekedési mutató leírása.** A képlet mindkét részarányhoz 0,5 százalékpontot ad (simítás); a jelentés és a grafikon ezt nem írja ki.
- **Óvatosabb összefoglaló.** A `SUMMARY_PROMPT` csak a mért részarány-változásokról kérjen szöveget, minőségi és ok-okozati állítások nélkül.
- **Futásrekord.** Kódverzió, modellek, beállítások és az adatok ellenőrzőösszege a jelentés fejlécében és egy `futas.json`-ban.
- **Hivatkozás-ellenőrzés.** A forrással alá nem támasztott állítás jelzése (most csak az ismeretlen azonosítót és a hivatkozás nélküli választ jelzi).
- **Közzététel.** A közlendő kérdés–válaszok kiválasztása; az azonos napi közzététel felülírja az archív oldalt (november 24-én négy közzététel volt).
- **Témanevek.** A gyűjtőtéma jelölése; a nevek futásonként változhatnak (lásd a 2. pontot). Az új, nagyobb adaton érdemes kipróbálni több témát is (`N_TOPICS` 35–40).

---

## 4. Új megállapítások

| Dátum | Hol | Mit tapasztaltál | Javaslat |
|---|---|---|---|
| | | | |
