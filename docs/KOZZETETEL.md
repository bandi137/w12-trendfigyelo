# Közzététel a GitHub Pages-re

A notebook 7. szakaszában indított webes felület **Közzététel** gombja a jelentést statikus weboldalként feltölti ide:
**https://bandi137.github.io/w12-trendfigyelo/**

Ehhez mindenkinek kell egy saját GitHub-token a Colab *Secrets* fülén, `GITHUB_TOKEN` néven.

## 1. Token készítése

**Sümegi András (bandi137, a repó tulajdonosa) – finomhangolt token**

1. GitHub → *Settings* → *Developer settings* → *Personal access tokens* → *Fine-grained tokens* → *Generate new token*.
2. Név: `w12-colab`; lejárat: 30 nap; *Resource owner*: `bandi137`.
3. *Repository access*: *Only select repositories* → `w12-trendfigyelo`.
4. *Permissions* → *Repository permissions* → **Contents: Read and write** (a *Metadata: Read* magától bekerül).
5. *Generate token*, és másold ki (csak egyszer látszik).

**Süle Szilvia (SuleSz, közreműködő) – klasszikus token**

A finomhangolt token közreműködőként nem működik más felhasználó repójában, ezért itt klasszikus token kell.

1. Előfeltétel: a közreműködői meghívás elfogadva (github.com/bandi137/w12-trendfigyelo/invitations).
2. GitHub → *Settings* → *Developer settings* → *Personal access tokens* → *Tokens (classic)* → *Generate new token (classic)*.
3. Név: `w12-colab`; lejárat: 30 nap; jogosultság: csak **`public_repo`**.
4. *Generate token*, és másold ki.

## 2. A token felvétele a Colabba

1. A notebookban bal oldalt a 🔑 *Secrets* fül → *Add new secret*.
2. Név: `GITHUB_TOKEN`, érték: a token. Kapcsold be a *Notebook access* kapcsolót.
3. A titok a te Google-fiókodhoz tartozik, a notebook megosztásakor nem kerül át senkihez.

## 3. Mi történik a gombnyomáskor

- Ha még nincsenek kiszámolva a trendek, vagy nincs összefoglaló, a felület elkészíti őket (egy-két Gemini-hívás).
- A *Közzététel* fülön jelölőnégyzetekkel választható ki, mely kérdés–válaszok kerüljenek a jelentésbe (alapból
  mind; a próbakérdéseket érdemes kivenni). Az *Előnézet (letöltés)* gomb a kijelöltekkel készült változatot adja.
- Egyetlen commit kerül a `docs/` mappába: `index.html` (a legfrissebb jelentés) és `archiv/ÉÉÉÉ-HH-NN_ÓÓPP.html`
  (ennek a közzétételnek a példánya, magyar idő szerint). A commit szerzője a token tulajdonosa. Az aznapi újabb
  közzététel külön archív oldalt kap, a korábbit nem írja felül.
- **Archív jelentés törlése:** a *Közzététel* fül alján, az *Archív jelentések törlése* részben. *Archívum listázása*,
  a törlendők kijelölése, a megerősítő jelölőnégyzet, majd *Kijelöltek törlése*. Egy commitban törli a fájlokat, és a
  kezdőoldal, valamint a többi archív oldal „Korábbi jelentések” listájából a rájuk mutató linkeket is kiveszi (a
  jelentések tartalmát nem számolja újra). Ha a legutóbbi közzététel archív példányát törlöd, a kezdőoldal attól még
  azt mutatja a következő közzétételig. A törölt fájl a git-előzményekben megmarad (a repó nyilvános).
- A GitHub Pages 1–2 perc alatt frissül. A kód (`trendfigyelo/`, notebook) nem változik, így nincs ütközés.

## Biztonság

- A tokent soha ne írd notebook-cellába, chatbe vagy e-mailbe, csak a *Secrets* fülre.
- A beadás után, vagy ha kiszivárgott, töröld a tokent a GitHubon (*Delete*/*Revoke*).
- A weboldal nyilvános: az arXiv-adatok mellett a **kijelölt kérdések és válaszok** is felkerülnek, ezért ne
  kérdezz személyes adatot tartalmazó dolgot.
