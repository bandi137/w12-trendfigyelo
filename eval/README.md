# Tesztkérdések (`kerdesek.csv`)

A notebook 6. szakasza ebből a fájlból méri, hogy a keresés megtalálja-e a jó cikkeket (hit@1, hit@5, hit@10).
A fájl a GitHub webes felületén szerkeszthető (ceruza ikon), a notebook első cellája minden futáskor lehúzza.

## Formátum

```csv
vart_cikkek,keszitette,kerdes
2512.19070; 2602.22144,SuleSz,Milyen tanítás nélküli módszerek csökkentik a képi nyelvi modellek hallucinációit?
2610.36980,bandi137,How are LLM agents evaluated on long-horizon tasks?
```

| Oszlop | Tartalom |
| --- | --- |
| `vart_cikkek` | Egy vagy több arXiv-azonosító, amelynek a találatok között kell lennie. Elválasztó: `;` vagy szóköz. Verziószám (`v2`) és `arXiv:` előtag nem gond. |
| `keszitette` | Ki írta a kérdést (GitHub-név), a munkamegosztás dokumentálásához. |
| `kerdes` | A kérdés, magyarul vagy angolul. Mindig az utolsó oszlop, így vessző is lehet benne idézőjel nélkül. |

## Hogyan válassz várt cikkeket

1. A notebook 6. szakaszának segédcellájában keress rá a témára kulcsszóval **és** a `rag.retrieve()`-vel is.
2. Olvasd el a címeket és az absztraktokat, és csak azt írd be, ami **tényleg válaszol a kérdésre**.
3. Ne csak a `rag.retrieve()` első találatait másold be, különben a mérés önmagát igazolja.

Cél: 15–20 kérdés, vegyesen magyar és angol, könnyű és nehéz. A betöltő figyelmeztet, ha egy azonosító nincs az adatok között (elírás).
