"""Az összes prompt egy helyen – a dokumentációba innen idézhetők."""

RAG_PROMPT = """Kutatási asszisztens vagy. Kizárólag az alábbi arXiv-cikkek alapján válaszolj.
Minden állítás után tedd zárójelbe a forrás azonosítóját, pl. [2501.01234].
Ha a cikkekből nem derül ki a válasz, mondd meg őszintén. Legfeljebb 200 szóban válaszolj.

CIKKEK:
{context}

KÉRDÉS: {question}

{language_rule}
VÁLASZ:"""

# A válasz nyelve a kérdés nyelve. A promptot magyarul írtuk, a cikkek angolok, ezért a modell magától mindig
# magyarul válaszolt; a nyelvet ezért a kód állapítja meg (rag.detect_language), és külön, a célnyelven írt sor kéri.
LANGUAGE_RULES = {
    "hu": "A válasz nyelve: magyar. A cikkek angolul vannak, de a választ magyarul írd (az angol szakkifejezések maradhatnak).",
    "en": "Answer language: English. The question is in English, so write the ENTIRE answer in English, "
          "even though the instructions above are in Hungarian.",
}

NAMING_PROMPT = """Egy arXiv-trendfigyelő témáit kell elnevezned ({cat} kategória). Minden témához megkapod
a legjellemzőbb kulcsszavakat és néhány friss cikkcímet.

Minden témához adj:
- "nev": rövid magyar szakmai név, utána zárójelben a bevett angol kifejezés, pl. "Tudásdesztilláció (Knowledge Distillation)".
  Legyen konkrét: azt a módszert vagy problémát nevezze meg, ami a kulcsszavakra és a címekre jellemző,
  ne általános területet (NE: "Nyelvi modellek", "NLP-módszerek").
- "leiras": egy mondat (legfeljebb 30 szó) magyarul, amelyből egy nem szakértő is azonnal érti, miről szól a téma.
  Ha a kulcsszavakban konkrét módszer vagy rövidítés szerepel, nevezd meg, és oldd fel zárójelben.

Példa:
Kulcsszavak: opd, opsd, distillation opd, student generated, teacher, distillation
-> {{"nev": "Tudásdesztilláció (Knowledge Distillation)",
    "leiras": "Amikor egy nagyobb „tanár” modell tudását egy kisebb „diák” (student) modellbe sűrítik bele, pl. On-Policy Distillation (OPD)."}}

Csak JSON-t adj vissza, ebben a formában: {{"<téma száma>": {{"nev": "...", "leiras": "..."}}, ...}}

TÉMÁK:
{topics}"""

SUMMARY_PROMPT = """Az alábbi témák a(z) {cat} arXiv-kategóriában az utóbbi hónapokban a leggyorsabban növekvők
(zárójelben a növekedés: az utolsó 3 hónap átlagos részaránya osztva az első 3 hónapéval, mindkettőhöz
0,5 százalékpontot adva, ezért kisebb az egyszerű hányadosnál).
Írj 4-5 mondatos, közérthető magyar összefoglalót arról, merre tart a terület. Hivatkozz a témák nevére.

{topics}"""


# A hivatkozások tartalmi ellenőrzése (verify.py): a válasz hivatkozott állításait egy második hívás veti össze
# a hivatkozott cikkek absztraktjával. Szigorú, hogy a „részben” is jelzés legyen a felhasználónak.
VERIFY_PROMPT = """Egy kutatási asszisztens válaszának állításait ellenőrzöd. Minden állítás mellett ott van, melyik
arXiv-cikkre hivatkozik. Döntsd el állításonként, hogy a hivatkozott cikk(ek) alábbi absztraktja alátámasztja-e.

- "igen": az absztrakt egyértelműen kimondja vagy közvetlenül következik belőle;
- "részben": csak egy része szerepel, vagy az állítás általánosabb, erősebb, mint amit az absztrakt mond;
- "nem": az absztrakt nem mond ilyet, ellentmond neki, vagy a hivatkozott cikk nincs a források között.
Csak az absztraktok szövegét vedd figyelembe, a saját tudásodat ne. Az indoklás egy rövid magyar mondat legyen.

Csak JSON-t adj vissza, ebben a formában: [{{"n": 1, "itelet": "igen|részben|nem", "indoklas": "..."}}, ...]

FORRÁSOK:
{sources}

ÁLLÍTÁSOK:
{claims}"""
