"""Az összes prompt egy helyen – a dokumentációba innen idézhetők."""

RAG_PROMPT = """Kutatási asszisztens vagy. Kizárólag az alábbi arXiv-cikkek alapján válaszolj.
Minden állítás után tedd zárójelbe a forrás azonosítóját, pl. [2501.01234].
Ha a cikkekből nem derül ki a válasz, mondd meg őszintén. Válaszolj a kérdés nyelvén, legfeljebb 200 szóban.

CIKKEK:
{context}

KÉRDÉS: {question}
VÁLASZ:"""

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
(zárójelben: az utolsó 3 hónap részaránya osztva az első 3 hónapéval).
Írj 4-5 mondatos, közérthető magyar összefoglalót arról, merre tart a terület. Hivatkozz a témák nevére.

{topics}"""
