**HU07 – közepes**

**Kérdés**
Hogyan maradhatnak a személyes adatok a saját eszközön úgy, hogy egy külső LLM-nek küldött kérdés az egyszerű kitakaráshoz képest jobban megőrizze az értelmét?

**Válasz**
Az AlienLM nevű, bevethető API-alapú expozíciócsökkentő réteg használatával [\[2601.22710\]](https://arxiv.org/abs/2601.22710). Ez a módszer úgy csökkenti a nyílt szöveges (plaintext) adatok küldését, hogy a szöveget egy szótári skálájú bijekció segítségével egy Idegen Nyelvre (Alien Language) fordítja át, ami a kliens oldalon veszteségmentesen helyreállítható [\[2601.22710\]](https://arxiv.org/abs/2601.22710). A célmodelleket egy standard finomhangolási API-kat használó, Idegen Adaptációs Tréning (Alien Adaptation Training – AAT) nevű eljárással készítik fel arra, hogy közvetlenül a leidegenített bemeneteken működjenek [\[2601.22710\]](https://arxiv.org/abs/2601.22710). Ennek köszönhetően az AlienLM megőrzi a sima szöveges teljesítmény több mint 81%-át, miközben jelentősen csökkenti az adatok kitakarásos expozícióját a feketedoboz LLM-API-k elérése során [\[2601.22710\]](https://arxiv.org/abs/2601.22710).

**Források**

1. [2609.36414](https://arxiv.org/abs/2609.36414) – Eternal Sunshine of the Spotless Mind: Systematically Erasing LLM's Memories (2026-09)
2. [2610.36948](https://arxiv.org/abs/2610.36948) – Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models (2026-10)
3. [2512.23836](https://arxiv.org/abs/2512.23836) – Retrieval Augmented Question Answering: When Should LLMs Admit Ignorance? (2025-12)
4. [2606.28898](https://arxiv.org/abs/2606.28898) – PASTA: A Paraphrasing And Self-Training Approach for Knowledge Updating in LLMs (2026-06)
5. [2601.22710](https://arxiv.org/abs/2601.22710) – AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs (2026-01)
