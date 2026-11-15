**HU10 – nehéz**

**Kérdés**
Egy pruninggal kisebbített nyelvi modell első válasza gyakrabban hibás. Ebből következik-e, hogy elvesztette a korábbi következtetési képességét?

**Válasz**
A megadott cikkek alapján a kérdésre nincs közvetlen válasz, mivel a szövegek nem vizsgálnak pruninggal (metszéssel) kisebbített nyelvi modelleket.

A cikkek ugyanakkor mutatnak példákat arra, hogy a kisebb vagy korlátozottabb modellek (vagy a poszt-tréninggel csökkentett nyelvi kompetenciájú rendszerek) mögött nem feltétlenül az explicit tudás teljes elvesztése áll. Például a poszt-tréningen átesett modellek megtartanak olyan nyelvi tudást is, amelyet explicit promptolással nem tudnak kifejezni [\[2610.00540\]](https://arxiv.org/abs/2610.00540). Hasonlóképpen, a konfabuláló vagy tévedő modellek esetében a belső rejtett állapotok vizsgálata azt mutatja, hogy a válasz mögött meghúzódó helyes tudás vagy irányíthatóság bizonyos esetekben fennmarad, és a hiba inkább a generálási folyamatban vagy a beszédcselekvés kapuzásában rejlik [\[2611.01272\]](https://arxiv.org/abs/2611.01272). A nagyobb modellek megbízhatóságával kapcsolatban pedig megjegyzik, hogy a hibák (hallucinációk) felerősödése nem feltétlenül jelenti a tudás kezdeti hiányát, hanem a válaszadás közbeni romlási folyamatokhoz kapcsolódik [\[2607.18292\]](https://arxiv.org/abs/2607.18292).

**Források**

1. [2610.00540](https://arxiv.org/abs/2610.00540) – Assessing the Impact of Language Disparity on Multilingual Linguistic Ability in Large Language Models (2026-09)
2. [2609.36214](https://arxiv.org/abs/2609.36214) – Lost in Translation: Measuring the Effect of Non-Native English on End User Performance of Large Language Models (2026-09)
3. [2611.01272](https://arxiv.org/abs/2611.01272) – Geometry of Confabulation and Refusal in Small Language Models (2026-10)
4. [2604.25249](https://arxiv.org/abs/2604.25249) – Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance (2026-04)
5. [2607.18292](https://arxiv.org/abs/2607.18292) – Reliability Scales Inversely: Hallucinations Snowball Faster in Bigger Language Models (2026-06)
