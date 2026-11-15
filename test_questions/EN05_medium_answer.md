**EN05 – közepes**

**Kérdés**
How do LLM essay graders differ from human raters when scoring grammar and conventions, and what does the research suggest about prompt length and bias correction?

**Válasz**
A kutatás szerint a nagy nyelvi modellek (LLM-ek) stabil és jelentős negatív irányú torzítást mutatnak az alacsonyabb szintű nyelvi szempontok (Lower-Order Concern, LOC), például a nyelvhelyesség és a konvenciók (Grammar and Conventions) értékelésében, ami azt jelenti, hogy szigorúbban pontozzák ezeket a területeket, mint az emberi értékelők [\[2501.01234\]](https://arxiv.org/abs/2501.01234).

Ami a promptok hosszát illeti, a többszempontú (analytic) értékelésben a tömör, kulcsszóalapú promptok általában jobban teljesítenek, mint a hosszabb, értékelési skálát (rubric) leíró promptok [\[2501.01234\]](https://arxiv.org/abs/2501.01234).

A torzítás kezelésére a kutatás egy „torzításkorrekció-azelőtt” (bias-correction-first) stratégiát javasol: a nyers, zero-shot pontszámok helyett a szisztematikus eltérések becsülhetők és korrigálhatók kis méretű, ember által címkézett torzításbecslő halmazok segítségével, anélkül hogy nagyszabású finomhangolásra lenne szükség [\[2501.01234\]](https://arxiv.org/abs/2501.01234).

> ⚠ A modell nem kapott forrásra is hivatkozott: 2501.01234

**Források**

1. [2604.00259](https://arxiv.org/abs/2604.00259) – LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias (2026-03)
2. [2512.19456](https://arxiv.org/abs/2512.19456) – Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations (2025-12)
3. [2606.29689](https://arxiv.org/abs/2606.29689) – Can MLLMs Critique Like Humans? Evaluating Open-Ended Aesthetic Reasoning in Multimodal Large Language Models (2026-06)
4. [2610.41391](https://arxiv.org/abs/2610.41391) – Comparative Evaluation of LLM-Based Review Paradigms for Academic Paper Quality Assessment (2026-10)
5. [2608.29517](https://arxiv.org/abs/2608.29517) – LLM Judges as Raters: A Pre-Registered Audit of Severity, Halo, Reliability, and Version Instability in LLM Essay Scoring on Public Corpora (2026-08)
