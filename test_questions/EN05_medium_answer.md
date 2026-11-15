**EN05 – közepes**

**Kérdés**
How do LLM essay graders differ from human raters when scoring grammar and conventions, and what does the research suggest about prompt length and bias correction?

**Válasz**
When scoring Lower-Order Concern (LOC) traits such as grammar and conventions, LLM essay graders exhibit a large and stable negative directional bias, meaning they score these traits more harshly than human raters [\[2604.00259\]](https://arxiv.org/abs/2604.00259).

Regarding prompt length, research shows that concise, keyword-based prompts generally outperform longer rubric-style prompts in multi-trait analytic scoring [\[2604.00259\]](https://arxiv.org/abs/2604.00259).

To address systematic deviations like bias, the findings support a "bias-correction-first deployment strategy" [\[2604.00259\]](https://arxiv.org/abs/2604.00259). Instead of relying on raw zero-shot scores, systematic score offsets can be estimated and corrected using small human-labeled bias-estimation sets, which avoids the need for large-scale fine-tuning [\[2604.00259\]](https://arxiv.org/abs/2604.00259). Furthermore, LOC bias can often be detected using very small validation sets [\[2604.00259\]](https://arxiv.org/abs/2604.00259).

**Források**

1. [2604.00259](https://arxiv.org/abs/2604.00259) – LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias (2026-03)
2. [2512.19456](https://arxiv.org/abs/2512.19456) – Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations (2025-12)
3. [2606.29689](https://arxiv.org/abs/2606.29689) – Can MLLMs Critique Like Humans? Evaluating Open-Ended Aesthetic Reasoning in Multimodal Large Language Models (2026-06)
4. [2610.41391](https://arxiv.org/abs/2610.41391) – Comparative Evaluation of LLM-Based Review Paradigms for Academic Paper Quality Assessment (2026-10)
5. [2608.29517](https://arxiv.org/abs/2608.29517) – LLM Judges as Raters: A Pre-Registered Audit of Severity, Halo, Reliability, and Version Instability in LLM Essay Scoring on Public Corpora (2026-08)
