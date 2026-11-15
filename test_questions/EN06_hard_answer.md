**EN06 – nehéz**

**Kérdés**
How can on-policy distillation use a black-box teacher without access to token logits, while avoiding unreliable supervision and policy collapse?

**Válasz**
Az OmniOPD keretrendszer lehetővé teszi egy feketodob (black-box) tanár modell használatát token-logitek elérése nélkül [\[2606.01476\]](https://arxiv.org/abs/2606.01476). Ez Monte Carlo-mintavételezésen alapuló gurításokkal (rollouts) történik, amelyek többtokenes blokkokon átívelő, folyamatos szemantikai hasonlósági metrikával közelítik meg a tanár helyi preferenciáit [\[2606.01476\]](https://arxiv.org/abs/2606.01476).

A megbízhatatlan felügyelet elkerülése és a policy-összeomlás megakadályozása érdekében az OmniOPD a következő módszereket alkalmazza:

1. Egy csúcs-entrópia ütemező (peak-entropy scheduler) alkalmazása, amely csak a hallgató nagy bizonytúságú rezonáló elágazásainál (reasoning forks) ellenőrzi a modellt [\[2606.01476\]](https://arxiv.org/abs/2606.01476).
2. Egy Dirichlet-Multinomial Bayes-féle prior (Dirichlet-Multinomial Bayesian prior) használata [\[2606.01476\]](https://arxiv.org/abs/2606.01476).
3. Egy bázismodell KL-horgony (base-model KL anchor) alkalmazása, amelyek korlátozzák a diszkrét mintavételezés varianciáját, és megakadályozzák a policy összeomlását az ellenőrizetlen (unaudited) tokenek esetében [\[2606.01476\]](https://arxiv.org/abs/2606.01476).

**Források**

1. [2603.25562](https://arxiv.org/abs/2603.25562) – Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes (2026-03)
2. [2606.01476](https://arxiv.org/abs/2606.01476) – OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification (2026-05)
3. [2611.00852](https://arxiv.org/abs/2611.00852) – Adaptive-Policy Distillation: Knowledge Injection under Information Asymmetry (2026-10)
4. [2606.01249](https://arxiv.org/abs/2606.01249) – Trust Region On-Policy Distillation (2026-05)
5. [2606.28562](https://arxiv.org/abs/2606.28562) – SEAD: Competence-Aware On-Policy Distillation via Entropy-Guided Supervision (2026-06)
