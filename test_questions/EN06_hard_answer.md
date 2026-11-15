**EN06 – nehéz**

**Kérdés**
How can on-policy distillation use a black-box teacher without access to token logits, while avoiding unreliable supervision and policy collapse?

**Válasz**
To use a black-box teacher without access to token logits while avoiding unreliable supervision and policy collapse, the OmniOPD framework replaces deterministic logit matching with a logit-free, chunk-level supervision signal [\[2606.01476\]](https://arxiv.org/abs/2606.01476). Specifically, it utilizes Monte Carlo rollouts that approximate the teacher's local preferences through a continuous semantic similarity metric over multi-token chunks [\[2606.01476\]](https://arxiv.org/abs/2606.01476). To avoid unreliable supervision, it concentrates this supervision via a peak-entropy scheduler that audits the student exclusively at its high-uncertainty reasoning forks [\[2606.01476\]](https://arxiv.org/abs/2606.01476). Finally, to prevent policy collapse across unaudited tokens, it employs a Dirichlet-Multinomial Bayesian prior combined with a base-model KL anchor to bound the variance of discrete sampling [\[2606.01476\]](https://arxiv.org/abs/2606.01476).

**Források**

1. [2603.25562](https://arxiv.org/abs/2603.25562) – Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes (2026-03)
2. [2606.01476](https://arxiv.org/abs/2606.01476) – OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification (2026-05)
3. [2611.00852](https://arxiv.org/abs/2611.00852) – Adaptive-Policy Distillation: Knowledge Injection under Information Asymmetry (2026-10)
4. [2606.01249](https://arxiv.org/abs/2606.01249) – Trust Region On-Policy Distillation (2026-05)
5. [2606.28562](https://arxiv.org/abs/2606.28562) – SEAD: Competence-Aware On-Policy Distillation via Entropy-Guided Supervision (2026-06)
