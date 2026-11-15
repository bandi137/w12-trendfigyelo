**EN02 – közepes**

**Kérdés**
What can a RAG system do when highly relevant but near-duplicate passages waste its context-token budget?

**Válasz**
When a RAG system faces redundant or near-duplicate passages that waste its token budget, it can use **AdaGReS**, a redundancy-aware context selection framework [\[2512.25052\]](https://arxiv.org/abs/2512.25052). AdaGReS optimizes a set-level objective that combines query-chunk relevance with intra-set redundancy penalties [\[2512.25052\]](https://arxiv.org/abs/2512.25052). It performs greedy selection under a token-budget constraint using marginal gains, and it uses a closed-form, instance-adaptive calibration of the relevance-redundancy trade-off parameter to eliminate manual tuning and adapt to candidate-pool statistics and budget limits [\[2512.25052\]](https://arxiv.org/abs/2512.25052).

**Források**

1. [2512.25052](https://arxiv.org/abs/2512.25052) – AdaGReS:Adaptive Greedy Context Selection via Redundancy-Aware Scoring for Token-Budgeted RAG (2025-12)
2. [2611.00871](https://arxiv.org/abs/2611.00871) – ShiftRAG: Bypassing the Textual Bottleneck via Decoupled Learning and Continuous Soft Tokens (2026-10)
3. [2606.29090](https://arxiv.org/abs/2606.29090) – AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering (2026-06)
4. [2602.00344](https://arxiv.org/abs/2602.00344) – When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs (2026-01)
5. [2610.37826](https://arxiv.org/abs/2610.37826) – DeFT-RAG: Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers (2026-10)
