**EN02 – közepes**

**Kérdés**
What can a RAG system do when highly relevant but near-duplicate passages waste its context-token budget?

**Válasz**
Ha egy RAG (Retrieval-Augmented Generation) rendszerben a magasan releváns, de egymáshoz nagyon hasonló (redundáns) részletek pazarolják a kontextushoz tartozó tokinkeretet, a rendszer használhatja az AdaGReS keretrendszert [\[2512.25052\]](https://arxiv.org/abs/2512.25052).

Az AdaGReS egy olyan redundanciatudatos kontextus-kiválasztási módszer token-keretes RAG-hez, amely optimalizálja a halmazszintű célfüggvényt a lekérdezés és a szövegrészlet relevanciája, valamint a halmaz belüli redundancia büntetése alapján [\[2512.25052\]](https://arxiv.org/abs/2512.25052). A keretrendszer mohó kiválasztást (greedy selection) végez a tokinkeret-korlát betartásával a célfüggvényből származó marginális nyereségek felhasználásával, és egy zárt alakú, példányhoz adaptív kalibrációt alkalmaz a relevanciárangi és redundancia közötti kompromisszum szabályozására [\[2512.25052\]](https://arxiv.org/abs/2512.25052). Ez kiküszöböli a manuális hangolást, és alkalmazkodik a jelöltállomány statisztikáihoz és a költségvetési korlátokhoz [\[2512.25052\]](https://arxiv.org/abs/2512.25052).

**Források**

1. [2512.25052](https://arxiv.org/abs/2512.25052) – AdaGReS:Adaptive Greedy Context Selection via Redundancy-Aware Scoring for Token-Budgeted RAG (2025-12)
2. [2611.00871](https://arxiv.org/abs/2611.00871) – ShiftRAG: Bypassing the Textual Bottleneck via Decoupled Learning and Continuous Soft Tokens (2026-10)
3. [2606.29090](https://arxiv.org/abs/2606.29090) – AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering (2026-06)
4. [2602.00344](https://arxiv.org/abs/2602.00344) – When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs (2026-01)
5. [2610.37826](https://arxiv.org/abs/2610.37826) – DeFT-RAG: Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers (2026-10)
