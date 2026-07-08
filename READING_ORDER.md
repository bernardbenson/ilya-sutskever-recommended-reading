# Recommended Reading Order

A **basics → intermediate → advanced** learning progression through Ilya Sutskever's
deep learning reading list. The original list is roughly topical; this reorders it into
a pedagogical sequence. Numbers in parentheses match the filenames in the [`papers/`](./papers) folder.

---

## Tier 1 — Basics (foundations & intuition)
Build core mental models before touching architectures.

1. **CS231n: Convolutional Neural Networks for Visual Recognition** (27) — the course; grounding in neural nets, backprop, optimization, CNNs. Start here if you want the scaffolding.
2. **The Unreasonable Effectiveness of Recurrent Neural Networks** (03) — Karpathy; intuition for why sequence models work.
3. **Understanding LSTM Networks** (04) — Olah; the clearest explanation of gating/memory.
4. **ImageNet Classification with Deep CNNs (AlexNet)** (08) — the paper that kicked off modern deep learning.
5. **Recurrent Neural Network Regularization** (05) — short, practical (dropout in RNNs); easy on-ramp to reading papers.

## Tier 2 — Intermediate (core architectures & scaling)
The workhorses of modern deep learning.

6. **Deep Residual Learning for Image Recognition (ResNet)** (11)
7. **Identity Mappings in Deep Residual Networks** (16) — the "why ResNets work" follow-up.
8. **Multi-Scale Context Aggregation by Dilated Convolutions** (12)
9. **Neural Machine Translation by Jointly Learning to Align and Translate** (15) — where attention was born.
10. **Attention Is All You Need** (14) — the Transformer. The keystone paper.
11. **The Annotated Transformer** (01) — implement the Transformer line-by-line right after reading it (see tutorials in [`learning/`](./learning)).
12. **Pointer Networks** (07)
13. **Order Matters: Sequence to Sequence for Sets** (09)
14. **Neural Message Passing for Quantum Chemistry** (13) — graph neural networks.
15. **Deep Speech 2** (22) — end-to-end systems at scale.
16. **GPipe: Micro-Batch Pipeline Parallelism** (10) — how large models are trained across devices.
17. **Scaling Laws for Neural Language Models** (23) — the empirical bridge into "why scale matters."

## Tier 3 — Advanced (memory, reasoning, generative & theory)
Harder architectures and the information-theoretic worldview that ties the list together.

18. **Neural Turing Machines** (21) — external memory / differentiable computation.
19. **A Simple Neural Network Module for Relational Reasoning** (17)
20. **Relational Recurrent Neural Networks** (19)
21. **Variational Lossy Autoencoder** (18) — generative modeling + latent variables.
22. **Keeping Neural Networks Simple by Minimizing the Description Length of the Weights** (06) — Hinton; the MDL bridge.
23. **A Tutorial Introduction to the Minimum Description Length Principle** (24) — full theory of MDL.
24. **The First Law of Complexodynamics** (02) — Aaronson.
25. **Quantifying the Rise and Fall of Complexity (Coffee Automaton)** (20) — Aaronson.
26. **Kolmogorov Complexity and Algorithmic Randomness** (26) — the theoretical bedrock.
27. **Machine Super Intelligence** (25) — Legg; the philosophical capstone (compression ≈ intelligence).

---

**Why this shape:** Tier 3 clusters the theory papers (MDL, Kolmogorov complexity,
complexodynamics) at the end on purpose — they're the intellectual "punchline" of the
list (intelligence as compression), and they land far better once you've seen the
architectures they're meant to explain.
