# 01 — The Annotated Transformer

**Author:** Sasha Rush, et al. (Harvard NLP)
**Source:** [Blog](https://nlp.seas.harvard.edu/annotated-transformer/) · [Code](https://github.com/harvardnlp/annotated-transformer/)
**Underlying paper:** Vaswani et al., *Attention Is All You Need* (2017) — see paper 14 in this list.

---

## What it is

The Annotated Transformer is a **line-by-line, runnable PyTorch reimplementation** of the original Transformer paper. Every equation from *Attention Is All You Need* is placed next to the code that implements it. It is the single best on-ramp to the architecture that underlies essentially all modern LLMs — so it is deliberately paper #1 on this list.

The goal for you is **not** to read it, but to *rebuild it from scratch* until the whole encoder–decoder stack fits in your head.

---

## The big idea

Before 2017, sequence transduction (e.g. translation) used RNNs/LSTMs that process tokens **sequentially**, which is slow and struggles with long-range dependencies. The Transformer throws recurrence out entirely and replaces it with **attention**: every position can look at every other position in a single, fully-parallel operation.

> "Attention is all you need" — no recurrence, no convolution. Just attention + feed-forward layers + normalization, stacked.

---

## Architecture at a glance

An **encoder–decoder** model, each side a stack of `N = 6` identical layers.

```
        INPUT                              OUTPUT (shifted right)
          │                                      │
   Input Embedding                        Output Embedding
          │                                      │
   + Positional Encoding                  + Positional Encoding
          │                                      │
   ┌──────────────┐                       ┌──────────────────┐
   │  Encoder ×6  │                       │   Decoder ×6     │
   │              │                       │                  │
   │ Multi-Head   │                       │ Masked Multi-Head│
   │  Self-Attn   │──────────┐            │   Self-Attn      │
   │   + Add&Norm │          │            │   + Add&Norm     │
   │              │          └──────────► │ Cross-Attn (enc  │
   │ Feed-Forward │      (keys/values)    │   keys/values)   │
   │   + Add&Norm │                       │   + Add&Norm     │
   └──────────────┘                       │ Feed-Forward     │
          │                               │   + Add&Norm     │
      memory ──────────────────────────► └──────────────────┘
                                                 │
                                          Linear + Softmax
                                                 │
                                          next-token probs
```

## The components you must be able to implement

1. **Scaled Dot-Product Attention**
   `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`
   The `√d_k` scaling keeps dot products from growing large and pushing softmax into low-gradient regions.

2. **Multi-Head Attention** — run attention `h = 8` times in parallel on `d_k = d_model/h = 64` dim projections, concatenate, project back. Lets the model attend to different "representation subspaces" at once.

3. **Position-wise Feed-Forward** — two linear layers with a ReLU: `max(0, xW₁ + b₁)W₂ + b₂`. Inner dim `d_ff = 2048`. Applied identically to every position.

4. **Positional Encoding** — since there's no recurrence, position is injected with fixed sinusoids: `PE(pos, 2i) = sin(pos/10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`. Lets the model use relative positions.

5. **Residual connection + LayerNorm** around every sub-layer: `LayerNorm(x + Sublayer(x))`. This "Add & Norm" is what makes deep stacks trainable.

6. **Embeddings + final linear** — token embeddings scaled by `√d_model`; the output projection weight is tied to the embedding.

7. **Masking** — a *source/pad mask* hides padding; a *subsequent mask* (lower-triangular) stops the decoder from peeking at future tokens (autoregressive property).

## Training details worth internalizing

- **Optimizer:** Adam with a custom LR schedule — warm up linearly for `warmup_steps` (4000), then decay ∝ `step^-0.5`.
- **Regularization:** dropout (0.1) on sub-layer outputs, embeddings, and attention weights; **label smoothing** (ε = 0.1) which *hurts* perplexity but *helps* accuracy/BLEU.
- **Base model:** `d_model=512, N=6, h=8, d_ff=2048` ≈ 65M params.
- **Greedy decoding** for inference in the tutorial (real systems use beam search).

## Why it matters

- It is the direct ancestor of GPT (decoder-only), BERT (encoder-only), T5 (full enc-dec), and every frontier LLM.
- Attention + residuals + LayerNorm is the reusable "transformer block" pattern you will see reappear in papers 14, 16, 23 of this list.
- Understanding masking and the LR warmup schedule saves you from the two most common bugs when people first build one.

## Key takeaways

- Parallelism over sequences is the whole point — attention replaces the sequential bottleneck of RNNs.
- The architecture is remarkably *simple*: a handful of linear layers, one softmax, sinusoids, and residual/norm plumbing.
- Almost every "trick" (scaling by √d_k, warmup, label smoothing, weight tying) exists to make optimization stable — the modeling idea itself is small.

---

*Next: work through `EXERCISES.md` and build it yourself.*
