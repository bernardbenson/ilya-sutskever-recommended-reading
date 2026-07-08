# Tutorial 04 — Understand by breaking (Tier 4)

These are the exercises where the learning actually lands. Each one changes **one
thing**, predicts the outcome, then runs it. Don't read ahead to the answer — form
a hypothesis first, then check it against the "What you should see."

**How to run an ablation.** Copy `solutions/transformer.py` and `train_copy.py`
into a scratch dir, make the one change, rerun `train_copy.py`, and compare the
loss curve + `exact copy` result against the baseline (loss ~2.2 → ~0.23,
`exact copy: True`).

A tiny hook for inspecting attention: after a forward pass, every
`MultiHeadAttention` caches its weights in `self.attn` with shape
`(batch, heads, seq, seq)`. Grab e.g. `model.encoder.layers[0].self_attn.attn`.

---

## 1. Remove the `√d_k` scaling

Change `scores = QKᵀ / √d_k` to just `scores = QKᵀ`.

**Hypothesis to form:** what happens to the size of the scores as `d_k = 64`
grows, and what does that do to softmax?

**What you should see.** Training is slower and noisier, sometimes stalling. The
attention weights become nearly one-hot (softmax saturates on the largest score),
so gradients through the attention weights shrink. The effect is bigger at larger
`d_model`/`d_k`. This is *why the scaling exists* — it keeps score variance ~1.

---

## 2. Remove positional encoding

Delete the `+ pe` addition (make `PositionalEncoding.forward` return `x`).

**Two sub-experiments:**
- **Copy** (`src == tgt`): still learns fine. Copying is position-agnostic — token
  `k` of the output just needs token `k` of the input, and cross-attention can
  match on *content*.
- **Reverse** (`tgt = src[::-1]`): now the model needs to know *where* each token
  is to place it. Without positional information it struggles badly.

**Lesson.** Attention alone is permutation-invariant; positional encoding is what
lets order matter. The task determines whether you notice its absence.

---

## 3. Drop the subsequent (causal) mask

In `make_std_mask`, stop AND-ing in `subsequent_mask` (return just the pad mask).

**What you should see.** Training loss drops *faster and lower than baseline* —
suspiciously good. That's the tell: the decoder is now attending to future
positions, i.e. peeking at the token it's asked to predict. Then **greedy decode
falls apart**, because at inference there is no future to peek at. Great train
loss + broken generation = missing causal mask. This is the most common
show-stopper bug when people build their first decoder.

---

## 4. `h = 1` vs `h = 8` (d_model fixed)

Set `h=1` in `make_model`. Same parameter count (`W_o` etc. are `d_model×d_model`
regardless of `h`), but now a single 512-dim attention instead of eight 64-dim
ones.

**What you should see.** On the toy copy task both converge (the task is easy), but
`h=8` typically converges a bit faster/cleaner. The real point is *representational*:
inspect `self.attn` — with 8 heads you can see different heads forming different
alignment patterns; with 1 head everything is forced through a single attention
distribution. Multi-head buys you multiple "relationships" per layer for free.

---

## 5. Post-norm vs pre-norm

The solution uses **pre-norm**: `x + Sublayer(LayerNorm(x))`. Switch
`SublayerConnection.forward` to **post-norm**: `LayerNorm(x + Sublayer(x))`.

**What you should see.** With the Noam warmup schedule, both train. Now **remove
warmup** (set `warmup` tiny, or use a flat LR): post-norm is much more likely to
diverge or stall early, while pre-norm stays stable. This is exactly why modern
LLMs moved to pre-norm — it keeps a clean residual highway and is forgiving of the
LR schedule. Connects directly to paper 16, *Identity Mappings in Deep Residual
Networks*, which makes the same argument for ResNets.

---

## 6. Tie vs untie output projection and input embedding

Weight tying: set `model.generator.proj.weight = model.tgt_embed[0].lut.weight`
(both are `vocab × d_model`).

**What you should see.** Parameter count drops by `vocab × d_model`. On a real
translation task tying usually *helps* generalization (the input and output token
spaces are the same language of embeddings) and is standard practice; on the toy
copy task the effect is small but the parameter saving is real. Verify shapes are
compatible and that gradients flow to the shared matrix from both places.

---

## Deliverable: `notes.md`

For each ablation record: (a) your hypothesis, (b) the loss curve / `exact copy`
result, (c) an attention heatmap where relevant, (d) one sentence on *why*. That
write-up is the artifact that proves you understand the architecture — not just
that you can run it.

---

## Where this goes next

You now hold the reusable "transformer block" (attention + residual + norm +
FFN). You'll see it again, barely changed, in:
- **Paper 14** — *Attention Is All You Need* (the source; read it now that you've
  built it — it'll feel obvious).
- **Paper 16** — *Identity Mappings in Deep Residual Networks* (why pre-norm/clean
  residuals matter).
- **Paper 23** — *Scaling Laws* (what happens when you make this exact block huge).
