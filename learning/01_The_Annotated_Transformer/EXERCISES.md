# Exercises — Implement the Transformer

Goal: rebuild the Transformer from *Attention Is All You Need* **from scratch** in PyTorch, then train it on a toy task. Don't copy the Annotated Transformer — write each piece yourself, then diff against it.

**Rules of engagement**
- Only allowed imports: `torch`, `torch.nn`, `math`, `numpy`. No `nn.Transformer`, no `nn.MultiheadAttention`.
- After each exercise, check the referenced shape assertions pass.
- Keep everything in one file `transformer.py` you build up incrementally.

---

## Tier 1 — Core building blocks

### 1.1 Scaled dot-product attention
Implement:
```python
def attention(query, key, value, mask=None, dropout=None):
    # query,key,value: (batch, heads, seq, d_k)
    # returns: (output (batch, heads, seq, d_k), attn_weights (batch, heads, seq, seq))
```
- Compute `scores = QKᵀ / √d_k`.
- If `mask` is provided, set masked positions to `-1e9` **before** softmax.
- Apply softmax over the last dim, optional dropout, then multiply by `V`.

**Check:** with `Q=K=V` random `(2, 8, 5, 64)` and no mask, output shape is `(2,8,5,64)` and each attention row sums to ~1.0.

### 1.2 Multi-head attention
Wrap 1.1 in an `nn.Module` with 4 linear layers (`W_q, W_k, W_v, W_o`), `h=8`, `d_model=512`.
- Project inputs, reshape to `(batch, h, seq, d_k)`, apply `attention`, concat heads back to `(batch, seq, d_model)`, apply `W_o`.

**Check:** input `(2, 5, 512)` → output `(2, 5, 512)`. Verify param count = `4 * 512 * 512 + 4*512` (with bias).

### 1.3 Position-wise feed-forward
`FFN(x) = Linear(dropout(relu(Linear(x))))` with `d_model=512`, `d_ff=2048`.

**Check:** input `(2, 5, 512)` → output `(2, 5, 512)`.

### 1.4 Positional encoding
Implement the fixed sinusoidal encoding as a buffer (not a parameter). Add it to the (scaled) embeddings.

**Check:** plot rows 0, 5, 20 of the PE matrix — they should differ; `pe[:, 0::2]` are sines, `pe[:, 1::2]` cosines. Confirm no gradient flows to `pe`.

### 1.5 Layer norm + residual ("Add & Norm")
Write your **own** LayerNorm (mean/var over last dim, learned γ/β) — don't use `nn.LayerNorm`. Then a `SublayerConnection` that does `x + dropout(sublayer(norm(x)))`.

**Check:** LayerNorm output has ~0 mean and ~1 std along the feature dim.

---

## Tier 2 — Assemble the stack

### 2.1 Encoder layer & Encoder
One encoder layer = self-attention sublayer → feed-forward sublayer (each wrapped in Add&Norm). Stack `N=6` with a final norm.

### 2.2 Decoder layer & Decoder
One decoder layer = **masked** self-attention → **cross-attention** over encoder memory → feed-forward. Stack `N=6`.

### 2.3 Masks
- `pad_mask(seq, pad_idx)` → hides padding tokens.
- `subsequent_mask(size)` → lower-triangular, hides future positions.

**Check:** `subsequent_mask(4)` equals the lower-triangular ones matrix; assert position `i` can only see `j ≤ i`.

### 2.4 Full EncoderDecoder model
Wire embeddings + positional encoding + encoder + decoder + final `Linear→log_softmax` generator. Expose `encode()`, `decode()`, and `forward()`.

**Check:** feed a batch of random token ids `(2, 7)` src and `(2, 6)` tgt → output logits `(2, 6, vocab)`.

---

## Tier 3 — Train it on a toy task

### 3.1 Copy task
Train the model to **copy** a sequence of random integers (src == tgt). This is the canonical Annotated Transformer sanity task.
- Vocab = 11 (10 symbols + pad), `d_model=512` (or shrink to 128 for speed).
- Implement the **Noam LR schedule**: `lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)`, warmup=400.
- Implement **label smoothing** (KL-divergence loss against a smoothed target distribution, ε=0.1).

**Success:** after a few hundred steps, greedy-decoding a held-out sequence reproduces the input exactly.

### 3.2 Greedy decode
Implement autoregressive greedy decoding: start with a `<start>` token, repeatedly append the argmax next token, feeding the growing sequence + `subsequent_mask` back in.

**Check:** decoded copy of `[1,2,3,4,5,6,7,8,9,10]` returns the same sequence.

---

## Tier 4 — Understand by breaking (stretch)

Answer each by **experiment**, not memory:

1. **Remove the `√d_k` scaling.** Train the copy task. What happens to the loss curve and the attention weight distribution? Why?
2. **Remove positional encoding entirely.** Can the model still learn to copy? Can it learn to *reverse* the sequence? Explain the difference.
3. **Drop the subsequent mask** in the decoder. Does training loss look suspiciously good? Why is inference then broken?
4. **Set `h=1` vs `h=8`** with `d_model` fixed. Compare convergence on the copy task and inspect what each head attends to.
5. **Swap post-norm `LayerNorm(x + Sublayer(x))` for pre-norm `x + Sublayer(LayerNorm(x))`.** Which trains more stably without warmup? (This is a real modern design change — connects to paper 16, *Identity Mappings in Deep Residual Networks*.)
6. **Tie vs untie** the output projection and input embedding weights. Effect on params and final accuracy?

---

## Deliverables

- [ ] `transformer.py` — from-scratch model, all Tier 1–2 checks passing.
- [ ] `train_copy.py` — trains the copy task to ~100% greedy accuracy.
- [ ] `notes.md` — your written answers + plots for the Tier 4 ablations.

## Reference (only after you've tried)

- Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
- Original code: https://github.com/harvardnlp/annotated-transformer/
- Paper: `../../papers/14_Attention_Is_All_You_Need.pdf`
