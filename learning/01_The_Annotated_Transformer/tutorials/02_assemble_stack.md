# Tutorial 02 — Assemble the stack (Tier 2)

You have the five blocks. Now wire them into an encoder, a decoder, the masks
that make the decoder autoregressive, and the full encoder–decoder model.

---

## A helper: `clones`

Layers are identical in structure but must have **independent weights**. Deep-copy
a prototype `N` times:

```python
def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
```

**Trap.** Don't do `[layer] * n` — that's the *same* object `n` times, so all
"layers" share one set of weights. Use `deepcopy`.

---

## 2.1 Encoder layer & Encoder

One encoder layer = self-attention sub-layer, then feed-forward sub-layer, each in
an Add&Norm residual:

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

Note **self**-attention: query = key = value = `x`. Every position attends to
every other position in the source (no masking except padding). The `Encoder`
stacks `N` of these and applies a final `LayerNorm` (needed because we use
pre-norm — the last sub-layer's output hasn't been normalized yet).

---

## 2.2 Decoder layer & Decoder

One decoder layer has **three** sub-layers:

```python
class DecoderLayer(nn.Module):
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   # masked self-attn
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))    # cross-attn
        return self.sublayer[2](x, self.feed_forward)
```

1. **Masked self-attention** over the target so far (`tgt_mask` blocks the future).
2. **Cross-attention**: queries come from the decoder (`x`), but keys/values come
   from the encoder output `memory`. This is how the decoder *reads* the source.
3. **Feed-forward.**

**Trap — the Q/K/V wiring of cross-attention.** It's `src_attn(x, m, m, ...)`:
query `x` (decoder), key `m`, value `m` (encoder memory). Swapping these is a
common and quiet bug — shapes still work, the model just can't translate.

---

## 2.3 Masks

Two jobs:

- **Pad mask** — hide padding tokens so attention ignores them:
  `(seq != pad).unsqueeze(-2)` → `(batch, 1, seq)`.
- **Subsequent mask** — the causal, lower-triangular mask so position `i` can only
  see `j ≤ i`:

```python
def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0        # True where attention is ALLOWED
```

`triu(…, diagonal=1)` puts 1s strictly above the diagonal (the future); `== 0`
flips it so `True` marks allowed positions. Decoder self-attention **ANDs** the pad
mask and the subsequent mask together (see `Batch.make_std_mask` in tutorial 03).

**This is the single most important line in the whole model.** Without it, at
training time the decoder can see the token it's supposed to predict, training
loss looks great, and inference is garbage. (Tier 4 #3.)

**Check.** `subsequent_mask(4)` is lower-triangular; assert `i` attends only to
`j ≤ i`.

---

## 2.4 Full model

Glue it together. Source/target each get `Embeddings → PositionalEncoding`. The
`Generator` is the final `Linear(d_model → vocab)` + `log_softmax`.

```python
class EncoderDecoder(nn.Module):
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
```

`make_model(...)` assembles the pieces with `deepcopy` (so encoder and decoder get
separate attention/FFN instances) and applies **Xavier uniform init** to every
weight matrix with `dim > 1`. That init matters: with default init the model
trains much slower or stalls.

**Separate `encode`/`decode`.** They're split (rather than only `forward`) because
inference needs to encode the source *once* and then call `decode` repeatedly as
it generates tokens (tutorial 03, greedy decode).

**Check.** `src (2,7)`, `tgt (2,6)` → `model.generator(out)` is `(2, 6, vocab)`.

---

## Run the checks

```bash
uv run learning/01_The_Annotated_Transformer/tutorials/solutions/transformer.py
# => 2.3 subsequent mask ok / 2.4 full model ok — logits (2, 6, 11)
```

Model builds and produces correctly-shaped logits → you have a Transformer. Now
make it *learn something*: [03_train_copy.md](03_train_copy.md).
