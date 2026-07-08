# Tutorial 01 — Core building blocks (Tier 1)

Everything in a Transformer is built from five small pieces. Get these right and
the rest is plumbing. We build each one, prove it with a shape check, and note
the traps.

---

## 1.1 Scaled dot-product attention

**The idea.** Every position emits a *query* ("what am I looking for?"), and every
position exposes a *key* ("what do I offer?") and a *value* ("what I'll hand over
if you pick me"). We score each query against every key with a dot product, turn
the scores into weights with softmax, and take the weighted sum of values.

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```python
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn
```

**Why `/ √d_k`?** For random Q, K with unit-variance entries, each dot product
sums `d_k` products, so its variance grows like `d_k`. Large scores push softmax
into a saturated region where gradients vanish. Dividing by `√d_k` renormalizes
the variance back to ~1. (Tier 4 #1 makes you feel this by deleting it.)

**Trap — mask before softmax.** You must add `-1e9` to masked scores *before*
softmax so those positions get ~0 weight. Masking after softmax leaves nonzero
leakage and the weights no longer sum to 1.

**Trap — which dim is softmax?** `dim=-1` (over keys). Each query's weights across
all keys must sum to 1. `softmax(dim=-2)` is a classic silent bug.

**Check.** `Q=K=V` of shape `(2,8,5,64)`, no mask → output `(2,8,5,64)` and every
attention row sums to ~1.

---

## 1.2 Multi-head attention

**The idea.** One attention is one "view." Multi-head runs `h=8` attentions in
parallel on `d_k = d_model/h = 64`-dim projections, so different heads can
specialize (one tracks syntax, another tracks a far-back token, …). Concatenate
the heads and project back to `d_model`.

Four linear layers: `W_q, W_k, W_v` project inputs; `W_o` projects the concat.

```python
query, key, value = [
    lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    for lin, x in zip(self.linears, (query, key, value))
]
x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
return self.linears[-1](x)
```

**The reshape dance.** `(batch, seq, d_model)` → view as `(batch, seq, h, d_k)` →
`transpose(1,2)` → `(batch, h, seq, d_k)` so attention treats `h` like an extra
batch dim. After attention, transpose back and *merge* heads.

**Trap — `.contiguous()`.** After `transpose`, memory is non-contiguous and
`.view()` throws. Call `.contiguous()` first (or use `.reshape()`).

**Trap — mask broadcasting.** The mask is `(batch, 1, seq)` or `(batch, seq, seq)`;
you need it to broadcast over the head dim, so `mask.unsqueeze(1)` →
`(batch, 1, 1 or seq, seq)`. The solution does this once at the top of `forward`.

**Check.** `(2,5,512)` in → `(2,5,512)` out. Params = `4·512·512 + 4·512 = 1,050,624`.

---

## 1.3 Position-wise feed-forward

Two linears with a ReLU between, applied independently at every position:

$$\text{FFN}(x) = \max(0,\, xW_1 + b_1)W_2 + b_2, \quad d_{ff}=2048$$

```python
def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

"Position-wise" = the *same* MLP runs on each token vector separately (it's just a
`Linear` over the last dim). This is where most of the model's parameters and its
per-token nonlinear "thinking" live. **Check:** `(2,5,512)` → `(2,5,512)`.

---

## 1.4 Positional encoding

Attention is **permutation-invariant** — shuffle the input tokens and you shuffle
the outputs identically; it has no idea what order they came in. We inject order
by *adding* a fixed signal that depends on position:

$$PE_{(pos,2i)} = \sin\!\big(pos/10000^{2i/d}\big), \quad PE_{(pos,2i+1)} = \cos(\cdots)$$

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
self.register_buffer("pe", pe.unsqueeze(0))   # buffer => no gradient
```

**Why sinusoids?** Each dimension is a sinusoid of a different wavelength
(geometric from 2π to ~10000·2π). Any relative offset `k` is a fixed linear
function of the encoding at `pos`, so the model can learn to attend "3 tokens
back" regardless of absolute position. And it extrapolates to longer sequences
than seen in training.

**Trap — parameter vs buffer.** Use `register_buffer`, not `nn.Parameter`. These
encodings are *fixed*; making them learnable is a different design and here would
just waste gradients. Verify no grad flows to `pe`.

**Embeddings scaling.** Token embeddings are multiplied by `√d_model` so their
magnitude is comparable to the positional signal you're adding.

---

## 1.5 LayerNorm + residual ("Add & Norm")

Write your **own** LayerNorm — normalize over the feature dim, then scale/shift
with learned `γ, β`:

```python
mean = x.mean(-1, keepdim=True)
std = x.std(-1, keepdim=True, unbiased=False)
return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

Then wrap every sub-layer in a residual connection:

```python
# pre-norm variant (what we use)
def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
```

**Why residuals?** They give gradients a direct path around each sub-layer, which
is what makes a 6- (or 96-) layer stack trainable at all — the same insight as
ResNet (papers 11 & 16 in this list).

**Post-norm vs pre-norm.**
- Paper (post-norm): `LayerNorm(x + Sublayer(x))`.
- Modern (pre-norm): `x + Sublayer(LayerNorm(x))`.

Pre-norm keeps a clean residual highway and trains stably even without LR warmup;
post-norm often needs warmup to not diverge early. We use pre-norm; Tier 4 #5 has
you compare them directly.

**Check.** LayerNorm output has ~0 mean and ~1 std along the last dim.

---

## Run the checks

```bash
uv run learning/01_The_Annotated_Transformer/tutorials/solutions/transformer.py
```

You should see `1.1 … ok` through `1.5 … ok`. Once all five pass, move to
[02_assemble_stack.md](02_assemble_stack.md).
