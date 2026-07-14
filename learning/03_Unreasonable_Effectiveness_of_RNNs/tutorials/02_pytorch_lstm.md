# Tutorial 02 — Rebuild in PyTorch, swap in an LSTM (Tier 2)

You wrote the backward pass by hand in Tier 1. Now hand that job to autograd, swap
the fragile vanilla cell for an **LSTM**, and add the two things that let you train
on real corpora: **mini-batching** and **truncated BPTT**. Reference:
[`solutions/char_rnn.py`](solutions/char_rnn.py).

---

## 2.1 From hand-rolled to autograd

Everything you did in `min_char_rnn.py` — cache activations, chain gradients
backward through `tanh`, accumulate shared-weight gradients — is exactly what
`loss.backward()` does automatically. The value of Tier 1 is that you now *know*
what that call is doing. Here we never write a backward pass again.

---

## 2.2 Embedding → LSTM → linear head

Three upgrades over the NumPy model:

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed=32, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)               # learned, not one-hot
        self.lstm  = nn.LSTM(embed, hidden, num_layers=layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.fc    = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        emb = self.embed(x)                 # (batch, seq, embed)
        out, state = self.lstm(emb, state)  # (batch, seq, hidden)
        return self.fc(out), state          # logits (batch, seq, vocab)
```

- **`nn.Embedding` replaces one-hot.** Instead of a fixed `(vocab,1)` indicator, each
  character gets a *learned* dense vector. It's just a lookup table trained by
  backprop — strictly more flexible than one-hot.
- **`nn.LSTM` replaces `tanh(Wxh·x + Whh·h)`.** The LSTM adds a **cell state** and
  input/forget/output **gates** that give gradients a protected highway across time,
  which is why it holds long-range dependencies the vanilla RNN drops. (The *why* is
  paper 04 — here you just use it and feel the difference in Tier 4.)
- **State is a tuple `(h, c)`** — hidden *and* cell — each shaped
  `(num_layers, batch, hidden)`, versus the vanilla RNN's single `h`.

**Check.** `(batch, seq)` indices in → logits `(batch, seq, vocab)` out.

---

## 2.3 Mini-batching + truncated BPTT

To use hardware efficiently and to bound the backprop depth, cut the character
stream into `batch_size` parallel rows and walk it in `seq_len` windows:

```python
def make_batches(data, batch_size, seq_len):
    n = (data.size(0) - 1) // (batch_size * seq_len) * (batch_size * seq_len)
    x = data[:n].view(batch_size, -1)        # (batch, stream_len)
    y = data[1:n+1].view(batch_size, -1)     # targets = inputs shifted by one
    for i in range(0, x.size(1), seq_len):
        xb, yb = x[:, i:i+seq_len], y[:, i:i+seq_len]
        if xb.size(1) == seq_len:
            yield xb, yb
```

We carry `(h, c)` from one window to the next so the model still sees long context —
but we **detach** it each window so autograd doesn't backprop through all of history:

```python
state = detach_state(state)          # tuple(s.detach() for s in state)
logits, state = model(xb, state)
loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # same idea as the NumPy clip
```

**This is truncated BPTT:** the forward state flows arbitrarily far, but gradients
only flow back `seq_len` steps. It's the standard compromise between learning long
dependencies and keeping compute/memory bounded.

**Trap — forgetting to detach the state.** Without `detach`, each window's graph
stays connected to every previous window. Memory grows without bound and you hit
`RuntimeError: Trying to backward through the graph a second time`. If you see that
error, a stale state is being carried into `backward()` — detach it.

**Trap — the `y` shift.** Targets are inputs shifted one position (`data[1:n+1]`).
Off-by-one here means you're predicting the *current* character instead of the next,
and the loss won't fall below the unigram entropy.

**Check.** Training runs at constant memory and the per-epoch loss drops.

---

## Run it

```bash
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/char_rnn.py
```
```
corpus 9480 chars, vocab 29, params 219,709

epoch  0  loss 3.061
epoch  4  loss 0.201
epoch 11  loss 0.016
```

Loss falling steadily and a six-figure parameter count that trains in under a minute
on CPU means the batching and truncated-BPTT plumbing is correct. Now make it
*generate* — and control that generation with temperature:
[03_train_sample.md](03_train_sample.md).
