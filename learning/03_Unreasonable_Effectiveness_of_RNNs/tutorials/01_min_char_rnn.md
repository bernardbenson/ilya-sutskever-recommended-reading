# Tutorial 01 — Vanilla RNN in pure NumPy (Tier 1)

The whole idea of a sequence model fits in ~110 lines of NumPy. We build it with
**no autograd** — you write the backward pass by hand — because doing so is the
one thing that turns "an RNN is a loop with a hidden state" from a sentence into
knowledge. Reference: [`solutions/min_char_rnn.py`](solutions/min_char_rnn.py).

---

## 1.1 Data: characters as one-hot vectors

Language modeling needs no labels — the text *is* the supervision. Char `t`'s label
is char `t+1`. Build the vocab and index maps:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
```

A character becomes a one-hot **column** vector `x` of shape `(vocab_size, 1)`.
That column-vector convention (not row) is what makes the matrix shapes below line
up. **Check:** round-trip a string through both maps and get it back.

---

## 1.2 The forward pass

Three parameter matrices define the model: `Wxh` (input→hidden), `Whh`
(hidden→hidden, *this* is the recurrence), `Why` (hidden→output). For each step:

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h), \quad y_t = W_{hy}h_t + b_y, \quad p_t = \text{softmax}(y_t)$$

```python
for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1)); xs[t][inputs[t]] = 1     # one-hot
    hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t-1] + bh)          # new memory
    ys[t] = Why @ hs[t] + by                                   # logits
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))              # probabilities
    loss += -np.log(ps[t][targets[t], 0])                      # cross-entropy
```

Two things to internalize:
- **`hs[-1]` is `hprev`**, the hidden state carried in from the previous chunk. The
  memory doesn't reset every 25 characters; it flows across chunks.
- **We cache `xs, hs, ps`** for every `t`. The backward pass needs all of them —
  this caching *is* what a framework's autograd tape does for you automatically.

**Check.** With tiny random weights, the loss on the first chunk is about
`-log(1/vocab_size) * seq_length`. For `vocab_size=29, seq_length=25` that's ≈ 84 —
exactly what the reference prints at iter 0. A uniform-random model has no reason to
score any better.

---

## 1.3 Backprop through time (the core of Tier 1)

Unroll the loop and backprop from the last step to the first. The softmax +
cross-entropy gradient is famously clean: `dy = p_t - onehot(target)`.

```python
for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t]); dy[targets[t]] -= 1        # softmax - onehot
    dWhy += dy @ hs[t].T;  dby += dy
    dh = Why.T @ dy + dhnext                         # <-- the recurrence term
    dhraw = (1 - hs[t] ** 2) * dh                    # backprop through tanh
    dbh += dhraw
    dWxh += dhraw @ xs[t].T
    dWhh += dhraw @ hs[t-1].T
    dhnext = Whh.T @ dhraw                           # gradient to the previous step
```

**The single most important line is `dh = Why.T @ dy + dhnext`.** The hidden state
`h_t` influences the loss **twice**: directly through this step's output `y_t`, and
indirectly through *every future step* (because `h_t` fed `h_{t+1}`). `dhnext` is
that future gradient flowing back. Drop it and you no longer have an RNN — you have
`seq_length` independent classifiers that can't learn any dependency across time.

**Why `dWxh`, `dWhh`, `dWhy` are `+=` and accumulate.** The same weights are used at
every timestep (weight sharing), so their total gradient is the **sum** of the
contributions from all steps. This is the mathematical face of "shared weights
across time."

**Trap — `tanh` derivative.** `d/dx tanh = 1 - tanh²`, and `hs[t]` already *is*
`tanh(...)`, so `dhraw = (1 - hs[t]**2) * dh`. Recomputing `tanh` here is a common
slip.

---

## 1.4 Gradient check — do not skip this

A hand-written backward pass is a bug farm. Before you trust it, verify it
numerically: perturb one parameter entry by `±ε` and compare the finite-difference
slope to your analytic gradient.

```bash
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/min_char_rnn.py --gradcheck
```
```
  Wxh  max rel err 4.28e-08  (tol 1e-05)  ok
  Whh  max rel err 1.88e-06  (tol 1e-03)  ok
  Why  max rel err 6.20e-07  (tol 1e-05)  ok
  bh   max rel err 3.54e-08  (tol 1e-05)  ok
  by   max rel err 3.63e-09  (tol 1e-05)  ok
```

**Why `Whh` gets a looser tolerance.** Its gradient chains through all
`seq_length` `tanh` steps, so the *finite-difference estimate itself* loses
precision — the analytic gradient is fine. A relative error around `1e-4`–`1e-3`
for `Whh` is expected; `O(1)` for any parameter is a real bug. Fix any real failure
here before training — a wrong gradient can still *look* like it's learning and
waste hours.

---

## 1.5 Gradient clipping

Backprop through time repeatedly multiplies by `Whh`. If its largest singular value
> 1, gradients grow exponentially with sequence length — they **explode** to `inf`
/ `NaN`. The blunt, effective fix:

```python
for dparam in (dWxh, dWhh, dWhy, dbh, dby):
    np.clip(dparam, -5, 5, out=dparam)
```

**Try it:** comment the clip out and train — the loss spikes or `NaN`s. Restore it
and it's stable. (The *opposite* failure, gradients **vanishing** toward zero over
long gaps, is not fixable by clipping — that's what LSTMs solve in paper 04, and
what you'll demonstrate in Tier 4 #2.)

---

## 1.6 Sampling: let the model dream

Generation feeds the model's own output back as its next input:

```python
for _ in range(n):
    h = np.tanh(Wxh @ x + Whh @ h + bh)
    y = Why @ h + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())   # sample, not argmax
    x = np.zeros((vocab_size, 1)); x[ix] = 1                 # feed it back
```

**Trap — argmax instead of sampling.** Taking the most-likely character every step
collapses into short repeating loops ("the the the…"). *Drawing* from the
distribution is what produces varied, natural text — and is why the same trained
model can generate many different samples.

---

## 1.7 Training loop (Adagrad + truncated BPTT)

Walk the corpus in `seq_length`-char chunks, carrying `hprev` between them and
resetting at the corpus end. Update with **Adagrad** (each parameter gets its own
adaptive step size from its running sum of squared gradients):

```python
for param, dparam, m in zip(params, grads, mem):
    m += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(m + 1e-8)
```

Track a **smoothed** loss (`0.999·old + 0.001·new`) so the per-chunk noise doesn't
hide the trend.

**Run it:**
```bash
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/min_char_rnn.py
```
```
iter      0  loss 84.182
  sample: 'caotk qpsnv,tct.yqhkj.z,pyehrgrwn...'          <- random characters
iter   5000  loss 1.610
  sample: ' the quick brown fox runs. a quick fox is...'   <- words, mostly spelled right
iter  20000  loss 0.039
  sample: ' the quick brown fox jumps over the lazy dog...' <- fluent
```

Watching the sample climb the ladder — noise → letters → words → phrasing — is the
"unreasonable effectiveness" moment. Nothing told the model what a word is; the
next-character objective alone forced it to discover them.

Once your gradient check passes and loss falls below ~0.1, move to
[02_pytorch_lstm.md](02_pytorch_lstm.md) and let a framework carry the backward
pass so you can scale up.
