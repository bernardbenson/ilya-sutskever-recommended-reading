# Exercises — Build a character-level RNN

Goal: rebuild Karpathy's `min-char-rnn` **from scratch** in NumPy (forward pass,
backprop-through-time, sampling), verify your gradients numerically, then scale
up to a PyTorch LSTM and generate text. Don't copy the reference — write each
piece yourself, then diff against `solutions/`.

**Rules of engagement**
- Tier 1 uses **only `numpy`**. No autograd — you write the backward pass by hand.
- Prove every backward pass with a **numerical gradient check** before you trust it.
- After each exercise, check the referenced assertion / expected behavior.
- Keep Tier 1 in one file `min_char_rnn.py`; Tier 2–3 in `char_rnn.py`.

---

## Tier 1 — Vanilla RNN in pure NumPy

### 1.1 Data pipeline
Load a text string, build `char_to_ix` / `ix_to_char` maps, and one-hot encode.
- `vocab_size = len(set(text))`.
- A character is a column vector `x` of shape `(vocab_size, 1)`, one-hot.

**Check:** round-trip a string through `char_to_ix` → `ix_to_char` and get it back.

### 1.2 Forward pass
Implement the recurrence over a chunk of `seq_length` characters:
```
h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
y_t = Why @ h_t + by
p_t = softmax(y_t)
loss += -log(p_t[target_t])          # cross-entropy
```
Carry `h` in from the previous chunk (`hprev`), and cache `xs, hs, ps` for the
backward pass.

**Check:** at initialization (tiny random weights) the loss for the first chunk
is ≈ `-log(1/vocab_size) * seq_length` — the model is uniform-random.

### 1.3 Backprop through time (the core exercise)
Walk `t` from `seq_length-1` down to `0`, accumulating gradients:
- `dy = p_t - onehot(target_t)` (softmax+cross-entropy gradient).
- `dWhy += dy @ h_t.T`, `dby += dy`.
- `dh = Why.T @ dy + dhnext` — **the `dhnext` term is the recurrence**: gradient
  arriving from the future.
- `dhraw = (1 - h_t²) * dh` (backprop through `tanh`).
- `dWxh += dhraw @ x_t.T`, `dWhh += dhraw @ h_{t-1}.T`, `dbh += dhraw`.
- `dhnext = Whh.T @ dhraw` — pass gradient to the previous step.

**Trap — forgetting `dhnext`.** If you drop the `+ dhnext` term you've built a
per-step classifier, not an RNN: no gradient flows across time and it can't learn
dependencies longer than one step.

### 1.4 Gradient check (do not skip)
Verify 1.3 numerically: for random parameter entries, compare your analytic
gradient to `(loss(w+ε) - loss(w-ε)) / 2ε`.

**Check:** relative error `< 1e-5` for `Wxh, Why, bh, by`. `Whh` is looser
(`~1e-4`–`1e-3`) because its gradient chains through many `tanh` steps — that's
expected, not a bug. If any is `O(1)`, you have a real backward-pass bug — fix it
before training.

### 1.5 Gradient clipping
Clip every gradient elementwise to `[-5, 5]` before the update.

**Check:** remove the clip and train — watch the loss `NaN` out or spike. Restore
it and it's stable. (This is Tier 4 #1, previewed.)

### 1.6 Sampling
Given a hidden state and a seed character, generate `n` characters: run the
recurrence one step, softmax, **draw** an index with `np.random.choice(p=...)`
(not argmax), feed it back in.

**Check:** before training, samples are gibberish. After training on the toy
corpus they reproduce recognizable phrases from it.

### 1.7 Training loop
Walk the corpus in `seq_length` chunks, carrying `hprev` between them and resetting
at the corpus end. Update with **Adagrad** (per-parameter running sum of squared
grads). Track a smoothed loss; print a sample every few thousand iterations.

**Success:** on the toy corpus, smoothed loss falls from ~`84` toward `< 0.1`, and
printed samples go from random characters to fluent copied text.

---

## Tier 2 — Rebuild in PyTorch, then swap in an LSTM

### 2.1 Autograd version of the vanilla RNN
Re-express 1.2 as an `nn.Module` and let autograd do 1.3. Confirm you get the same
learning behavior with **none** of the hand-written backward pass.

**Check:** loss curve matches the NumPy version's shape.

### 2.2 Embedding + `nn.LSTM` + linear head
Replace the one-hot input with an `nn.Embedding`, the vanilla cell with `nn.LSTM`
(1–2 layers), and read out with a `Linear(hidden → vocab)`.

**Check:** feed `(batch, seq)` indices → logits `(batch, seq, vocab)`.

### 2.3 Mini-batching + truncated BPTT
Cut the character stream into `batch_size` parallel rows; iterate in windows of
`seq_len`. **Carry `(h, c)` across windows but `.detach()` it** so gradients don't
backprop past the window (truncated BPTT).

**Trap — not detaching the state.** Without `detach`, autograd tries to backprop
through the entire history each step: memory grows unboundedly and you eventually
get a "backward through the graph a second time" error.

**Check:** training runs at constant memory and loss decreases each epoch.

---

## Tier 3 — Train and generate

### 3.1 Train on a real corpus
Point it at a bigger text file (Shakespeare, a source-code file, your own writing).
Clip grad norm to 5, use Adam. Train until loss plateaus.

**Success:** sampled text captures the *style* of the corpus — line structure,
punctuation, vocabulary — even if it's semantically nonsense.

### 3.2 Temperature sampling
Divide logits by a temperature `T` before softmax. Generate samples at `T ∈
{0.5, 1.0, 1.5}`.

**Check:** `T=0.5` is repetitive and "safe"; `T=1.5` is diverse and starts making
spelling mistakes. You should be able to *see* the coherence/diversity trade-off.

---

## Tier 4 — Understand by breaking (stretch)

Answer each by **experiment**, not memory:

1. **Remove gradient clipping.** Train the NumPy RNN. When/why does the loss blow
   up? What does this tell you about repeated multiplication by `Whh`?
2. **Vanishing gradients / long dependencies.** Build a toy task where the target
   depends on a character ~50 steps back (e.g. match an opening bracket). Compare
   the **vanilla RNN** vs the **LSTM** on it. Which one learns the dependency? This
   is the entire motivation for paper 04.
3. **Temperature sweep.** Plot sample coherence vs `T` from 0.2 to 2.0. Explain the
   two failure modes at the extremes.
4. **Hidden size.** Train with `hidden_size ∈ {16, 64, 256}`. How does capacity
   change what the model can memorize vs generalize?
5. **Truncated BPTT window.** Vary `seq_len ∈ {5, 25, 100}`. What's the trade-off
   between the length of dependency the model can learn and compute/memory?
6. **Interpretable cells (the payoff).** After training on structured text (code or
   text with quotes/brackets), record one hidden unit's activation over a sample
   and color the characters by it. Can you find a neuron that tracks quote-state or
   bracket depth, as in the blog post?

---

## Deliverables

- [ ] `min_char_rnn.py` — from-scratch NumPy RNN, gradient check passing, trains on
      the toy corpus to `< 0.1` loss.
- [ ] `char_rnn.py` — PyTorch LSTM, trains on a real corpus, temperature sampling.
- [ ] `notes.md` — your written answers + plots for the Tier 4 experiments,
      including at least one interpretable-neuron visualization.

## Reference (only after you've tried)

- Blog post: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- `min-char-rnn.py` gist: https://gist.github.com/karpathy/d4dee566867f8291f086
- `char-rnn` (Torch): https://github.com/karpathy/char-rnn
- Next paper: `../../papers/04_Understanding_LSTM_Networks.pdf` (the fix for #2 above)
