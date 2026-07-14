# Tutorial 04 — Understand by breaking (Tier 4)

These are the exercises where the learning lands. Each changes **one thing**,
predicts the outcome, then runs it. Don't read ahead to the answer — form a
hypothesis first, then check it against "What you should see."

**How to run an ablation.** Copy the relevant solution into a scratch dir, make the
one change, rerun, and compare against the baseline (NumPy: loss ~84 → ~0.04;
PyTorch LSTM: loss → ~0.02 with clean `T=0.5` samples).

---

## 1. Remove gradient clipping

In `min_char_rnn.py`, delete the `np.clip(dparam, -5, 5)` loop and train.

**Hypothesis to form:** BPTT multiplies gradients by `Whh` once per step. What
happens to a product of `seq_length` such factors if `Whh`'s largest singular value
is > 1?

**What you should see.** The smoothed loss spikes and often goes to `inf`/`NaN`
within a few hundred iterations. A single large gradient blows a parameter out to a
huge value, softmax saturates, and training never recovers. Restoring the clip fixes
it instantly. **Lesson:** clipping doesn't make gradients *correct*, it just caps
their magnitude so one bad step can't destroy the model — the standard price of
training any recurrent net.

---

## 2. Vanishing gradients: vanilla RNN vs LSTM on a long dependency

This is the most important experiment in the tutorial — it *is* the motivation for
paper 04. Build a toy task where the answer depends on a character far in the past.
Simplest version: sequences of the form `A ....... a` and `B ....... b`, where the
final character must match the *first* one, with ~50 random filler characters
between. The model must carry the identity of char 0 across 50 steps.

Train (a) the vanilla NumPy RNN and (b) the PyTorch LSTM on it.

**What you should see.** The **vanilla RNN** learns the local filler statistics but
predicts the final character at ≈ chance — the gradient linking step 50 back to step
0 has been multiplied through 50 `tanh` derivatives (each `< 1`) and **vanished** to
nothing, so the connection is never learned. The **LSTM** learns it: its cell state
and forget gate provide a near-additive path that carries gradient across the gap
undiminished.

**Lesson.** Clipping (ablation 1) fixes *exploding* gradients; it does nothing for
*vanishing* ones. Vanishing gradients are precisely why the field moved from vanilla
RNNs to LSTMs — read paper 04 next and this experiment will make its diagrams
obvious.

---

## 3. Temperature sweep

Generate from the trained LSTM at `T ∈ {0.2, 0.5, 1.0, 1.5, 2.0}`.

**What you should see.** At `T=0.2` the text is near-deterministic and collapses into
short repeating loops. Around `T=0.5–1.0` it's coherent and varied. By `T=1.5–2.0`
spelling breaks down and words dissolve. Two failure modes, one at each extreme:
too-low `T` = mode collapse (no diversity), too-high `T` = incoherence (too much).
The interesting behavior lives in the middle. **Lesson:** sampling temperature is a
coherence/diversity dial, independent of how well the model was trained.

---

## 4. Hidden size

Train `min_char_rnn.py` with `hidden_size ∈ {16, 64, 256}` (or the LSTM's `hidden`).

**What you should see.** Small hidden state → the model can't store enough context;
it learns bigrams but garbles longer structure and plateaus at higher loss. Larger →
lower loss and longer coherent spans, up to the point where it simply memorizes the
tiny toy corpus. **Lesson:** the hidden state *is* the memory bandwidth — its size
caps how much of the past the model can compress and carry forward.

---

## 5. Truncated BPTT window

In `char_rnn.py`, vary `seq_len ∈ {5, 25, 100}`.

**What you should see.** With `seq_len=5`, gradients only connect characters ≤ 5
apart, so the model can't learn dependencies longer than that window even though the
forward state still flows. With `seq_len=100`, it can learn longer structure but each
step costs more memory and compute, and very long windows reintroduce
vanishing-gradient trouble in the vanilla cell. **Lesson:** the BPTT window is a
direct knob on the *length of dependency the model can learn*, traded against
compute — a tension attention (paper 14) later sidesteps by connecting all positions
directly.

---

## 6. Interpretable cells (the payoff)

Karpathy's most memorable figures show *individual neurons* that, trained only to
predict the next character, spontaneously specialize: one fires inside quotes,
another tracks line length, another counts bracket depth.

Reproduce it: train on structured text (source code, or text full of `"quotes"` and
`(brackets)`). Then run a sample and record **one** hidden unit's activation at every
character:

```python
# after model(x, state), pull one unit from the LSTM hidden output for each step
# color each character by tanh(activation): red = negative, blue = positive
```

Print the generated characters colored by that activation (a terminal 256-color
escape, or an HTML span per char).

**What you should see.** With a little hunting across units you can find a neuron
whose activation flips when a quote opens and flips back when it closes, or one that
ramps up across a line and resets at the newline. Nothing supervised these features —
the next-character objective alone made them worth learning.

**Lesson.** The hidden state isn't an opaque blob; it's a structured, partly
*interpretable* compression of the input's relevant history. That "models learn
meaningful internal features" thread runs through the rest of this reading list.

---

## Deliverable: `notes.md`

For each ablation record: (a) your hypothesis, (b) the loss curve / sample result,
(c) a plot or colored-text figure where relevant, (d) one sentence on *why*. The
vanilla-vs-LSTM long-dependency result (#2) and at least one interpretable-neuron
figure (#6) are the two that most prove you understood the post.

---

## Where this goes next

You've now felt both what an RNN *can* do (learn sequence structure from a trivial
objective) and what it *can't* (hold information across long gaps, in the vanilla
form). That gap is the setup for:

- **Paper 04** — *Understanding LSTM Networks* (Olah): the gating mechanism that
  fixes ablation #2's vanishing gradient. Read it right after this.
- **Paper 05** — *Recurrent Neural Network Regularization*: dropout done right in
  RNNs; a short, practical follow-up.
- **Papers 15 & 14** — attention and the Transformer: same "predict the next token"
  objective, but the sequential recurrence you built here is replaced by a parallel
  mechanism that connects all positions directly (killing ablation #5's window
  trade-off). You'll recognize the skeleton immediately.
