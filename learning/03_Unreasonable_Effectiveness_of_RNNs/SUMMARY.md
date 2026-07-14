# 03 — The Unreasonable Effectiveness of Recurrent Neural Networks

**Author:** Andrej Karpathy
**Source:** [Blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) · [char-rnn (Torch)](https://github.com/karpathy/char-rnn) · [min-char-rnn.py gist](https://gist.github.com/karpathy/d4dee566867f8291f086)

---

## What it is

Not a paper but a **blog post** — and one of the most influential pieces of writing in modern deep learning. Karpathy trains a character-level RNN on raw text (Shakespeare, the Linux kernel, Wikipedia, LaTeX, baby names) and shows it generating startlingly plausible output one character at a time. The post's job is **intuition**: *why* do sequence models work at all, and why is a tiny recurrent network "unreasonably" good?

It is paper #2 in this reading order on purpose. Before you meet LSTMs (04), attention (15), or the Transformer (14), you should feel — in your hands — the core idea every one of them builds on: **a model that carries a hidden state through time and predicts the next token.**

The goal for you is **not** to read it, but to *rebuild* `min-char-rnn.py` from scratch — forward pass, backprop-through-time by hand, sampling — until the recurrence loop fits in your head.

---

## The big idea

A feed-forward net maps one fixed-size input to one fixed-size output. But language, audio, and code are **sequences** of varying length where order is everything. An RNN handles this by adding a loop:

> At each timestep, combine the current input with a **hidden state** (a summary of everything seen so far), produce an output, and pass an updated hidden state to the next step.

$$h_t = \tanh(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h), \qquad y_t = W_{hy}\,h_t + b_y$$

That single recurrence — the same weights applied at every step — lets one small network process a sequence of *any* length. The hidden state `h_t` is the model's memory; learning is learning **what to remember**.

---

## The task: character-level language modeling

Feed the network text one character at a time; ask it to predict the **next** character. Train by cross-entropy on the true next char. That's it. From this trivial objective the model is forced to learn spelling, then words, then grammar, then structure (balanced brackets, valid syntax, Markdown headers) — purely to lower its next-character surprise.

```
input:   h  e  l  l  o
target:  e  l  l  o  <space>
              ▲
     at each step, predict the next character
```

Then you **sample**: pick a seed character, get the model's next-char distribution, draw from it, feed that character back in, and repeat. The network hallucinates text in its own trained style.

---

## Architecture at a glance

```
   x₁        x₂        x₃            (one-hot chars in)
    │         │         │
 ┌──┴──┐   ┌──┴──┐   ┌──┴──┐
 │ RNN │──▶│ RNN │──▶│ RNN │──▶ ...  (same weights reused each step;
 └──┬──┘ h₁└──┬──┘ h₂└──┬──┘         the arrow carrying hₜ is the "memory")
    │         │         │
   y₁        y₂        y₃            (next-char logits → softmax)
```

Unrolled, an RNN is just a very deep feed-forward net that **shares weights across depth** — and depth here equals sequence length.

## The components you must be able to implement

1. **The recurrence** — `h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)`. One hidden layer, reused at every timestep. `Whh` is what makes it recurrent.

2. **The readout + softmax** — `y_t = Why·h_t + by`, then `softmax(y_t)` gives a probability over the vocabulary; cross-entropy against the true next char is the loss.

3. **Backprop through time (BPTT)** — unroll the loop, then backprop. The gradient at step `t` flows backward into *every* earlier step through `Whh`. Writing this by hand (Tier 1) is the whole point.

4. **Gradient clipping** — repeatedly multiplying by `Whh` makes gradients **explode**; clip them to a fixed norm. (The opposite failure, **vanishing** gradients, is what LSTMs in paper 04 fix.)

5. **Sampling with temperature** — divide logits by `T` before softmax. Low `T` → sharp, safe, repetitive; high `T` → flat, diverse, more mistakes.

## Why it works (the "unreasonable" part)

- **Weight sharing across time** means the model learns *rules* ("close a quote you opened"), not position-specific facts — so it generalizes across a sequence and to new lengths.
- **The hidden state is a learned, lossy compression** of the relevant past. The network isn't told what to remember; the next-char objective forces it to discover that brackets, quotes, and indentation matter.
- **Interpretable cells emerge for free.** Karpathy visualizes individual neurons that learn, unsupervised, to track quote-open/close, line length, or bracket nesting depth — a striking hint that these models build structured internal representations. (Foreshadows the "features" story that recurs through this whole list.)

## Key takeaways

- Sequence modeling = **carry a state forward, predict the next token, repeat.** GPT does exactly this; it just swaps the recurrent cell for attention.
- Almost everything expensive is optimization plumbing (clipping, truncated BPTT, Adagrad/Adam). The modeling idea is tiny — a loop with shared weights.
- Vanilla RNNs *can* learn short-range structure but struggle to hold information across long gaps because of vanishing gradients. That single limitation is the entire motivation for the **next** paper: LSTMs (04).

## Where this sits in the reading order

- **← Paper 27 (CS231n)** gave you backprop and softmax on feed-forward nets; an unrolled RNN is those same tools with weight sharing across time.
- **→ Paper 04 (Understanding LSTM Networks)** fixes the vanishing-gradient weakness you'll *feel* in Tier 4 here.
- **→ Papers 15 & 14 (attention, the Transformer)** keep the "predict the next token" objective but replace the sequential recurrence with a parallel attention mechanism — read them and you'll recognize the same skeleton.

---

*Next: work through `EXERCISES.md` and build `min-char-rnn` yourself.*
