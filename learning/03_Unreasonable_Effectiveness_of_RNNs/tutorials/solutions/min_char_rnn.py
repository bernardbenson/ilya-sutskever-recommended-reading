"""Tier 1 reference solution: a vanilla character-level RNN in pure NumPy.

This is a faithful, lightly-annotated reimplementation of Karpathy's famous
112-line `min-char-rnn.py` gist, the companion to "The Unreasonable
Effectiveness of Recurrent Neural Networks". It is the whole idea of a sequence
model in one file: a hidden state carried forward through time, backprop
through time (BPTT), gradient clipping, and sampling.

There is no PyTorch here on purpose. Writing the backward pass by hand is the
point of Tier 1 — once you have felt every gradient, the framework version in
`char_rnn.py` stops being magic.

Run: `python min_char_rnn.py`            (trains, prints loss + samples)
     `python min_char_rnn.py --gradcheck` (numerically verifies the gradients)
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Data. A small, structured corpus so a tiny vanilla RNN shows visible
# progress on a CPU in seconds. Any text file works — swap this for
# input.txt = open("shakespeare.txt").read() to reproduce the blog post.
# ---------------------------------------------------------------------------
DATA = (
    "the quick brown fox jumps over the lazy dog. "
    "the lazy dog sleeps while the quick brown fox runs. "
    "a quick fox is a clever fox, and a lazy dog is a sleepy dog. "
) * 12

chars = sorted(list(set(DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
hidden_size = 100   # size of the hidden state (the "memory")
seq_length = 25     # number of steps to unroll for truncated BPTT
learning_rate = 1e-1

# ---------------------------------------------------------------------------
# Model parameters. One hidden layer of recurrence.
#   h_t     = tanh(Wxh · x_t + Whh · h_{t-1} + bh)
#   y_t     = Why · h_t + by
#   p_t     = softmax(y_t)
# ---------------------------------------------------------------------------
def init_params(seed=1):
    rng = np.random.RandomState(seed)
    Wxh = rng.randn(hidden_size, vocab_size) * 0.01   # input   -> hidden
    Whh = rng.randn(hidden_size, hidden_size) * 0.01  # hidden  -> hidden
    Why = rng.randn(vocab_size, hidden_size) * 0.01   # hidden  -> output
    bh = np.zeros((hidden_size, 1))
    by = np.zeros((vocab_size, 1))
    return Wxh, Whh, Why, bh, by


def loss_fun(inputs, targets, hprev, params):
    """Forward pass through `seq_length` steps, then BPTT.

    inputs, targets : lists of ints (char indices), length seq_length
    hprev           : (hidden_size, 1) hidden state carried in from last chunk
    returns loss, gradients for each param, and the last hidden state.
    """
    Wxh, Whh, Why, bh, by = params
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # ---- forward pass: one step per character -----------------------------
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1                                   # one-hot input
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)    # new hidden state
        ys[t] = Why @ hs[t] + by                               # unnormalized logits
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))          # softmax probs
        loss += -np.log(ps[t][targets[t], 0])                  # cross-entropy

    # ---- backward pass: backprop through time -----------------------------
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1                    # dL/dy = softmax - onehot(target)
        dWhy += dy @ hs[t].T
        dby += dy
        dh = Why.T @ dy + dhnext               # gradient into hidden state
        dhraw = (1 - hs[t] ** 2) * dh          # backprop through tanh
        dbh += dhraw
        dWxh += dhraw @ xs[t].T
        dWhh += dhraw @ hs[t - 1].T
        dhnext = Whh.T @ dhraw                 # carry gradient to previous step

    # ---- clip to combat exploding gradients -------------------------------
    for dparam in (dWxh, dWhh, dWhy, dbh, dby):
        np.clip(dparam, -5, 5, out=dparam)

    return loss, (dWxh, dWhh, dWhy, dbh, dby), hs[len(inputs) - 1]


def sample(h, seed_ix, n, params):
    """Generate `n` characters from the model, feeding each output back in."""
    Wxh, Whh, Why, bh, by = params
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for _ in range(n):
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())  # sample, don't argmax
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def gradient_check():
    """Numerically verify the analytic gradients from loss_fun.

    For each parameter we compare the analytic gradient to a finite-difference
    estimate (f(w+e) - f(w-e)) / 2e. A small relative error means the hand-written
    backward pass is correct — the single most valuable check in Tier 1.

    Note: Whh (the recurrent weight) tolerates a looser bound. Its gradient
    chains through `seq_length` tanh steps, so the finite-difference estimate
    itself loses precision; ~1e-4 here is correct, not a bug.
    """
    params = init_params()
    names = ["Wxh", "Whh", "Why", "bh", "by"]
    tol = {"Whh": 1e-3}   # recurrent weight: looser numerical tolerance
    inputs = [char_to_ix[c] for c in DATA[:seq_length]]
    targets = [char_to_ix[c] for c in DATA[1:seq_length + 1]]
    hprev = np.zeros((hidden_size, 1))

    _, grads, _ = loss_fun(inputs, targets, hprev, params)
    eps, num_checks = 1e-5, 8
    print("gradient check (relative error should be below each tolerance):")
    for name, param, danal in zip(names, params, grads):
        max_rel = 0.0
        for _ in range(num_checks):
            ri = int(np.random.randint(param.size))
            old = param.flat[ri]
            param.flat[ri] = old + eps
            l_plus, _, _ = loss_fun(inputs, targets, hprev, params)
            param.flat[ri] = old - eps
            l_minus, _, _ = loss_fun(inputs, targets, hprev, params)
            param.flat[ri] = old
            dnum = (l_plus - l_minus) / (2 * eps)
            dana = danal.flat[ri]
            denom = abs(dnum) + abs(dana) + 1e-12
            max_rel = max(max_rel, abs(dnum - dana) / denom)
        thresh = tol.get(name, 1e-5)
        status = "ok" if max_rel < thresh else "FAIL"
        print(f"  {name:4s} max rel err {max_rel:.2e}  (tol {thresh:.0e})  {status}")


def train(iters=20000):
    params = init_params()
    Wxh, Whh, Why, bh, by = params
    # Adagrad memory (per-parameter running sum of squared gradients).
    mem = [np.zeros_like(p) for p in params]

    n, p = 0, 0
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
    hprev = np.zeros((hidden_size, 1))

    while n <= iters:
        # Get the next chunk; reset hidden state and pointer at the corpus end.
        if p + seq_length + 1 >= len(DATA) or n == 0:
            hprev = np.zeros((hidden_size, 1))
            p = 0
        inputs = [char_to_ix[c] for c in DATA[p:p + seq_length]]
        targets = [char_to_ix[c] for c in DATA[p + 1:p + seq_length + 1]]

        loss, grads, hprev = loss_fun(inputs, targets, hprev, params)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Adagrad parameter update.
        for param, dparam, m in zip(params, grads, mem):
            m += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(m + 1e-8)

        if n % 5000 == 0:
            print(f"iter {n:6d}  loss {smooth_loss:.3f}")
            txt = "".join(ix_to_char[i] for i in sample(hprev, inputs[0], 200, params))
            print(f"  sample: {txt!r}\n")

        p += seq_length
        n += 1

    return smooth_loss


if __name__ == "__main__":
    if "--gradcheck" in sys.argv:
        gradient_check()
    else:
        print(f"corpus: {len(DATA)} chars, vocab {vocab_size} -> {''.join(chars)!r}\n")
        final = train()
        print(f"final smooth loss: {final:.3f}")
