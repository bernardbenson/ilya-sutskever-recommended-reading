# Tutorials — The Unreasonable Effectiveness of RNNs

Step-by-step walkthroughs that *teach* the exercises in `../EXERCISES.md`. Each
tutorial explains the idea, shows the code, calls out the bugs people actually
hit, and points at the verified reference solution.

> **Work order:** read `../SUMMARY.md` → attempt an exercise yourself → get stuck
> or finish → read the matching tutorial → compare with `solutions/`.
> Reading the tutorial *first* robs you of the learning. Try, then check.

## Contents

| Tutorial | Covers | Exercises |
|----------|--------|-----------|
| [01_min_char_rnn.md](01_min_char_rnn.md) | the recurrence, forward pass, backprop-through-time by hand, gradient check, sampling | Tier 1 |
| [02_pytorch_lstm.md](02_pytorch_lstm.md) | autograd RNN, embedding + `nn.LSTM`, batching, truncated BPTT | Tier 2 |
| [03_train_sample.md](03_train_sample.md) | training on a corpus, temperature sampling | Tier 3 |
| [04_ablations.md](04_ablations.md) | clipping, vanishing gradients, temperature, interpretable cells | Tier 4 |

## Reference solutions

- [`solutions/min_char_rnn.py`](solutions/min_char_rnn.py) — pure-NumPy vanilla RNN, Tier 1. Run with `--gradcheck` to verify the hand-written gradients.
- [`solutions/char_rnn.py`](solutions/char_rnn.py) — PyTorch LSTM, Tier 2–3, with temperature sampling.

Both are **verified to run** on this repo (a `uv` project), CPU-only, in seconds:

```bash
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/min_char_rnn.py --gradcheck
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/min_char_rnn.py
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/char_rnn.py
```

Expected: `min_char_rnn.py --gradcheck` prints `ok` for every parameter;
`min_char_rnn.py` drives smoothed loss from ~84 down to ~0.04 and its samples turn
from gibberish into fluent copied text; `char_rnn.py` reaches ~0.02 loss in 12
epochs and prints three samples showing the temperature trade-off.

## Conventions used throughout

- The NumPy model treats a character as a **one-hot column vector** of shape
  `(vocab_size, 1)`; the hidden state is `(hidden_size, 1)`. Keep those shapes in
  your head — most Tier 1 bugs are a transpose or a wrong axis.
- We deliberately start with a **vanilla** RNN (Tier 1) so you *feel* the
  vanishing/exploding-gradient problem, then switch to an **LSTM** (Tier 2) which
  fixes it. That contrast is the bridge to paper 04.
- The toy corpus is tiny and repetitive on purpose: it lets a small model show
  visible learning in seconds. Everything scales unchanged to a real text file.
