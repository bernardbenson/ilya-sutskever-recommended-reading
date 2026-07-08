# Tutorials — The Annotated Transformer

Step-by-step walkthroughs that *teach* the exercises in `../EXERCISES.md`. Each
tutorial explains the idea, shows the code, calls out the bugs people actually
hit, and points at the verified reference solution.

> **Work order:** read `../SUMMARY.md` → attempt an exercise yourself → get stuck
> or finish → read the matching tutorial → compare with `solutions/`.
> Reading the tutorial *first* robs you of the learning. Try, then check.

## Contents

| Tutorial | Covers | Exercises |
|----------|--------|-----------|
| [01_core_blocks.md](01_core_blocks.md)     | attention, multi-head, FFN, positional encoding, LayerNorm | Tier 1 |
| [02_assemble_stack.md](02_assemble_stack.md) | encoder/decoder layers, masks, full model | Tier 2 |
| [03_train_copy.md](03_train_copy.md)         | batching, Noam schedule, label smoothing, greedy decode | Tier 3 |
| [04_ablations.md](04_ablations.md)           | understand-by-breaking experiments | Tier 4 |

## Reference solutions

- [`solutions/transformer.py`](solutions/transformer.py) — full model, Tier 1–2. Run it to execute all shape checks.
- [`solutions/train_copy.py`](solutions/train_copy.py) — copy-task training, Tier 3.

Both are **verified to run**. On this repo (a `uv` project):

```bash
uv add torch numpy                                   # one-time
uv run learning/01_The_Annotated_Transformer/tutorials/solutions/transformer.py
uv run learning/01_The_Annotated_Transformer/tutorials/solutions/train_copy.py
```

Expected: `transformer.py` prints `All shape checks passed.`; `train_copy.py`
drives loss/token from ~2.2 down to ~0.23 in 15 epochs and finishes with
`exact copy: True`.

## Conventions used throughout

- Tensor shapes are written as `(batch, heads, seq, d_k)` in comments — keep a
  shape in your head for every line; 90% of transformer bugs are shape/mask bugs.
- We use **pre-norm** residuals (`x + Sublayer(LayerNorm(x))`). The original paper
  is post-norm; pre-norm trains more stably and is what modern models use. Tier 4
  asks you to compare them.
