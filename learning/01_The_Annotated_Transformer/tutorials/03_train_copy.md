# Tutorial 03 — Train the copy task (Tier 3)

The copy task: given a random integer sequence, output the same sequence. It's
trivial for the model but exercises *every* moving part — batching, teacher
forcing, masking, the LR schedule, label smoothing, and autoregressive decoding.
If your copy task converges, your Transformer is wired correctly.

Reference: [`solutions/train_copy.py`](solutions/train_copy.py). It runs on CPU in
~20s and ends with `exact copy: True`.

---

## Batching + teacher forcing

During training we feed the decoder the *correct* previous tokens ("teacher
forcing") rather than its own predictions. So from one sequence we build two
shifted views:

```python
self.tgt   = tgt[:, :-1]   # decoder input:  everything but the last token
self.tgt_y = tgt[:, 1:]    # gold output:    shifted one step left
```

At position `i`, the decoder sees `tgt[:i+1]` and must predict `tgt[i+1]`. The
mask is the pad mask **AND**-ed with the subsequent mask:

```python
@staticmethod
def make_std_mask(tgt, pad):
    mask = (tgt != pad).unsqueeze(-2)
    mask = mask & subsequent_mask(tgt.size(-1)).type_as(mask)
    return mask
```

**Trap — off-by-one.** If you forget the shift and feed `tgt` as both input and
target, the model can (with the causal mask) still cheat via position alignment,
and loss won't reflect real next-token prediction. Input and target must be
offset by exactly one.

---

## 3.1 Noam learning-rate schedule

Adam with a hand-rolled schedule: **warm up** linearly for `warmup` steps, then
**decay** as `step^-0.5`:

$$lr = d_{model}^{-0.5}\cdot\min\!\big(step^{-0.5},\; step\cdot warmup^{-1.5}\big)$$

```python
def noam_rate(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
```

**Why warmup?** Early on, Adam's variance estimates are noisy and the model is far
from any good region; a large LR then can diverge (especially with post-norm).
Ramping up avoids the early blow-up; decaying afterward settles into a minimum. We
set the LR manually each step by overwriting `param_group["lr"]`.

**Trap — LR starts at 0.** Initialize `Adam(lr=0)` and let the schedule drive it.
If you also pass a nonzero base LR that you don't overwrite, you get two schedules
fighting.

---

## 3.1 Label smoothing

Instead of a one-hot target (all mass on the gold token), put `1-ε` on the gold
token and spread `ε` across the rest. Implemented as KL-divergence against that
smoothed distribution:

```python
true_dist.fill_(self.smoothing / (self.size - 2))     # -2: skip pad + gold
true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
true_dist[:, self.padding_idx] = 0
```

**Why?** It stops the model becoming over-confident (driving a logit to +∞), which
improves calibration and BLEU/accuracy — even though it *worsens* perplexity, as
the paper notes. Padding positions get zero target mass so they contribute no loss.

**Trap — normalize the loss by tokens.** `KLDivLoss(reduction="sum")` sums over the
batch; divide by the number of real (non-pad) tokens so the loss (and gradients)
don't scale with batch size or sequence length.

---

## 3.2 Greedy decoding

Inference has no gold tokens, so generate one at a time: encode the source once,
start with the `<start>` symbol, repeatedly take the argmax next token and append:

```python
memory = model.encode(src, src_mask)
ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src)
for _ in range(max_len - 1):
    out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src))
    next_word = model.generator(out[:, -1]).argmax(dim=-1).item()
    ys = torch.cat([ys, torch.full((1, 1), next_word).type_as(src)], dim=1)
```

**Encode once, decode many.** Re-encoding the source every step wastes compute —
`memory` is constant. Note we rebuild `subsequent_mask` each step because `ys`
grows. Greedy = always take the argmax; real systems use beam search, but greedy
is enough to prove the model learned the task.

---

## Run it

```bash
uv run learning/01_The_Annotated_Transformer/tutorials/solutions/train_copy.py
```

Expected (loss/token falls steadily; exact copy at the end):

```
epoch  0  loss/token 2.184  step 20
...
epoch 14  loss/token 0.228  step 300

source: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
decode: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exact copy: True
```

If loss stalls near the start-of-training value, check, in order: the tgt
shift, the mask AND, the LR schedule actually being applied, and Xavier init.

Working model that learns? Now **break it on purpose** to understand it:
[04_ablations.md](04_ablations.md).
