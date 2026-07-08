"""Tier 3 reference solution: train the Transformer on the copy task.

The model learns to copy a sequence of random integers (src == tgt). This is
the canonical Annotated Transformer sanity check: it exercises the whole
encoder-decoder loop, masking, the Noam LR schedule, and label smoothing.

Run: `python train_copy.py`  (CPU is fine; ~20s)
"""

import torch
import torch.nn as nn

from transformer import make_model, subsequent_mask


# ---------------------------------------------------------------------------
# Batch: holds src/tgt and builds the masks. tgt is shifted for teacher forcing.
# ---------------------------------------------------------------------------
class Batch:
    def __init__(self, src, tgt, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # (batch, 1, seq)
        # tgt input is everything but the last token; tgt_y is shifted by one.
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt, pad)
        self.ntokens = (self.tgt_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        mask = (tgt != pad).unsqueeze(-2)
        mask = mask & subsequent_mask(tgt.size(-1)).type_as(mask)
        return mask


def data_gen(vocab, batch_size, nbatches, seq_len=10):
    """Random sequences whose first token is forced to 1 (a <start> symbol)."""
    for _ in range(nbatches):
        data = torch.randint(1, vocab, (batch_size, seq_len))
        data[:, 0] = 1
        src = data.clone()
        tgt = data.clone()
        yield Batch(src, tgt, pad=0)


# ---------------------------------------------------------------------------
# 3.1  Label smoothing (KL-div against a smoothed one-hot target)
# ---------------------------------------------------------------------------
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        # x: (N, vocab) log-probs ; target: (N,) gold indices
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0 and mask.numel() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


# ---------------------------------------------------------------------------
# 3.1  Noam learning-rate schedule
# ---------------------------------------------------------------------------
def noam_rate(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.reshape(-1, x.size(-1)), y.reshape(-1)) / norm
        return loss


def run_epoch(data_iter, model, loss_compute, optimizer, d_model, warmup, step0=0):
    total_loss, total_tokens = 0.0, 0
    step = step0
    for batch in data_iter:
        step += 1
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        loss.backward()
        for g in optimizer.param_groups:
            g["lr"] = noam_rate(step, d_model, warmup)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * batch.ntokens.item()
        total_tokens += batch.ntokens.item()
    return total_loss / total_tokens, step


# ---------------------------------------------------------------------------
# 3.2  Greedy decoding
# ---------------------------------------------------------------------------
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src)
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src))
        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src).fill_(next_word)], dim=1)
    return ys


def main():
    torch.manual_seed(1)
    V = 11
    d_model, warmup = 128, 400

    model = make_model(V, V, n=2, d_model=d_model, d_ff=256, h=8)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    loss_compute = SimpleLossCompute(model.generator, criterion)

    step = 0
    for epoch in range(15):
        model.train()
        avg_loss, step = run_epoch(
            data_gen(V, batch_size=32, nbatches=20), model, loss_compute, optimizer, d_model, warmup, step
        )
        print(f"epoch {epoch:2d}  loss/token {avg_loss:.3f}  step {step}")

    # Evaluate greedy copy.
    model.eval()
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, src.size(1))
    out = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print("\nsource:", src.tolist()[0])
    print("decode:", out.tolist()[0])
    match = out[0].tolist() == src[0].tolist()
    print("exact copy:", match)


if __name__ == "__main__":
    main()
