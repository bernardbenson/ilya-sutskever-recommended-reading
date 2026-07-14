"""Tier 2-3 reference solution: a character-level LSTM in PyTorch.

Same task as `min_char_rnn.py` (predict the next character), but now the RNN is
an `nn.LSTM`, the backward pass is autograd's job, and we train in mini-batches
with truncated backprop through time. This is the setup Karpathy's `char-rnn`
Torch project used to generate Shakespeare, Linux source, and Wikipedia.

What to notice versus the NumPy version:
  * LSTM replaces the vanilla tanh recurrence -> long-range dependencies survive
    (the gates give gradients a protected path; see paper 04, Understanding LSTMs).
  * We batch many sequence chunks at once and carry (h, c) between chunks
    (truncated BPTT) instead of resetting every step.
  * Sampling uses a temperature knob to trade coherence for diversity.

Run: `python char_rnn.py`   (~30-60s on CPU; prints loss + temperature samples)
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA = (
    "the quick brown fox jumps over the lazy dog. "
    "the lazy dog sleeps while the quick brown fox runs. "
    "a quick fox is a clever fox, and a lazy dog is a sleepy dog. "
) * 60

chars = sorted(list(set(DATA)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([stoi[c] for c in DATA], dtype=torch.long)


# ---------------------------------------------------------------------------
# Model: embedding -> LSTM -> linear projection back to vocab logits.
# ---------------------------------------------------------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed=32, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(
            embed, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        # x: (batch, seq) of char indices
        emb = self.embed(x)                    # (batch, seq, embed)
        out, state = self.lstm(emb, state)     # (batch, seq, hidden)
        logits = self.fc(out)                  # (batch, seq, vocab)
        return logits, state

    def init_state(self, batch, device):
        h = torch.zeros(self.layers, batch, self.hidden, device=device)
        c = torch.zeros(self.layers, batch, self.hidden, device=device)
        return (h, c)


# ---------------------------------------------------------------------------
# Batching: cut the stream into `batch_size` parallel rows, then walk it in
# windows of `seq_len`. Carry (h, c) across windows == truncated BPTT.
# ---------------------------------------------------------------------------
def make_batches(data, batch_size, seq_len):
    n = (data.size(0) - 1) // (batch_size * seq_len) * (batch_size * seq_len)
    x = data[:n].view(batch_size, -1)          # (batch, stream_len)
    y = data[1:n + 1].view(batch_size, -1)     # targets shifted by one
    for i in range(0, x.size(1), seq_len):
        xb = x[:, i:i + seq_len]
        yb = y[:, i:i + seq_len]
        if xb.size(1) == seq_len:
            yield xb, yb


def detach_state(state):
    # Stop gradients flowing past the truncation window.
    return tuple(s.detach() for s in state)


@torch.no_grad()
def sample(model, device, prime="the ", length=200, temperature=1.0):
    """Prime the model, then autoregressively sample `length` chars.

    temperature < 1 sharpens the distribution (safer, more repetitive);
    temperature > 1 flattens it (more diverse, more mistakes).
    """
    model.eval()
    state = model.init_state(1, device)
    out = list(prime)
    x = torch.tensor([[stoi[c] for c in prime]], dtype=torch.long, device=device)
    logits, state = model(x, state)
    for _ in range(length):
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)   # sample from the distribution
        out.append(itos[ix.item()])
        logits, state = model(ix, state)
    return "".join(out)


def main():
    torch.manual_seed(1)
    device = "cpu"
    batch_size, seq_len, epochs = 16, 40, 12

    model = CharLSTM(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"corpus {len(DATA)} chars, vocab {vocab_size}, "
          f"params {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(epochs):
        model.train()
        state = model.init_state(batch_size, device)
        total, count = 0.0, 0
        for xb, yb in make_batches(data, batch_size, seq_len):
            state = detach_state(state)
            logits, state = model(xb.to(device), state)
            loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1).to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # same idea as NumPy clip
            opt.step()
            total += loss.item()
            count += 1
        print(f"epoch {epoch:2d}  loss {total / count:.3f}")

    print("\n--- samples ---")
    for temp in (0.5, 1.0, 1.5):
        txt = sample(model, device, prime="the ", length=180, temperature=temp)
        print(f"\n[temperature {temp}]\n{txt!r}")


if __name__ == "__main__":
    main()
