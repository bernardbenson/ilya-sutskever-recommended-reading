# Tutorial 03 — Train and generate with temperature (Tier 3)

You have a trainable LSTM. Now the fun part of the blog post: point it at a corpus,
train until it captures the *style*, and use a **temperature** knob to steer
generation between "safe and repetitive" and "wild and creative." Reference:
[`solutions/char_rnn.py`](solutions/char_rnn.py).

---

## 3.1 Train on a corpus

The training loop is standard once batching (Tutorial 02) is in place: Adam, cross-
entropy on next-char, grad-norm clipping, repeat.

```python
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(epochs):
    state = model.init_state(batch_size, device)
    for xb, yb in make_batches(data, batch_size, seq_len):
        state = detach_state(state)
        logits, state = model(xb, state)
        loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
```

**Swap the corpus, change nothing else.** Replace the toy `DATA` string with
`open("shakespeare.txt").read()` (or a source-code file, or your own writing). The
model, loss, and loop are identical — this is the point of character-level modeling:
it's format-agnostic. Karpathy's post trains this exact recipe on Shakespeare, the
Linux kernel, Wikipedia, and LaTeX.

**What "learning the style" means.** Character LM never learns *facts*; it learns the
*surface statistics* of the corpus — line lengths, punctuation habits, which letters
follow which, how to open and close brackets. On code it produces plausible-looking
(non-compiling) functions; on Shakespeare, iambic-ish dialogue with speaker names.

---

## 3.2 Temperature sampling

At generation time, divide the logits by a temperature `T` before softmax:

```python
@torch.no_grad()
def sample(model, device, prime="the ", length=200, temperature=1.0):
    state = model.init_state(1, device)
    x = torch.tensor([[stoi[c] for c in prime]], device=device)
    logits, state = model(x, state)
    out = list(prime)
    for _ in range(length):
        logits = logits[:, -1, :] / temperature          # <-- the knob
        probs = torch.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)      # sample from it
        out.append(itos[ix.item()])
        logits, state = model(ix, state)                  # feed back one char
    return "".join(out)
```

**What `T` does to the distribution:**
- `T → 0`: approaches argmax — always the single most likely character. Safe,
  grammatical, and prone to falling into repetitive loops.
- `T = 1`: sample from the model's true distribution — the honest setting.
- `T > 1`: flattens the distribution toward uniform — more surprising choices, more
  typos and broken words.

**Priming / conditioning.** We seed the state by first running the model over a prime
string (`"the "`). Generation continues *from that context* — the same mechanism a
chatbot uses when it conditions on your prompt before replying.

---

## Run it

```bash
uv run learning/03_Unreasonable_Effectiveness_of_RNNs/tutorials/solutions/char_rnn.py
```
```
[temperature 0.5]
'the quick brown fox jumps over the lazy dog. the lazy dog sleeps while the quick brown fox runs...'

[temperature 1.0]
'the quick brown fox runs. a quick fox is a clever fox, and a lazy dog is a sleepy dog...'

[temperature 1.5]
'the quickr brown fox rumm squwmkc fox ru amc brown fox lummpsr wher the lazy do. whis la vlayy oog...'
```

The trade-off is right there: `0.5` reproduces the corpus almost verbatim, `1.0`
recombines it fluently, `1.5` slides into creative gibberish. Being able to *dial*
that is the practical payoff of the whole exercise — and the same temperature
parameter is exposed by every modern LLM API.

**Trap — temperature at training time.** Temperature is a *sampling-time* concept
only. Don't divide logits by `T` while computing the training loss; the model is
trained at `T=1` and you choose `T` afterward when you generate.

Once you can steer generation, break the model on purpose to understand its limits:
[04_ablations.md](04_ablations.md).
