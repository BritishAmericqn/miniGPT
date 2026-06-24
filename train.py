import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters

batch_size = 32        # how many text chunks we train on at once
block_size = 128       # how many characters of context the model sees
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100

n_embd = 128           # size of each token vector
n_head = 4             # number of attention heads
n_layer = 4            # number of transformer blocks
dropout = 0.1

torch.manual_seed(1337)

# -------------------------
# load data
# -------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)

# train/validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    source = train_data if split == "train" else val_data

    # random starting positions
    ix = torch.randint(len(source) - block_size, (batch_size,))

    # x is current characters, y is next characters
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

# -------------------------
# model pieces
# -------------------------

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # lower-triangular mask: token can only see previous tokens
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape

        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]

        # attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # [B, T, T]

        # mask future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # convert scores into probabilities
        wei = F.softmax(wei, dim=-1)  # [B, T, T]
        wei = self.dropout(wei)

        v = self.value(x)  # [B, T, head_size]

        # weighted average of values
        out = wei @ v      # [B, T, head_size]

        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate all head outputs along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, C]
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """MLP: expand, activate, compress."""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # up-projection
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),  # down-projection
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication, then computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connection around attention
        x = x + self.sa(self.ln1(x))

        # residual connection around MLP
        x = x + self.ffwd(self.ln2(x))

        return x


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [B, T]
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # [T, C]

        x = tok_emb + pos_emb  # [B, T, C]
        x = self.blocks(x)     # [B, T, C]
        x = self.ln_f(x)       # [B, T, C]

        logits = self.lm_head(x)  # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape

            # flatten batch and time so every token position is a training example
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: [B, T]
        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # focus only on the final time step
            logits = logits[:, -1, :]  # [B, vocab_size]

            temperature = 0.4
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # sample from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # append sampled token
            idx = torch.cat((idx, idx_next), dim=1)  # [B, T+1]

        return idx


# -------------------------
# train
# -------------------------

model = MicroGPT().to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"device: {device}")
print(f"vocab size: {vocab_size}")
print(f"parameters: {num_params / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from a blank/newline prompt
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()

print("---- generated text ----")
print(decode(generated))

torch.save(model.state_dict(), "microgpt.pt")
print("saved model to microgpt.pt")