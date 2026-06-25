import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


# -------------------------
# hyperparameter defaults
# -------------------------
batch_size = 32         # how many text chunks we train on at once
block_size = 128        # how many characters of context the model sees
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100

n_embd = 128            # size of each token vector
n_head = 4              # number of attention heads
n_layer = 4             # number of transformer blocks
dropout = 0.1


# -------------------------
# tokenizer state (populated by build_vocab before model init)
# -------------------------
stoi: dict = {}
itos: dict = {}
vocab_size: int = 0


def build_vocab(text: str) -> None:
    global stoi, itos, vocab_size
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list:
    return [stoi[c] for c in s]


def decode(ids) -> str:
    return "".join(itos[i] for i in ids)


# -------------------------
# data utilities
# -------------------------
def get_batch(source: torch.Tensor):
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    out = {}
    model.eval()
    for split, source in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [B, T]
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                    # [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))     # [T, C]
        x = tok_emb + pos_emb   # [B, T, C]
        x = self.blocks(x)      # [B, T, C]
        x = self.ln_f(x)        # [B, T, C]
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
            logits, _ = self(idx_cond)
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
# standalone run (original behavior preserved)
# -------------------------
def run_standalone():
    torch.manual_seed(1337)

    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    build_vocab(text)
    data = torch.tensor(encode(text), dtype=torch.long)

    # train/validation split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = MicroGPT().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"vocab size: {vocab_size}")
    print(f"parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {iter}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data)
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


# -------------------------
# benchwarmer run
# -------------------------
def run_benchwarmer():
    global batch_size, block_size, max_iters, learning_rate

    bench_input = json.loads(Path("bench_input.json").read_text(encoding="utf-8"))
    run_id = bench_input["run_id"]
    seed = bench_input["seed"]
    mode = bench_input["mode"]
    dataset = bench_input["dataset"]
    hparams = bench_input.get("hyperparameters", {})
    inference_cfg = bench_input.get("inference", {})

    # apply hyperparameters before model init (override module globals)
    if "lr" in hparams:
        learning_rate = float(hparams["lr"])
    if "batch_size" in hparams:
        batch_size = int(hparams["batch_size"])
    if "max_steps" in hparams:
        max_iters = int(hparams["max_steps"])
    if "max_seq_len" in hparams:
        block_size = int(hparams["max_seq_len"])

    # seed before anything else
    torch.manual_seed(seed)

    train_path = dataset["train_path"]
    eval_path = dataset.get("eval_path")
    prompts_path = dataset.get("prompts_path")

    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    eval_text = ""
    if eval_path:
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_text = f.read()

    # union vocab so eval characters are never OOV
    build_vocab(train_text + eval_text)

    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    eval_data = torch.tensor(encode(eval_text), dtype=torch.long) if eval_text else None

    # sanity-check dataset size before touching the model
    min_tokens = block_size + 1
    if len(train_data) < min_tokens:
        raise ValueError(
            f"train dataset is too small: {len(train_data)} tokens, "
            f"need at least {min_tokens} (block_size={block_size} + 1). "
            f"train_path={train_path!r}"
        )

    # architecture block (total_params filled after model init)
    architecture = {
        "name": "microgpt",
        "total_params": None,
        "tokenizer": {"name": "char", "vocab_size": vocab_size},
        "layers": (
            [{"type": "embedding"}]
            + [{"type": "transformer_block"}] * n_layer
            + [{"type": "lm_head"}]
        ),
        "n_layers": n_layer,
        "d_model": n_embd,
    }

    training_output = None
    inference_output = None
    status = "complete"

    try:
        model = MicroGPT().to(device)
        num_params = sum(p.numel() for p in model.parameters())
        architecture["total_params"] = num_params

        print(f"device: {device}", flush=True)
        print(f"vocab size: {vocab_size}", flush=True)
        print(f"parameters: {num_params / 1e6:.2f}M", flush=True)

        # ---- training phase ----
        if mode in ("training", "both"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            steps_log = []
            tokens_seen = 0
            t0 = time.perf_counter()

            eval_src = eval_data if eval_data is not None else train_data

            for it in range(max_iters):
                do_eval = (it % eval_interval == 0) or (it == max_iters - 1)

                if do_eval:
                    losses = estimate_loss(model, train_data, eval_src)
                    print(
                        f"step {it}: "
                        f"train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}"
                    )

                xb, yb = get_batch(train_data)
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5

                optimizer.step()
                tokens_seen += batch_size * block_size

                step_record: dict = {
                    "step": it + 1,
                    "train_loss": loss.item(),
                    "grad_norm": grad_norm,
                    "lr": learning_rate,
                }
                if do_eval:
                    step_record["val_loss"] = losses["val"]
                steps_log.append(step_record)

            training_output = {
                "steps": steps_log,
                "tokens_seen": tokens_seen,
                "wall_clock_sec": time.perf_counter() - t0,
            }

            torch.save(model.state_dict(), "microgpt.pt")
            print("saved model to microgpt.pt")

        # ---- inference phase ----
        if mode in ("inference", "both"):
            # in inference-only mode try to load a prior checkpoint
            if mode == "inference" and Path("microgpt.pt").exists():
                model.load_state_dict(torch.load("microgpt.pt", map_location=device))
                print("loaded model from microgpt.pt")

            # --- eval logprobs (required for bits_per_byte) ---
            logprob_records = []
            if eval_data is not None and len(eval_data) > 1:
                model.eval()
                with torch.no_grad():
                    eval_tensor = eval_data.to(device)
                    # slide a block_size window, scoring each next token
                    for i in range(0, len(eval_data) - 1, block_size):
                        chunk_x = eval_tensor[i : i + block_size].unsqueeze(0)      # [1, T]
                        chunk_y = eval_tensor[i + 1 : i + block_size + 1]           # [T]
                        logits, _ = model(chunk_x)                                  # [1, T, V]
                        log_probs = F.log_softmax(logits[0], dim=-1)                # [T, V]
                        for t in range(len(chunk_y)):
                            token_id = chunk_y[t].item()
                            logprob = log_probs[t, token_id].item()
                            # byte_len = UTF-8 bytes of the surface character
                            byte_len = len(itos[token_id].encode("utf-8"))
                            logprob_records.append({
                                "token_id": token_id,
                                "logprob": logprob,
                                "byte_len": byte_len,
                            })
                model.train()

                Path("eval_logprobs.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in logprob_records) + "\n",
                    encoding="utf-8",
                )

            # --- generations from prompts ---
            generations = []
            timing = None

            if prompts_path:
                prompts_text = Path(prompts_path).read_text(encoding="utf-8")
                prompts = [line.strip() for line in prompts_text.splitlines() if line.strip()]

                max_new_tokens = inference_cfg.get("max_new_tokens", 100)
                samples_per_prompt = inference_cfg.get("samples_per_prompt", 1)

                total_prefill_sec = 0.0
                total_prefill_tokens = 0
                total_decode_sec = 0.0
                total_tokens_gen = 0

                model.eval()
                with torch.no_grad():
                    for prompt in prompts:
                        # skip characters not in vocab rather than crashing
                        prompt_ids = [stoi[c] for c in prompt if c in stoi]
                        prompt_tokens = len(prompt_ids)
                        if prompt_tokens == 0:
                            prompt_ids = [0]
                            prompt_tokens = 1

                        for s in range(samples_per_prompt):
                            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

                            t_pre = time.perf_counter()
                            _ = model(idx)   # prefill
                            prefill_sec = time.perf_counter() - t_pre

                            t_dec = time.perf_counter()
                            idx_out = model.generate(idx, max_new_tokens)
                            decode_sec = time.perf_counter() - t_dec

                            generated_ids = idx_out[0, prompt_tokens:].tolist()
                            output_text = decode(generated_ids)
                            tokens_gen = len(generated_ids)

                            generations.append({
                                "prompt": prompt,
                                "sample_index": s,
                                "output": output_text,
                                "prompt_tokens": prompt_tokens,
                                "tokens_generated": tokens_gen,
                            })

                            total_prefill_sec += prefill_sec
                            total_prefill_tokens += prompt_tokens
                            total_decode_sec += decode_sec
                            total_tokens_gen += tokens_gen

                timing = {
                    "prefill_sec": total_prefill_sec,
                    "prefill_tokens": total_prefill_tokens,
                    "decode_sec": total_decode_sec,
                    "tokens_generated": total_tokens_gen,
                }
                model.train()

            inference_output = {}
            if logprob_records:
                inference_output["eval_logprobs_path"] = "eval_logprobs.jsonl"
            if generations:
                inference_output["generations"] = generations
            if timing:
                inference_output["timing"] = timing

    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        status = "failed"

    output: dict = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "architecture": architecture,
    }
    if training_output is not None:
        output["training"] = training_output
    if inference_output is not None:
        output["inference"] = inference_output

    Path("bench_output.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    sys.exit(0 if status == "complete" else 1)


# -------------------------
# entry point
# -------------------------
if __name__ == "__main__":
    if Path("bench_input.json").exists():
        run_benchwarmer()
    else:
        run_standalone()
