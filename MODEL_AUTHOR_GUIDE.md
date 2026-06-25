# Model Author Guide — Making Your Repo Benchwarmer-Compatible

**Give this file (or its URL) to your agent.** It contains everything needed to make a model repo pass `benchwarmer validate` and produce valid benchmark results.

---

## What benchwarmer expects from your model

Benchwarmer runs your model as an isolated subprocess. The contract is simple:

1. Benchwarmer writes `bench_input.json` to your repo directory.
2. Benchwarmer runs your entry point (default: `python train.py`).
3. Your script reads `bench_input.json`, trains/infers, and writes `bench_output.json`.
4. Benchwarmer reads `bench_output.json` and computes all metrics itself.

**The golden rule:** your script reports raw artifacts only. Never compute or report derived comparison metrics (perplexity, bits_per_byte, FLOPs, throughput) — benchwarmer computes those from your raw outputs so they are consistent and comparable across models.

---

## Prerequisites: install benchwarmer

The `benchwarmer` command needs to be on your PATH. The recommended approach is
[pipx](https://pipx.pypa.io/), which keeps it isolated and makes it easy to uninstall:

```bash
# Install pipx if you don't have it:
brew install pipx   # macOS  |  or: pip install --user pipx

# Clone the benchwarmer repo and install from it:
git clone https://github.com/your-org/llm-benchwarmer
cd llm-benchwarmer
pipx install -e .

# Verify (should print the help menu):
benchwarmer --help

# To uninstall later:
pipx uninstall llm-benchwarmer
```

Once installed, `benchwarmer` works from any directory on your machine — including inside
your model repos.

---

## Quickstart: validate your repo

```bash
# From inside your model repo:
benchwarmer validate . --mode both --timeout 120

# If it passes, register and run:
benchwarmer add . --trust
benchwarmer run . --seeds 3 --output bench_result.json
```

---

## Step 1 — Optional: add `benchwarmer.toml`

Create `benchwarmer.toml` at the root of your repo to override defaults:

```toml
[invoke]
entry_point = "python train.py"   # default; change if your entry point differs
working_dir = "."                  # relative to repo root
# setup_command = "pip install -r requirements.txt"  # run once before first invoke
# env = { TOKENIZERS_PARALLELISM = "false" }         # extra env vars

[contract]
input_file  = "bench_input.json"   # default
output_file = "bench_output.json"  # default
```

If this file is absent, all defaults apply.

---

## Step 2 — Read `bench_input.json`

Benchwarmer writes this before calling your script. Parse it at startup:

```python
import json
from pathlib import Path

bench_input = json.loads(Path("bench_input.json").read_text(encoding="utf-8"))

run_id    = bench_input["run_id"]          # str; echo this in bench_output.json
seed      = bench_input["seed"]            # int; seed your RNG with this
mode      = bench_input["mode"]            # "training" | "inference" | "both"
dataset   = bench_input["dataset"]         # paths to train/eval/prompts files
hparams   = bench_input.get("hyperparameters", {})
inference = bench_input.get("inference", {})
```

### `bench_input.json` schema

```json
{
  "schema_version": "1.0",
  "run_id": "abc-123",
  "seed": 42,
  "mode": "both",
  "dataset": {
    "name": "tinystories-v1",
    "train_path": "/path/to/train.txt",
    "eval_path":  "/path/to/eval.txt",
    "prompts_path": "/path/to/prompts.txt"   // null if no inference prompts
  },
  "hyperparameters": {
    "lr": 0.001,
    "batch_size": 32,
    "max_steps": 1000,
    "max_seq_len": 128
  },
  "inference": {
    "max_new_tokens": 100,
    "samples_per_prompt": 1
  },
  "trajectory": {
    "capture": false,
    "every_n_steps": 50
  }
}
```

**Key rules:**
- Set your random seed to `bench_input["seed"]` before anything else.
- Read `mode` and only run the requested phase(s).
- All dataset paths are absolute; use them directly.
- `hyperparameters` are passed through from `benchwarmer run --set k=v` flags.

---

## Step 3 — Write `bench_output.json`

Write this file before exiting. The `run_id` must match exactly.

### Minimal example (training only)

```json
{
  "schema_version": "1.0",
  "run_id": "abc-123",
  "status": "complete",
  "architecture": {
    "name": "my-gpt",
    "total_params": 10000000,
    "tokenizer": {"name": "gpt2", "vocab_size": 50257},
    "layers": [
      {"type": "embedding"},
      {"type": "transformer_block"},
      {"type": "transformer_block"},
      {"type": "lm_head"}
    ],
    "n_layers": 2,
    "d_model": 256
  },
  "training": {
    "steps": [
      {"step": 1, "train_loss": 3.2, "val_loss": 3.4, "grad_norm": 1.1, "lr": 0.001},
      {"step": 2, "train_loss": 3.0, "val_loss": 3.2, "grad_norm": 0.9, "lr": 0.001}
    ],
    "tokens_seen": 65536,
    "wall_clock_sec": 5.2,
    "peak_memory_mb": 512.0
  }
}
```

### Full example (training + inference)

```json
{
  "schema_version": "1.0",
  "run_id": "abc-123",
  "status": "complete",
  "architecture": {
    "name": "my-gpt",
    "total_params": 10000000,
    "tokenizer": {"name": "gpt2", "vocab_size": 50257},
    "layers": [{"type": "transformer_block"}, {"type": "transformer_block"}],
    "n_layers": 2,
    "d_model": 256
  },
  "training": {
    "steps": [
      {"step": 1, "train_loss": 3.2, "val_loss": 3.4, "grad_norm": 1.1, "lr": 0.001}
    ],
    "tokens_seen": 65536,
    "wall_clock_sec": 5.2
  },
  "inference": {
    "eval_logprobs_path": "eval_logprobs.jsonl",
    "generations": [
      {
        "prompt": "Once upon a time",
        "sample_index": 0,
        "output": "there was a small dragon.",
        "prompt_tokens": 4,
        "tokens_generated": 5
      }
    ],
    "timing": {
      "prefill_sec": 0.01,
      "prefill_tokens": 4,
      "decode_sec": 0.05,
      "tokens_generated": 5
    },
    "peak_memory_mb": 256.0
  }
}
```

### `bench_output.json` field reference

| Field | Required | Description |
|---|---|---|
| `schema_version` | ✅ | Must be `"1.0"` |
| `run_id` | ✅ | Must exactly match `bench_input.run_id` |
| `status` | ✅ | `"complete"` or `"failed"` |
| `architecture.name` | ✅ | Human-readable model name |
| `architecture.total_params` | ✅ | Total parameter count (integer) |
| `architecture.tokenizer.name` | ✅ | Tokenizer name (for labeling) |
| `architecture.tokenizer.vocab_size` | ✅ | Vocabulary size |
| `architecture.layers` | ✅ | List of layer dicts; each must have `"type"` |
| `architecture.n_layers` | recommended | Number of transformer blocks (for FLOPs calculation) |
| `architecture.d_model` | recommended | Hidden dimension (for FLOPs calculation) |
| `training.steps` | if mode includes training | List of per-step records |
| `training.steps[].step` | ✅ | Step number (integer) |
| `training.steps[].train_loss` | ✅ | Training loss (float) |
| `training.steps[].val_loss` | optional | Validation loss |
| `training.steps[].grad_norm` | optional | Gradient norm |
| `training.steps[].lr` | optional | Learning rate at this step |
| `training.tokens_seen` | ✅ | Total tokens processed |
| `training.wall_clock_sec` | ✅ | Your measured wall time (benchwarmer also measures independently) |
| `training.peak_memory_mb` | optional | Peak memory in MB |
| `inference.eval_logprobs_path` | optional | Relative path to `eval_logprobs.jsonl` |
| `inference.generations` | optional | List of generated outputs |
| `inference.timing` | optional | Prefill/decode split timing |

**Never include:** `perplexity`, `bits_per_byte`, `flops`, `tokens_per_sec`, `throughput`, `converged_step`, or any other derived metric. Benchwarmer will log a warning and drop these fields.

---

## Step 4 — Write `eval_logprobs.jsonl` (for quality metrics)

This enables `bits_per_byte` (the default leaderboard metric) and `perplexity`. Without it, quality metrics are skipped.

**One JSON object per line:**

```jsonl
{"token_id": 1234, "logprob": -2.31, "byte_len": 3}
{"token_id": 5678, "logprob": -1.05, "byte_len": 2}
{"token_id": 0,    "logprob": -0.80, "byte_len": 0}
```

### Rules

| Rule | Why |
|---|---|
| `logprob` is natural log (ln), not log₂ or log₁₀ | `bits_per_byte = (-Σlogprob / ln(2)) / Σbyte_len` |
| `logprob ≤ 0` and finite | Natural log of a probability; >0 or ±inf fails sanity checks |
| `byte_len` = UTF-8 byte length of the token's surface string | The denominator of bits_per_byte |
| `byte_len = 0` for special tokens (BOS, EOS, PAD) | Excluded from numerator AND denominator — they can't inflate the score |
| `sum(byte_len)` ≈ byte length of `eval.txt` (within ±0.5%) | Ensures bits_per_byte is genuinely tokenizer-agnostic |

**The byte-coverage rule is the most common failure.** If your `sum(byte_len)` doesn't cover the eval text within 0.5%, benchwarmer rejects the run with a clear error. To compute it correctly:

```python
# For each token your model processes over eval.txt:
surface_text = tokenizer.decode([token_id])   # the surface string for this token
byte_len = len(surface_text.encode("utf-8"))  # UTF-8 bytes

# Special tokens have no surface text:
if token_id in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}:
    byte_len = 0
```

---

## Step 5 — Handle failures gracefully

If your training crashes, exit nonzero and write as much diagnostic output to stderr as possible. You can also write `bench_output.json` with `"status": "failed"` before exiting — benchwarmer will record the failed run.

```python
import sys

try:
    # ... training ...
    status = "complete"
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    status = "failed"

output = {
    "schema_version": "1.0",
    "run_id": run_id,
    "status": status,
    "architecture": architecture,
}
if status == "complete":
    output["training"] = training_output

Path("bench_output.json").write_text(json.dumps(output), encoding="utf-8")
sys.exit(0 if status == "complete" else 1)
```

---

## Validate before submitting results

```bash
benchwarmer validate /path/to/your-model --mode both --timeout 120
```

This runs your model once, checks every contract requirement, and reports exactly what passes or fails — without storing any results.

---

## Reference implementation

`examples/reference-model/train.py` in the benchwarmer repo is a complete working implementation of this contract. It uses stub training/inference logic but demonstrates every step with comments. Copy it as a starting point.

```bash
# Clone it:
cp /path/to/benchwarmer/examples/reference-model/train.py ./train.py
cp /path/to/benchwarmer/examples/reference-model/benchwarmer.toml ./benchwarmer.toml

# Validate immediately (should pass out of the box):
benchwarmer validate . --mode both
```

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `run_id mismatch` | Forgot to echo `bench_input["run_id"]` in output | Set `output["run_id"] = run_id` |
| `logprob X is positive` | Using log₂ or log₁₀ instead of ln | Use `math.log(prob)` not `math.log2` or `math.log10` |
| `byte coverage mismatch` | `sum(byte_len)` ≠ `eval.txt` byte count | See §4 byte-coverage rule above |
| `Missing 'training' block` | Mode is "training" but `output["training"]` absent | Always write the block for the requested mode |
| `bench_output.json not found` | Script exited 0 but didn't write the file | Ensure you write the file before `sys.exit(0)` |
| `perplexity was ignored` | Computed perplexity inside your script | Remove it — benchwarmer computes this from logprobs |

---

## What benchwarmer computes for you

Once your script produces raw artifacts, benchwarmer derives:

- **bits_per_byte** — tokenizer-agnostic quality metric (default leaderboard sort)
- **perplexity** — per-token; same-tokenizer only (labeled, not ranked cross-model)
- **FLOPs** — `6 × params × tokens` + attention term; uses your `n_layers`/`d_model` hints
- **throughput** — prefill and decode tokens/sec from your timing
- **latency** — p50/p99 per-token latency
- **convergence** — FLOPs to reach a target val_loss; generalization gap
- **behavior** — repetition rate, distinct-N, mean output length

All metrics are tagged with their trust level:
- `benchwarmer_measured` — fully computed by benchwarmer (behavior metrics, model_size_on_disk)
- `derived_from_reported` — computed from your raw outputs (all the above)

Only `benchwarmer_measured` and `derived_from_reported` metrics appear in leaderboards.
