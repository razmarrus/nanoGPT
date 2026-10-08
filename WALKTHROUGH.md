# nanoGPT Developer Walkthrough

Step-by-step guide to reading the codebase: data preparation → training → model → inference.

---

## 1. Entry Points (Repo Map)

| Script / path | Role |
|---|---|
| `data/*/prepare.py` | One-time: download raw text, tokenize, write `train.bin` / `val.bin` (+ `meta.pkl` for char-level) |
| `train.py` | Main training loop: load data, build model, train/eval, save `out_dir/ckpt.pt` |
| `model.py` | GPT architecture: `GPTConfig`, `GPT`, attention blocks, `from_pretrained`, `generate` |
| `sample.py` | Inference: load checkpoint or GPT-2 weights, encode prompt, autoregressive generation |
| `configurator.py` | Config engine: `exec(config_file)` then `--key=value` CLI overrides |
| `config/*.py` | Preset hyperparameters (dataset, model size, LR, `out_dir`, `init_from`) |
| `bench.py` | Stripped-down benchmark of the training forward/backward (no full loop) |

**Typical commands:**

```sh
# Prepare data (once)
python data/shakespeare_char/prepare.py

# Train
python train.py config/train_shakespeare_char.py --device=cpu --compile=False

# Sample
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

**Config files at a glance:**

| Config | Purpose |
|---|---|
| `train_shakespeare_char.py` | Small char-level GPT on Shakespeare (~3 min GPU) |
| `train_gpt2.py` | Full GPT-2 124M on OpenWebText (multi-GPU, days) |
| `finetune_shakespeare.py` | Finetune `gpt2-xl` on BPE-tokenized Shakespeare |
| `eval_gpt2*.py` | Load pretrained GPT-2, run eval only (`eval_only=True`) |

---

## 2. Data Pipeline

### Flow

```
raw text → prepare.py → train.bin / val.bin [+ meta.pkl]
                              ↓
                    train.py: get_batch() → (x, y) tensors
```

### Three prepare scripts compared

| Dataset | Script | Tokenizer | Artifacts | Vocab |
|---|---|---|---|---|
| `shakespeare_char` | `data/shakespeare_char/prepare.py` | Char-level (`stoi`/`itos`) | `train.bin`, `val.bin`, `meta.pkl` | ~65 chars |
| `shakespeare` | `data/shakespeare/prepare.py` | GPT-2 BPE (`tiktoken`) | `train.bin`, `val.bin` | 50257 (implicit) |
| `openwebtext` | `data/openwebtext/prepare.py` | GPT-2 BPE + EOT per doc | `train.bin`, `val.bin` | 50257 (implicit) |

### Binary format contract

- **Files:** `data/<dataset>/train.bin`, `data/<dataset>/val.bin`
- **dtype:** `np.uint16` (max token id 50256 fits in uint16)
- **Layout:** flat 1D array of token IDs, no headers or document boundaries (except EOT tokens embedded in OWT)
- **Split:** 90/10 train/val (char + shakespeare); OWT uses HF `train_test_split(0.0005)`

### `meta.pkl` (char-level only)

```python
meta = {'vocab_size': int, 'stoi': dict, 'itos': dict}
```

Used by:
- `train.py` lines 138–144: sets `model_args['vocab_size']` from `meta['vocab_size']`
- `sample.py` lines 57–68: char-level encode/decode for generation

### `get_batch(split)` in `train.py` (lines 116–131)

```python
data = np.memmap(..., dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
x = data[i : i+block_size]       # input tokens
y = data[i+1 : i+1+block_size]   # targets = next token (shifted by 1)
```

- Recreates memmap every batch (avoids memory leak)
- Output shape: `(batch_size, block_size)` as `torch.long`
- Random contiguous windows — no epoch boundaries, no DataLoader class

---

## 3. Training Control Flow (`train.py`)

### Startup sequence

```
defaults in train.py
    → exec(configurator.py)     # config file + CLI overrides
    → DDP init (if RANK env set)
    → os.makedirs(out_dir)
    → load meta.pkl → vocab_size
    → model init (scratch / resume / gpt2*)
    → crop block_size if needed
    → configure_optimizers()
    → torch.compile (optional)
    → DDP wrap (optional)
    → training loop
```

### Config system (`configurator.py`)

1. Args without `=`: treated as config file path → `exec(open(file).read())`
2. Args with `--key=value`: override globals, type-checked via `literal_eval`
3. Order matters: config file first, then CLI overrides

### Model initialization (`init_from`)

| Value | Behavior |
|---|---|
| `'scratch'` | New `GPT(GPTConfig(...))`. Vocab from `meta.pkl` or default 50304 |
| `'resume'` | Load `out_dir/ckpt.pt`; restore model, optimizer, `iter_num`, `best_val_loss` |
| `'gpt2'`, `'gpt2-medium'`, etc. | `GPT.from_pretrained()` via HuggingFace weights |

Structural keys (`n_layer`, `n_head`, `n_embd`, `block_size`, `bias`, `vocab_size`) are forced from checkpoint on resume.

### Training loop (lines 255–333)

Each iteration:
1. Set LR via `get_lr(iter_num)` — warmup + cosine decay
2. Every `eval_interval`: `estimate_loss()` on train/val; optionally save checkpoint
3. Gradient accumulation over `gradient_accumulation_steps` micro-batches
4. Forward → scaled loss → backward → grad clip → optimizer step
5. Prefetch next batch during forward (overlap)
6. Log loss + MFU every `log_interval`
7. Stop when `iter_num > max_iters`

### Checkpoint schema (`out_dir/ckpt.pt`)

```python
{
    'model': state_dict,
    'optimizer': state_dict,
    'model_args': dict,      # n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout
    'iter_num': int,
    'best_val_loss': float,
    'config': dict,          # full training config snapshot (includes 'dataset')
}
```

Saved when: `val loss improves OR always_save_checkpoint`, and `iter_num > 0`.

### DDP notes

- Enabled when `RANK` env var is set (via `torchrun`)
- `gradient_accumulation_steps` divided by `world_size` per process
- Only rank 0 does logging, checkpointing, `makedirs`
- Grad sync only on last micro-step via `require_backward_grad_sync`

---

## 4. Model Architecture (`model.py`)

### Class hierarchy

```
GPTConfig (dataclass)
GPT
├── transformer.wte          # token embedding  (vocab_size → n_embd)
├── transformer.wpe          # position embedding (block_size → n_embd)
├── transformer.drop
├── transformer.h[i]         # Block × n_layer
│   ├── ln_1 → CausalSelfAttention → residual
│   └── ln_2 → MLP → residual
├── transformer.ln_f
└── lm_head                  # Linear(n_embd → vocab_size), tied to wte
```

### Block internals

**CausalSelfAttention:**
- Single linear `c_attn`: `n_embd → 3*n_embd` (Q, K, V concatenated)
- Multi-head reshape: `(B, T, C) → (B, nh, T, hs)`
- PyTorch ≥2: `scaled_dot_product_attention(..., is_causal=True)` (Flash Attention)
- Fallback: manual masked softmax with lower-triangular buffer

**MLP:** `n_embd → 4*n_embd → n_embd`, GELU, dropout

**Block:** Pre-norm residual: `x + attn(ln_1(x))`, `x + mlp(ln_2(x))`

### `forward(idx, targets=None)`

| Mode | Input | Output |
|---|---|---|
| Training | `idx` (B,T), `targets` (B,T) | logits (B,T,vocab), cross_entropy loss |
| Inference | `idx` (B,T), `targets=None` | logits (B,1,vocab) for last position only, loss=None |

Loss: `F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)`

### Weight tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

Token embedding matrix shared with output projection — fewer params, better generalization.

### `from_pretrained(model_type)`

- Loads HuggingFace `GPT2LMHeadModel`
- Forces `vocab_size=50257`, `block_size=1024`, `bias=True`
- Transposes Conv1D-style weights: `c_attn`, `c_proj`, `c_fc` (HF uses Conv1D, nanoGPT uses Linear)
- Only `dropout` can be overridden

### `generate(idx, max_new_tokens, temperature, top_k)`

Loop `max_new_tokens` times:
1. Crop context to `block_size` if needed
2. Forward → logits at last position
3. Scale by `temperature`
4. Mask to top-k if set
5. Softmax → multinomial sample → append token

---

## 5. Inference (`sample.py`)

### Load path

```
init_from == 'resume'  →  load out_dir/ckpt.pt  →  GPT(GPTConfig(**model_args))
init_from == 'gpt2*'   →  GPT.from_pretrained(init_from)
```

Strips `_orig_mod.` prefix from state dict keys (artifact of `torch.compile`).

### Tokenizer selection

```
resume + meta.pkl exists  →  char-level encode/decode from meta['stoi']/['itos']
else                      →  tiktoken GPT-2 BPE
```

Dataset name comes from `checkpoint['config']['dataset']`.

### Generation parameters

| Param | Effect |
|---|---|
| `start` | Prompt text, or `FILE:path.txt` |
| `num_samples` | Independent generations |
| `max_new_tokens` | Tokens to generate per sample |
| `temperature` | `<1` sharper, `>1` more random |
| `top_k` | Keep only top-k logits before sampling |

---

## 6. Contract Checklist (Safe Edits Reference)

Use this before modifying any component.

### Dataset contract

- [ ] `train.bin` / `val.bin` exist at `data/<dataset>/`
- [ ] Stored as flat `np.uint16` arrays
- [ ] Token IDs in range `[0, vocab_size)`
- [ ] If char-level: `meta.pkl` with `vocab_size`, `stoi`, `itos`
- [ ] `train.py` `dataset` config matches folder name under `data/`

### Model contract

- [ ] `vocab_size` matches data (from `meta.pkl` or 50304/50257 for BPE)
- [ ] `block_size` ≤ sequence length in data (random windows sampled up to `block_size`)
- [ ] On resume: structural args must match checkpoint
- [ ] `from_pretrained`: only `dropout` overridable; vocab/block/bias fixed

### Training contract

- [ ] Batch shapes: `x`, `y` both `(batch_size, block_size)`
- [ ] Target shift: `y[t] = x[t+1]` (next-token prediction)
- [ ] `gradient_accumulation_steps % ddp_world_size == 0` for DDP
- [ ] `out_dir` writable; checkpoint saved as `ckpt.pt`

### Checkpoint contract

- [ ] Required keys: `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `config`
- [ ] `config['dataset']` needed for `sample.py` to find `meta.pkl`
- [ ] Resume loads optimizer state; scratch/gpt2 start fresh

### Inference contract

- [ ] Tokenizer matches training: char `meta.pkl` vs BPE `tiktoken`
- [ ] `out_dir` points to directory containing `ckpt.pt`
- [ ] Prompt encoded with same vocabulary as training data
- [ ] `block_size` in model limits max context during generation

---

## Suggested Reading Order

1. `config/train_shakespeare_char.py` — smallest config, good mental model
2. `data/shakespeare_char/prepare.py` — simplest data pipeline
3. `train.py` → `get_batch()` — how data reaches the model
4. `model.py` → `GPT.forward()` — loss computation
5. `train.py` → training loop + checkpoint save
6. `sample.py` — close the loop with generation
7. `model.py` → `Block`, `CausalSelfAttention`, `generate()` — architecture depth

---

## File Dependency Graph

```
prepare.py ──→ train.bin, val.bin, [meta.pkl]
                    │
config/*.py ──→ configurator.py ──→ train.py ──→ model.py
                    │                    │
                    │                    └──→ out_dir/ckpt.pt
                    │                              │
sample.py ──→ model.py ←─────────────────────────┘
     │
     └──→ meta.pkl (for char decode)
```
