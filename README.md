# Tiny LLM — a Transformer and Chat UI Built From Scratch

A small, educational GPT-style causal language model implemented from scratch in
**PyTorch** — tokenizer, embeddings, multi-head causal self-attention, and
transformer blocks are all hand-written, not imported from a library — plus a
**FastAPI backend and web chat UI** for talking to it.

> **Important:** this is an educational Tiny LLM, not a production-scale LLM.
> The dataset and model are intentionally very small so the *entire* pipeline —
> pretraining, fine-tuning, and inference — runs comfortably on a laptop CPU.
> See [Current Limitations](#current-limitations) for the honest details.

---

## Live demo

```bash
pip install -e ".[api]"
tiny-llm-serve
```

Then open **http://localhost:8000**. The UI shows:

- A chat window that talks to the instruction-tuned (SFT) checkpoint
- A live **model internals** panel: the forward-pass pipeline, architecture
  stats (params, heads, layers, vocab size), a pretrained/instruction-tuned
  toggle, temperature & max-tokens controls, and a token-level preview of how
  the word-level tokenizer split your last message (including `<UNK>` tokens)

The trained checkpoints in `checkpoints/` are included in the repo, so the demo
works immediately after cloning — no training required.

---

## Project goals

This project demonstrates the core lifecycle of a small language model:

1. Build a Transformer architecture from scratch
2. Implement tokenization
3. Implement token and positional embeddings
4. Implement causal self-attention
5. Implement multi-head attention
6. Build Transformer blocks
7. Train using next-token prediction
8. Save and reload model checkpoints
9. Perform supervised instruction fine-tuning (SFT)
10. Implement response-only loss masking
11. Implement autoregressive generation
12. Diagnose training and generation problems
13. Serve the model behind an API and a web chat UI

Future stages can extend the project with:

14. Preference fine-tuning using DPO
15. Reasoning fine-tuning
16. Better evaluation
17. Improved tokenization and generation

---

## Architecture

```text
Input Text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Token Embeddings + Positional Embeddings
    |
    v
Transformer Block 1
    |
    v
Transformer Block 2
    |
    v
Transformer Block 3
    |
    v
Transformer Block 4
    |
    v
Final LayerNorm
    |
    v
Language Model Head
    |
    v
Logits
    |
    v
Next Token Prediction
```

### Transformer block

```text
Input
  |
  v
LayerNorm
  |
  v
Multi-Head Causal Self-Attention
  |
  v
Residual Connection
  |
  v
LayerNorm
  |
  v
Feed-Forward Network
  |
  v
Residual Connection
  |
  v
Output
```

## Model configuration

| Parameter                     |     Value |
| ------------------------------ | --------: |
| Embedding dimension            |       128 |
| Number of attention heads      |         4 |
| Head dimension                 |        32 |
| Feed-forward hidden dimension  |       512 |
| Number of Transformer blocks   |         4 |
| Maximum sequence length        |       128 |
| Final vocabulary size          |       227 |
| Framework                      |   PyTorch |

All of the above live in one place — `src/tiny_llm/config.py` — instead of
being copy-pasted across training scripts.

---

## Project structure

```text
tiny-llm/
│
├── pyproject.toml            # installable package + console-script entry points
├── requirements.txt          # plain pip alternative to the pyproject extras
│
├── data/
│   ├── train.txt              # pretraining corpus
│   └── instructions.txt       # instruction/response pairs for SFT
│
├── checkpoints/
│   ├── tiny_llm_pretrained.pt
│   ├── tiny_llm_sft.pt
│   └── tokenizer.pt
│
├── src/tiny_llm/              # the installable `tiny_llm` package
│   ├── config.py               # hyperparameters + project paths (single source of truth)
│   ├── tokenizer.py            # SimpleTokenizer (word-level)
│   ├── embeddings.py           # TokenAndPositionEmbedding
│   ├── self_attention.py       # CausalSelfAttention (Phase 1 reference impl.)
│   ├── multi_head_attention.py # MultiHeadCausalSelfAttention (used by the model)
│   ├── transformer_block.py    # TransformerBlock
│   ├── model.py                # TinyLLM
│   ├── dataset.py              # LanguageModelDataset (pretraining)
│   ├── instruction_dataset.py  # InstructionDataset + sft_collate_fn (SFT)
│   ├── checkpoint_utils.py     # shared model/tokenizer loading
│   ├── generation.py           # shared greedy/temperature generation, used by
│   │                            # the CLI scripts AND the API
│   ├── prepare_data.py         # CLI: inspect the pretraining pipeline
│   ├── train.py                # CLI: Phase 2 pretraining
│   ├── train_sft.py            # CLI: Phase 3 instruction fine-tuning
│   ├── generate.py             # CLI: generate one response
│   └── evaluate_sft.py         # CLI: run the fixed evaluation question set
│
├── api/                       # FastAPI backend
│   ├── main.py                  # /api/health, /api/model-info, /api/generate
│   ├── schemas.py                # pydantic request/response models
│   └── serve.py                  # `tiny-llm-serve` entry point
│
├── web/                       # static chat UI served by the API
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── tests/                     # pytest suite (unit tests + checkpoint integration tests)
    ├── conftest.py
    ├── test_tokenizer.py
    ├── test_embeddings.py
    ├── test_attention.py
    ├── test_transformer_block.py
    ├── test_model.py
    ├── test_dataset.py
    ├── test_instruction_dataset.py
    ├── test_generation.py
    └── test_checkpoints.py       # skipped automatically if checkpoints are missing
```

### What changed from the original layout

The original version of this project mixed flat files (`src/train.py`) with
per-component subfolders (`src/embeddings/embeddings.py`), had a dead unused
file (`sft_dataset.py`, superseded by `instruction_dataset.py`), committed
`__pycache__/`, and repeated the same five hyperparameters and the same
checkpoint-loading boilerplate in every script. This version:

- Flattens everything into one importable package, `src/tiny_llm/`
- Centralizes hyperparameters in `config.py` and loading logic in
  `checkpoint_utils.py` / `generation.py`
- Replaces the print-and-eyeball test scripts with real `pytest` assertions
- Adds `api/` + `web/` for a servable demo
- Adds `pyproject.toml` so the whole thing installs as a package with
  console scripts, instead of relying on `python -m src.train`-style relative
  imports

Every model class keeps its original internal attribute names
(`embedding`, `transformer_blocks`, `final_layer_norm`, `lm_head`, …), so the
existing checkpoints in `checkpoints/` still load without retraining.

---

# Phase 1 — Transformer from scratch

The first phase implemented the major Transformer components without using a
pretrained Transformer library.

**Components:** tokenizer, token embeddings, positional embeddings,
self-attention, multi-head attention, feed-forward network, layer
normalization, residual connections, transformer blocks, final LM head.

**Why build it from scratch?** Using a library such as Hugging Face
Transformers would make the project much shorter, but it would hide the
concepts this project is meant to teach:

```text
tokens → embeddings → attention → transformer blocks → logits
```

---

# Phase 2 — Pretraining

The model was pretrained using a small AI-related text corpus on
**next-token prediction**.

```text
Text → Tokenization → Training sequences → Transformer → Logits
     → Cross Entropy Loss → Backpropagation → Optimizer → Updated weights
```

The final pretraining corpus contained ~178 tokens and produced 146 training
sequences. After 20 epochs the loss dropped from ~4.54 to ~0.0433 — the model
learned the training corpus (a low loss here isn't evidence of general
language understanding; the dataset is intentionally tiny).

Saved as `checkpoints/tiny_llm_pretrained.pt` (+ `checkpoints/tokenizer.pt`).

---

# Phase 3 — Supervised fine-tuning (SFT)

The pretrained model was fine-tuned on instruction-response examples, e.g.:

```text
User: What is machine learning?
Assistant: Machine learning is a method where computers learn patterns
from data and use those patterns to make predictions or decisions.
```

Pretraining teaches *"predict the next token."* SFT additionally teaches
*"produce an appropriate response to an instruction."*

### Response-only loss masking

During SFT the input is `user tokens + assistant tokens`, but the model
shouldn't be penalized for failing to predict the user's own prompt:

```text
User target tokens      → -100   (ignored by cross-entropy)
Assistant target tokens → actual token IDs   (learned)
```

### Padding

Examples in a batch are padded to equal length: inputs with `<PAD>` (id `0`),
targets with `-100` so padding never contributes to the loss.

### The 20 → 100 epoch experiment

At 20 epochs, loss decreased (~3.10) but generation quality was poor — the
model produced generic, unrelated responses. Rather than changing several
things at once, the whole pipeline was verified step by step (dataset
creation, tokenization, masking, padding, batching, forward/backward pass,
gradients, checkpointing, generation — all ✓), then epochs were increased
20 → 100 with nothing else changed:

```text
Epoch 1    6.2639
Epoch 20   3.1052
Epoch 50   1.2958
Epoch 75   0.5979
Epoch 100  0.2864
```

Instruction-following improved substantially. **Caveat:** with only 12
instruction examples, the model may have partly memorized them — strong
performance on the training questions isn't evidence of generalization.

Saved as `checkpoints/tiny_llm_sft.pt`.

---

# Tokenizer limitation

The project uses a simple word-level tokenizer. Unknown words become
`<UNK>` — e.g. an unseen question can arrive at the model as
`"What is a <UNK> <UNK>"`. A production LLM would use a subword tokenizer
(BPE or similar). The web UI's token-preview panel makes this failure mode
visible instead of hiding it.

# Generation

Autoregressive, one token at a time:

```text
Prompt → predict next token → append token → predict next token → repeat
```

`src/tiny_llm/generation.py` implements greedy decoding (`temperature=0`, the
original behavior) plus optional temperature/top-k sampling, exposed as a
slider in the web UI. There's no sophisticated stopping mechanism, so the
model may keep generating after a correct answer.

---

# Key concepts demonstrated

Tokenization · vocabulary construction · embeddings · positional information ·
self-attention · Q/K/V · scaled dot-product attention · causal masking ·
multi-head attention · residual connections · layer normalization ·
feed-forward networks · transformer blocks · logits · softmax · cross-entropy
loss · backpropagation · AdamW · next-token prediction · autoregressive
generation · checkpointing · supervised fine-tuning · response-only loss
masking · padding · `<UNK>` handling · training diagnostics · overfitting
considerations · serving a model behind a REST API and web UI.

---

# Running the project

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[api,dev]"        # model + API + tests
# or, without the pyproject extras:
pip install -r requirements.txt
```

### 2. Chat with the included, already-trained model

```bash
tiny-llm-serve
# → open http://localhost:8000
```

### 3. Or use the CLI directly

```bash
tiny-llm-generate "What is machine learning?"
tiny-llm-evaluate                 # runs the fixed 12-question eval set
```

### 4. Retrain from scratch (optional)

```bash
tiny-llm-prepare-data             # sanity-check the data pipeline
tiny-llm-train                    # Phase 2: pretraining
tiny-llm-train-sft                # Phase 3: instruction fine-tuning
```

### 5. Run the tests

```bash
pytest
```

Unit tests (tokenizer, attention causality, model shapes, loss masking,
generation) run with no checkpoint required. `tests/test_checkpoints.py`
additionally exercises the real saved checkpoints and is skipped
automatically if they aren't present.

---

# Current limitations

1. **Tiny dataset** — the training corpus is extremely small.
2. **Simple word-level tokenizer** — unknown words become `<UNK>`.
3. **Small model** — 4 transformer blocks, 128-dim embeddings, 4 heads.
4. **Limited generation** — greedy decoding by default, no repetition
   penalty or beam search.
5. **Limited evaluation** — a 12-question qualitative set, not a scored
   benchmark.
6. **Potential overfitting** — 12 SFT examples is small enough to memorize.

---

# Future roadmap

**Phase 4 — Preference fine-tuning.** Direct Preference Optimization (DPO)
over `(prompt, chosen, rejected)` triples.

**Phase 5 — Reasoning fine-tuning.** Training examples that encourage
`problem → reasoning process → answer`.

**Phase 6 — Evaluation.** Held-out questions, loss curves, exact-match
scoring, generalization tests, generation comparisons.

---

# Interview summary

> I built a small GPT-style causal language model from scratch in PyTorch to
> understand Transformer internals — tokenization, embeddings, causal
> multi-head self-attention, feed-forward networks, residual connections,
> LayerNorm, transformer blocks, and a language-model head. I pretrained it
> with next-token prediction (cross-entropy + AdamW), checkpointed it, then
> ran supervised instruction fine-tuning with a response-only loss mask. I
> debugged the training pipeline by validating tokenization, tensor shapes,
> gradients, and weight updates end to end, then packaged the whole thing as
> an installable Python package with a real test suite, a FastAPI backend,
> and a web chat UI that exposes the model's internals — architecture stats,
> tokenization, and a pretrained-vs-fine-tuned toggle — instead of hiding
> them. The main limitation is that the model and datasets are intentionally
> tiny, so the results demonstrate the *mechanics* of LLM training rather
> than production-level language understanding.
