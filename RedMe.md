# Tiny LLM — Transformer and Language Model From Scratch

A small educational GPT-style causal language model implemented from scratch using **PyTorch**.

The purpose of this project is to understand the internal mechanics of modern Transformer-based language models rather than relying entirely on high-level libraries.

> **Important:** This is an educational Tiny LLM, not a production-scale LLM. The datasets and model are intentionally very small so that the complete training pipeline can run locally.

---

## Project Goals

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

Future stages can extend the project with:

13. Preference fine-tuning using DPO
14. Reasoning fine-tuning
15. Better evaluation
16. Improved tokenization and generation

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

### Transformer Block

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

---

## Model Configuration

| Parameter                     |     Value |
| ----------------------------- | --------: |
| Embedding dimension           |       128 |
| Number of attention heads     |         4 |
| Head dimension                |        32 |
| Feed-forward hidden dimension |       512 |
| Number of Transformer blocks  |         4 |
| Maximum sequence length       |       128 |
| Final vocabulary size         |       227 |
| Training device               | Apple MPS |
| Framework                     |   PyTorch |

---

# Phase 1 — Transformer From Scratch

The first phase implemented the major Transformer components without using a pretrained Transformer model.

### Components

* Tokenizer
* Token embeddings
* Positional embeddings
* Self-attention
* Multi-head attention
* Feed-forward network
* Layer normalization
* Residual connections
* Transformer blocks
* Final language-model head

### Why build it from scratch?

Using a library such as Hugging Face Transformers would make the project much shorter, but it would hide many of the concepts this project is intended to teach.

The goal was therefore to understand:

```text
tokens
  ↓
embeddings
  ↓
attention
  ↓
transformer blocks
  ↓
logits
```

---

# Phase 2 — Pretraining

The model was pretrained using a small AI-related text corpus.

## Objective

The model learns **next-token prediction**.

For example:

```text
Input:
Machine learning learns

Target:
learning learns patterns
```

At each position, the model predicts the next token.

The training pipeline is:

```text
Text
 ↓
Tokenization
 ↓
Training sequences
 ↓
Transformer
 ↓
Logits
 ↓
Cross Entropy Loss
 ↓
Backpropagation
 ↓
Optimizer
 ↓
Updated weights
```

## Dataset

The final pretraining corpus contained approximately:

```text
178 tokens
```

and produced:

```text
146 training sequences
```

## Result

After 20 epochs:

```text
Initial loss: approximately 4.54
Final loss:   0.0433
```

The model successfully learned the training corpus.

> The low loss should not be interpreted as evidence of general language understanding because the dataset is intentionally tiny.

---

# Checkpointing

The pretrained model was saved as:

```text
checkpoints/tiny_llm_pretrained.pt
```

The tokenizer was saved as:

```text
checkpoints/tokenizer.pt
```

The model can subsequently be reconstructed using the same architecture and checkpoint.

---

# Phase 3 — Supervised Fine-Tuning

The pretrained model was then fine-tuned on instruction-response examples.

Example:

```text
User: What is machine learning?

Assistant: Machine learning is a method where computers learn patterns
from data and use those patterns to make predictions or decisions.
```

## Objective

Pretraining teaches:

> Predict the next token.

SFT additionally teaches:

> Produce an appropriate response to an instruction.

---

# Response-Only Loss Masking

During SFT, the input consists of:

```text
User tokens + Assistant tokens
```

However, we do not want the model to be penalized for failing to predict the user's prompt.

Therefore:

```text
User target tokens      → -100
Assistant target tokens → actual token IDs
```

PyTorch's cross-entropy loss ignores targets with value:

```text
-100
```

Conceptually:

```text
User:
What is machine learning?

        ↓ ignored

Assistant:
Machine learning is a method...

        ↓ learned
```

This allows the model to focus the training signal on generating the assistant response.

---

# Padding

Instruction responses have different lengths.

Therefore examples in the same batch are padded to the same length.

Input padding uses:

```text
<PAD> = 0
```

Target padding uses:

```text
-100
```

so padding does not contribute to the training loss.

---

# SFT Training Experiment

The first SFT experiment used:

```text
20 epochs
```

The loss decreased to approximately:

```text
3.10
```

However, generation quality was poor.

The model frequently produced generic responses that were not related to the question.

Example behavior:

```text
Question:
What is machine learning?

Generated:
A neural network contains layers...
Transformers use attention...
```

---

# Debugging the SFT Pipeline

Instead of changing multiple components simultaneously, the training pipeline was tested step-by-step.

Verified:

```text
Dataset creation              ✓
Tokenization                  ✓
Response masking              ✓
Padding                       ✓
Batch creation                ✓
Forward pass                  ✓
Logit dimensions              ✓
Cross-entropy loss            ✓
Backward pass                 ✓
Gradient calculation          ✓
Weight update                 ✓
Checkpoint loading            ✓
Generation                    ✓
```

Example batch:

```text
Input shape:
[4, 26]

Target shape:
[4, 26]
```

Model output:

```text
[4, 26, 227]
```

---

# 100-Epoch SFT Experiment

To determine whether the model simply needed more optimization, the number of epochs was changed from:

```text
20 → 100
```

No other major training configuration was changed.

The loss progressed approximately as follows:

```text
Epoch 1    6.2639
Epoch 20   3.1052
Epoch 50   1.2958
Epoch 75   0.5979
Epoch 100  0.2864
```

Instruction-following behavior improved substantially.

The model successfully generated appropriate answers for the 12 instruction examples.

---

# Important Observation: Training Loss vs Behavior

A decreasing training loss does not automatically guarantee good instruction following.

The 20-epoch model had a decreasing loss but poor generation behavior.

After additional optimization, the 100-epoch model produced much better answers.

However, because the dataset contains only 12 instruction examples, the model may have memorized the training examples.

Therefore:

> Strong performance on the training questions should not be interpreted as strong generalization.

---

# Tokenizer Limitation

The project currently uses a simple word-level tokenizer.

Unknown words are represented as:

```text
<UNK>
```

During an experiment with an unseen question, the model received:

```text
What is a <UNK> <UNK>
```

This demonstrated a major limitation of the toy tokenizer.

A production LLM would generally use a subword tokenizer such as BPE or a related tokenizer.

---

# Generation

The model generates text autoregressively.

```text
Prompt
 ↓
Predict next token
 ↓
Append token
 ↓
Predict next token
 ↓
Append token
 ↓
Repeat
```

The current implementation uses greedy decoding:

```text
Select the highest-probability next token.
```

Generation is limited using a maximum number of new tokens.

Because this educational implementation does not have a sophisticated stopping mechanism, the model may sometimes continue generating after producing a correct answer.

---

# Key Concepts Demonstrated

This project demonstrates:

* Tokenization
* Vocabulary construction
* Embeddings
* Positional information
* Self-attention
* Query / Key / Value
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* Residual connections
* Layer normalization
* Feed-forward networks
* Transformer blocks
* Logits
* Softmax
* Cross-entropy loss
* Backpropagation
* AdamW optimization
* Next-token prediction
* Autoregressive generation
* Model checkpointing
* Supervised fine-tuning
* Response-only loss masking
* Padding
* `<UNK>` handling
* Training diagnostics
* Overfitting considerations

---

# Project Structure

```text
Tiny LLM/
│
├── data/
│   ├── train.txt
│   └── instructions.txt
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── self_attention.py
│   ├── multi_head_attention.py
│   ├── transformer_block.py
│   ├── tiny_llm.py
│   ├── prepare_data.py
│   ├── test_training_step.py
│   ├── train.py
│   ├── test_checkpoint.py
│   ├── test_shared_tokenizer.py
│   ├── test_sft_tokenization.py
│   ├── test_sft_dataset.py
│   ├── instruction_dataset.py
│   ├── test_instruction_dataset.py
│   ├── sft_collate.py
│   ├── test_sft_batch.py
│   ├── test_sft_mask.py
│   ├── test_sft_training_step.py
│   ├── train_sft.py
│   ├── generate.py
│   ├── evaluate_sft.py
│   └── test_sft_predictions.py
│
├── checkpoints/
│   ├── tiny_llm_pretrained.pt
│   ├── tiny_llm_sft.pt
│   └── tokenizer.pt
│
└── README.md
```

---

# Running the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch:

```bash
pip install torch
```

Run the pretraining pipeline:

```bash
python3 -m src.train
```

Test the checkpoint:

```bash
python3 -m src.test_checkpoint
```

Run SFT:

```bash
python3 -m src.train_sft
```

Evaluate the SFT model:

```bash
python3 -m src.evaluate_sft
```

Generate a response:

```bash
python3 -m src.generate
```

---

# Current Limitations

This project intentionally has significant limitations.

### 1. Tiny dataset

The training corpus is extremely small.

### 2. Simple word-level tokenizer

Unknown words become `<UNK>`.

### 3. Small model

The model has only:

```text
4 Transformer blocks
128-dimensional embeddings
4 attention heads
```

### 4. Limited generation

The current implementation uses simple greedy decoding.

### 5. Limited evaluation

The evaluation set is very small and is not sufficient to measure real-world language-model quality.

### 6. Potential overfitting

The SFT dataset contains only 12 examples, so the model can memorize the training examples.

---

# Future Roadmap

## Phase 4 — Preference Fine-Tuning

Implement Direct Preference Optimization (DPO).

Training examples will contain:

```text
Prompt
Chosen response
Rejected response
```

The objective will teach the model to prefer the chosen response.

---

## Phase 5 — Reasoning Fine-Tuning

Explore training examples that encourage:

```text
Problem
 ↓
Reasoning process
 ↓
Answer
```

The goal will be to study how additional fine-tuning can affect problem-solving behavior.

---

## Phase 6 — Evaluation

Add more systematic evaluation:

* held-out questions
* loss curves
* exact-match evaluation
* response quality checks
* generalization tests
* generation comparisons

---

# Interview Summary

A concise project explanation:

> I built a small GPT-style causal language model from scratch in PyTorch to understand Transformer internals. I implemented tokenization, embeddings, causal multi-head self-attention, feed-forward networks, residual connections, LayerNorm, Transformer blocks and a language-model head. I pretrained it using next-token prediction with cross-entropy loss and AdamW, saved the checkpoint, and then performed supervised instruction fine-tuning using a response-only loss mask. I debugged the training pipeline by validating tokenization, tensor shapes, gradients and weight updates. The main limitation is that the model and datasets are intentionally tiny, so the results demonstrate the mechanics of LLM training rather than production-level language understanding.
