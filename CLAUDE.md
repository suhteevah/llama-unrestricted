# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repo is **pre-scaffold**. The only file currently present is `QWEN CI FINETUNE SPEC.pdf`, the authoritative spec for the project. There is no source code, no build system, no git history, and no tests yet. Future Claude Code sessions will be asked to scaffold and implement the project described in that spec — read it first before making changes.

Intended project root name (per spec §8): `brander-ci-agent/`. The current directory `J:\llama-unrestricted` is the workspace where scaffolding happens.

## What this project is

Fine-tune **Qwen2.5-7B-Instruct** into a competitive-intelligence / executive-headhunting research agent, then export it as a **GGUF** file runnable via `ollama run` or `llama.cpp`. The whole pipeline (synthetic data generation → curation → LoRA fine-tune → merge → GGUF quantize → Ollama Modelfile) is implemented in this repo.

## Big-picture architecture (from spec)

Four stages, each a separate script under `scripts/`:

1. **Synthetic data generator** (`generate_training_data.py`) — calls the Claude API with hand-written seed examples as few-shot prompts, generating ChatML-format training examples across 7 task categories (exec profiling, company intel, candidate sourcing, competitive landscape, opposition research, tool-use traces, refusal calibration). Target ~20k examples total.
2. **Dataset curator/validator** (`validate_data.py`, `balance_dataset.py`) — schema validation, dedup, quality scoring (reject <0.7), category balancing to target distribution, then train/eval/test split.
3. **Fine-tune runner** (`train.py`) — **unsloth** (preferred for 2× speed / 60% less VRAM) LoRA training with r=64, alpha=128, dropout=0.05, all attention + MLP projection targets. Cosine schedule, lr=2e-5, warmup 100, bf16, gradient checkpointing, effective batch size 32 (4 × 8 grad accum), max_seq_length=4096, 8–10 epochs with early stop. W&B project `brander-ci-agent`. Eval every 250 steps on 5% holdout.
4. **Post-training** (`merge_lora.py`, `export_gguf.py`) — merge LoRA into base → `convert_hf_to_gguf.py` → `llama-quantize` to **Q4_K_M, Q5_K_M, Q8_0** → write `Modelfile` → `ollama create ci-agent`.
5. **Eval harness** (`eval.py`) — tool-call accuracy (>90%), dossier completeness (>80%), refusal accuracy (>95%), candidate ranking Kendall tau >0.7, hallucination rate <5%, latency <30s on Q4_K_M.

### Data format

All training examples are **ChatML** JSONL with `messages: [...]` containing `system`, `user`, `assistant` (possibly with `tool_calls`), and `tool` roles. Tool-use traces interleave assistant tool calls with tool results and a final synthesized assistant message. Schema for each task category (especially exec profiling) is defined in spec §3.

### Tool definitions

The set of tools the model is trained to call is fixed and lives in `configs/tool_definitions.json`: `web_search`, `web_fetch`, `sec_filing_search`, `linkedin_search`, `patent_search`, `court_records_search`, `property_records`, `corporate_registry`, `news_search`, `github_search`, `crunchbase_search`, `save_to_dossier`. All scope to **public** data sources — this is a hard constraint.

### Refusal calibration boundary

This is deliberate, not an afterthought. The model is trained to **comply** with legitimate professional research (exec backgrounds from public filings, comp bands, org charts, litigation history) and invasive personal requests (home addresses, kids' schools, medical records, private accounts, deepfakes, stalking). Spec §3.7 has the canonical comply/refuse examples — mirror that framing in seed examples and system prompts. `configs/system_prompts.yaml` is where this is reinforced at inference time.

## Target directory layout (spec §8)

```
brander-ci-agent/
├── data/
│   ├── seeds/         # hand-written gold examples, one JSONL per category
│   ├── generated/     # synthetic output from Claude API
│   ├── curated/       # post-validation dataset
│   └── splits/        # train/eval/test
├── scripts/           # generate_training_data.py, validate_data.py,
│                      # balance_dataset.py, train.py, merge_lora.py,
│                      # export_gguf.py, eval.py
├── configs/           # training_config.yaml, tool_definitions.json,
│                      # system_prompts.yaml
├── evals/             # fixtures/, results/
├── Modelfile
└── requirements.txt
```

Spec §9 gives the canonical execution order when building from scratch: scaffold → tool defs → system prompts → seed examples → generator → validator → generate dataset → train script → train → eval → merge+GGUF → ollama test.

## Commands (once scripts exist)

Nothing is runnable yet. The commands below are what the spec prescribes — use them verbatim once the corresponding scripts have been written.

```bash
# Environment (on target GPU box)
pip install -r requirements.txt
# spec lists: unsloth transformers datasets peft trl wandb  (axolotl is an alternative)

# Data pipeline
python scripts/generate_training_data.py --category exec_profiling --count 3000
python scripts/validate_data.py  data/generated/  data/curated/
python scripts/balance_dataset.py data/curated/   data/splits/

# Training
python scripts/train.py --config configs/training_config.yaml

# Post-training export
python scripts/merge_lora.py --base Qwen/Qwen2.5-7B-Instruct --lora ./output/checkpoint-best
python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile ci-agent-7b.gguf
./llama.cpp/build/bin/llama-quantize ci-agent-7b.gguf ci-agent-7b-Q4_K_M.gguf Q4_K_M
./llama.cpp/build/bin/llama-quantize ci-agent-7b.gguf ci-agent-7b-Q5_K_M.gguf Q5_K_M
./llama.cpp/build/bin/llama-quantize ci-agent-7b.gguf ci-agent-7b-Q8_0.gguf Q8_0

# Ollama
ollama create ci-agent ./Modelfile
ollama run ci-agent

# Evals
python scripts/eval.py --benchmark tool_call_accuracy
python scripts/eval.py --benchmark refusal_accuracy
```

## Hardware assumptions

- **Primary dev box (`kokonoe`)**: i9-11900K, RTX 3070 Ti **8 GB** — LoRA fits but only with unsloth + gradient checkpointing + bf16. Full fine-tune will not fit; don't propose it.
- **Fallback**: P40 24 GB for comfort.
- Training run budget: ~8–12 hours on the 3070 Ti. Synthetic data gen: ~4–6 hours of Claude API calls, ~$30–50 on Sonnet.

If hyperparameters change, check they still fit 8 GB before suggesting the change.

## Environment notes

- **Platform is Windows** with a bash shell (Git Bash / MSYS2). Use forward-slash paths and Unix syntax in commands. `pdftotext` (poppler) is available at `/mingw64/bin/pdftotext` if you need to re-read the spec PDF — `pdftoppm` is **not** installed, so the Read tool's PDF image mode will fail; use `pdftotext -layout` via Bash instead.
- CUDA 12.6 is on PATH. Python 3.10, 3.13, and 3.14 are all installed; pick one deliberately when creating the venv (unsloth compatibility is the constraint — check unsloth's current Python support before picking).
- Ollama is installed locally (`~/AppData/Local/Programs/Ollama`).
