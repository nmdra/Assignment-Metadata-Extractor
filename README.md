# Assignment-Metadata-Extractor

A lightweight pipeline for extracting student assignment metadata from unstructured text into a strict JSON format by fine-tuning `HuggingFaceTB/SmolLM2-135M-Instruct` and exporting an Ollama-ready GGUF model.

## Published model

- Hugging Face: https://huggingface.co/nimendraai/SmolLM2-360M-Assignment-Metadata-Extractor

## What this repository contains

- `data/generate_dataset.py`  
  Generates synthetic instruction-tuning examples.
- `training/train.py`  
  Fine-tunes `HuggingFaceTB/SmolLM2-135M-Instruct` using Unsloth + LoRA, then exports HF and GGUF artifacts.
- `training/train.ipynb`  
  Notebook version of the same training workflow.

## Tech stack

- Python 3.10–3.11
- [uv](https://docs.astral.sh/uv/) for environment + dependency management
- PyTorch, Hugging Face Datasets, TRL (`SFTTrainer`)
- Unsloth for efficient LoRA fine-tuning and GGUF export

## Prerequisites

- Python `>=3.10,<3.12`
- `uv` installed
- Recommended for training: NVIDIA GPU with CUDA (CPU training is possible but slow)

## Quickstart

### 1) Set up the environment

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
uv sync
```

### 2) Generate synthetic dataset

```bash
uv run python data/generate_dataset.py --size 400 --output data/dataset.json
```

### 3) Run fine-tuning

```bash
uv run python training/train.py
```

### 4) Produced artifacts

- `./smollm-student-extractor/` (Hugging Face model/tokenizer)
- `./smollm-student-gguf/` (GGUF export for Ollama)

## Dataset format

`data/generate_dataset.py` creates a JSON list where each item contains:

- `instruction`
- `input`
- `output` (JSON string with keys: `student_number`, `student_name`, `assignment_number`)

## Notebook workflow

```bash
uv run python -m ipykernel install --user --name assignment-metadata-extractor
uv run jupyter notebook training/train.ipynb
```

## Notes

- `training/train.py` expects `data/dataset.json` to exist.
- If the dataset file is missing or empty, training exits with a clear error.
