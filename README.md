# Assignment-Metadata-Extractor

A lightweight pipeline for extracting student assignment metadata from unstructured text into a strict JSON format by fine-tuning a SmolLM2-family base model and exporting an Ollama-ready GGUF model.

## Published model

- Fine-tuned checkpoint (360M variant): https://huggingface.co/nimendraai/SmolLM2-360M-Assignment-Metadata-Extractor
- Note: the default training script in this repository currently uses `HuggingFaceTB/SmolLM2-135M-Instruct` as the base model.

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

- Python 3.10 or 3.11
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

## Ollama Modelfile

Create a `Modelfile` with the following content:

```text
FROM hf.co/nimendraai/SmolLM2-360M-Assignment-Metadata-Extractor:Q4_K_M

# Apply the strict instruction template used during training
TEMPLATE """### Instruction:
Extract student info as JSON from the following text.

### Input:
{{ .Prompt }}

### Response:
"""

# Set the System constraints
SYSTEM """
You are a precise student assignment data extractor.
Output ONLY a valid JSON object. No explanation. No extra text. No markdown.
Always output exactly: {"student_number":"...","student_name":"...","assignment_number":"..."}
"""

# Turn off creativity
PARAMETER temperature 0

# Stop generating once the JSON is closed
PARAMETER stop "}"
```

Build and run with Ollama:

```bash
ollama create assignment-metadata-extractor -f Modelfile
ollama run assignment-metadata-extractor
```

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
