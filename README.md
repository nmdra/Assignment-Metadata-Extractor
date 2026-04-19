# Assignment-Metadata-Extractor

A lightweight, locally hosted LLM pipeline that extracts and normalizes unstructured student assignment text into a strict JSON schema using a fine-tuned SmolLM2 model served via Ollama.

## Model

Hugging Face model: https://huggingface.co/nimendraai/SmolLM2-360M-Assignment-Metadata-Extractor

## Training (Part 1)

This repository includes:

- `data/generate_dataset.py`: Generates a diverse synthetic training dataset.
- `training/train.py`: Fine-tunes `HuggingFaceTB/SmolLM2-135M-Instruct` with Unsloth + LoRA and exports GGUF.
- `training/train.ipynb`: Jupyter notebook version of the same training pipeline.

## Environment Setup (UV)

This project uses **uv** for package management.

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell

uv sync
```

If you want to run the notebook locally:

```bash
uv run python -m ipykernel install --user --name assignment-metadata-extractor
```

## Generate Dataset

```bash
uv run python data/generate_dataset.py --size 400 --output data/dataset.json
```

## Run Fine-Tuning Script

```bash
uv run python training/train.py
```

Outputs:

- `./smollm-student-extractor/` (HuggingFace format)
- `./smollm-student-gguf/model-Q4_K_M.gguf` (Ollama-ready GGUF)

## Run Notebook Version

```bash
uv run jupyter notebook training/train.ipynb
```
