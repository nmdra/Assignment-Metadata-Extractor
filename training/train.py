import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_PATH = Path("data/dataset.json")
HF_OUTPUT_DIR = "./smollm-student-extractor"
GGUF_OUTPUT_DIR = "smollm-student-gguf"
MAX_SEQ_LENGTH = 512


def load_dataset(path: Path) -> Dataset:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not raw:
        raise ValueError("Dataset is empty. Add examples to data/dataset.json before training.")

    def format_example(ex: dict) -> dict:
        return {
            "text": (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}"
            )
        }

    return Dataset.from_list(raw).map(format_example, remove_columns=list(raw[0].keys()))


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `uv run python data/generate_dataset.py` first."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset(DATASET_PATH)
    cuda_available = torch.cuda.is_available()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=cuda_available and not torch.cuda.is_bf16_supported(),
            bf16=cuda_available and torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="./outputs",
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(HF_OUTPUT_DIR)
    tokenizer.save_pretrained(HF_OUTPUT_DIR)

    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print("Done — GGUF ready for Ollama.")


if __name__ == "__main__":
    main()
