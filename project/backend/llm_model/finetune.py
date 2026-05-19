"""
finetune.py
-----------
Fine-tunes TinyLlama on the merged Marathi math + Pythagoras dataset
using HuggingFace Transformers + PEFT (LoRA).

Works on:
  - GPU  → 4-bit quantization (bitsandbytes), fp16 training
  - CPU  → float32, no quantization (slower, ~2-4 hrs for full dataset)

Usage:
    cd project
    python backend/llm_model/finetune.py

    # Faster run with fewer samples (for testing):
    python backend/llm_model/finetune.py --max_samples 200 --epochs 2

Output:
    project/backend/llm_model/marathi_tutor_lora/
"""

import os, sys, json, argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(_HERE, "..", "..", "dataset")
MATH_JSONL   = os.path.join(DATASET_DIR, "marathi_math_dataset.jsonl")
PYTH_JSONL   = os.path.join(DATASET_DIR, "pythagoras_dataset.jsonl")
OUTPUT_DIR   = os.path.join(_HERE, "marathi_tutor_lora")
MERGED_PATH  = os.path.join(DATASET_DIR, "merged_dataset.jsonl")

# ── Model ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # ~600 MB, CPU-friendly

# ── System prompt (injected at training time) ──────────────────────────────────
SYSTEM_PROMPT = (
    "तुम्ही एक मराठी गणित शिक्षक आहात जे दृष्टिहीन विद्यार्थ्यांना "
    "अंकगणित आणि पायथागोरस प्रमेय शिकवता. "
    "नेहमी मराठीत उत्तर द्या. स्पष्ट, सोप्या भाषेत समजावून सांगा."
)


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        print(f"[WARN] Dataset not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_datasets(max_samples: int | None = None) -> list[dict]:
    """
    Merge math + Pythagoras datasets.
    Normalise keys: instruction / response → instruction / output
    """
    math_rows  = load_jsonl(MATH_JSONL)
    pyth_rows  = load_jsonl(PYTH_JSONL)

    merged = []
    for row in math_rows + pyth_rows:
        instruction = row.get("instruction", "").strip()
        output      = row.get("response", row.get("output", "")).strip()
        if instruction and output:
            merged.append({"instruction": instruction, "input": "", "output": output})

    if max_samples:
        import random; random.shuffle(merged)
        merged = merged[:max_samples]

    # Save merged file for reference
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DATA] Merged {len(merged)} samples → {MERGED_PATH}")
    return merged


def format_prompt(sample: dict) -> str:
    """
    Alpaca-style prompt with system context.
    Format used at both training and inference time.
    """
    instruction = sample["instruction"]
    inp         = sample.get("input", "").strip()
    output      = sample.get("output", "")

    if inp:
        user_part = f"{instruction}\n{inp}"
    else:
        user_part = instruction

    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_part}</s>\n"
        f"<|assistant|>\n{output}</s>"
    )


def format_prompt_inference(instruction: str, inp: str = "") -> str:
    """Prompt without the output — used at inference time."""
    user_part = f"{instruction}\n{inp}".strip() if inp else instruction
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_part}</s>\n"
        f"<|assistant|>\n"
    )


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_batch(batch: dict, tokenizer, max_length: int) -> dict:
    texts = [
        format_prompt({"instruction": i, "input": inp, "output": o})
        for i, inp, o in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # Mask padding tokens in labels so loss is not computed on them
    labels = []
    for ids, attn in zip(tokens["input_ids"], tokens["attention_mask"]):
        label = [id_ if attn[i] == 1 else -100 for i, id_ in enumerate(ids)]
        labels.append(label)
    tokens["labels"] = labels
    return tokens


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (useful for quick tests)")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--max_length",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lora_r",      type=int, default=8)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device  = "cuda" if use_gpu else "cpu"
    print(f"[INFO] Device: {device}")
    if use_gpu:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    raw = merge_datasets(args.max_samples)
    dataset = Dataset.from_list(raw)
    print(f"[DATA] {len(dataset)} training samples")

    # ── 2. Tokenizer ───────────────────────────────────────────────────────
    print(f"[MODEL] Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 3. Base model ──────────────────────────────────────────────────────
    print(f"[MODEL] Loading base model: {BASE_MODEL}")
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("[INFO] CPU mode — this will be slow. Consider --max_samples 500 for a quick test.")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    model.config.use_cache = False   # required for gradient checkpointing

    # ── 4. LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Tokenise dataset ────────────────────────────────────────────────
    print("[DATA] Tokenising...")
    tokenized = dataset.map(
        lambda b: tokenize_batch(b, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenising",
    )

    # ── 6. Training args ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        warmup_ratio=0.05,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=use_gpu,
        bf16=False,
        report_to="none",
        dataloader_pin_memory=use_gpu,
        optim="adamw_torch",
        gradient_checkpointing=use_gpu,
        remove_unused_columns=False,
    )

    # ── 7. Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[TRAIN] Starting fine-tuning...")
    trainer.train()

    # ── 8. Save ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save prompt format alongside model so inference knows how to format
    meta = {
        "base_model": BASE_MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_format": "tinyllama-chat",
        "max_new_tokens": 300,
    }
    with open(os.path.join(OUTPUT_DIR, "tutor_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Model saved to: {OUTPUT_DIR}")
    print("       Run inference with: python backend/llm_model/inference.py")


if __name__ == "__main__":
    main()
