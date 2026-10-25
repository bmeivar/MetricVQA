#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    Qwen2_5_VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig
import json

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


# =============================================================================
# Per-model LoRA target modules (verified by architecture inspection)
# =============================================================================

LORA_TARGETS = {
    "qwen": {
        "all": ["qkv", "proj", "gate_proj", "up_proj", "down_proj",
                "q_proj", "k_proj", "v_proj", "o_proj"],
        "language_only": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    },
    "llava": {
        "all": ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                "gate_proj", "up_proj", "down_proj", "fc1", "fc2"],
        "language_only": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    },
    "paligemma": {
        "all": ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                "gate_proj", "up_proj", "down_proj", "fc1", "fc2"],
        "language_only": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    },
    "gemma3": {
        "all": ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                "gate_proj", "up_proj", "down_proj", "fc1", "fc2"],
        "language_only": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    },
}

MAX_LENGTH_DEFAULTS = {"qwen": 1024, "llava": 8192, "paligemma": 1200, "gemma3": 1024}


def get_model_family(model_id: str) -> str:
    lower = model_id.lower()
    if "llava" in lower:
        return "llava"
    if "qwen" in lower:
        return "qwen"
    if "paligemma" in lower:
        return "paligemma"
    if "gemma-3" in lower or "gemma3" in lower:
        return "gemma3"
    raise ValueError(f"Unsupported model: {model_id}")


# =============================================================================
# Dataset
# =============================================================================

class MetricVQADataset(Dataset):
    """Returns raw samples; collator handles tokenization."""
    def __init__(self, data_df, image_dir):
        self.data_df = data_df
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["image"]))
        return {
            "image": Image.open(img_path).convert("RGB"),
            "query": str(row["query"]),
            "answer": str(row["answer"]),
        }


class MetricVQACollator:
    """Batch collator that tokenizes with dynamic padding."""
    def __init__(self, processor, model_family):
        self.processor = processor
        self.model_family = model_family

    def __call__(self, examples):
        messages_list = []
        images = []

        for ex in examples:
            img = ex["image"]
            query = ex["query"]
            answer = ex["answer"]

            if self.model_family == "paligemma":
                messages_list.append({"query": query, "answer": answer})
                images.append(img)
                continue

            if self.model_family == "qwen":
                msgs = [
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": query}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            else:
                msgs = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            messages_list.append(msgs)
            images.append(img)

        if self.model_family == "paligemma":
            batch = self.processor(
                text=["<image> " + m["query"] for m in messages_list],
                images=images,
                suffix=[m["answer"] for m in messages_list],
                padding=True, return_tensors="pt",
            )
        elif self.model_family == "qwen":
            texts = [self.processor.apply_chat_template(m, tokenize=False) for m in messages_list]
            image_inputs = [process_vision_info(m)[0] for m in messages_list]
            batch = self.processor(
                text=texts, images=image_inputs,
                padding=True, return_tensors="pt",
            )
        else:
            texts = [self.processor.apply_chat_template(m, tokenize=False) for m in messages_list]
            batch = self.processor(
                text=texts, images=images,
                padding=True, return_tensors="pt",
            )

        if "labels" not in batch:
            labels = batch["input_ids"].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            # Mask image token IDs explicitly
            if self.model_family == "qwen":
                for tok_id in [151652, 151653, 151655]:
                    labels[labels == tok_id] = -100
            else:
                img_tok_id = self.processor.tokenizer.convert_tokens_to_ids(
                    self.processor.image_token
                )
                labels[labels == img_tok_id] = -100

            # Mask prompt tokens — process prompt WITH images to get correct lengths
            prompt_examples = []
            for ex in examples:
                if self.model_family == "qwen":
                    msgs = [
                        {"role": "user", "content": [{"type": "image", "image": ex["image"]}, {"type": "text", "text": ex["query"]}]},
                    ]
                else:
                    msgs = [
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["query"]}]},
                    ]
                prompt_examples.append(msgs)

            prompt_texts = [
                self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in prompt_examples
            ]

            if self.model_family == "qwen":
                prompt_images = [process_vision_info(m)[0] for m in prompt_examples]
            else:
                prompt_images = [ex["image"] for ex in examples]

            prompt_batch = self.processor(
                text=prompt_texts, images=prompt_images,
                padding=True, return_tensors="pt",
            )
            prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()

            seq_len = labels.size(1)
            for i, plen in enumerate(prompt_lens):
                plen = min(int(plen), seq_len)
                labels[i, :plen] = -100

            batch["labels"] = labels

        return batch


# =============================================================================
# Data loading & stratified splitting
# =============================================================================

def load_objectron_data(dataset_arg, val_size=500):
    """Objectron questions as a DataFrame with columns image, query, answer.

    dataset_arg is a local CSV or an HF dataset repo. Two layouts are read: the
    public release (bmeivar/MetricVQA: objectron/questions.csv with columns
    image, question, answer_cm; images not redistributed) and the original
    bmeivar/metric-vqa-objectron (questions_objectron.csv + images/). Rows are
    identical and in the same order in both, so the seed-42 split is the same.
    """
    if os.path.isfile(dataset_arg):
        csv_path = dataset_arg
    else:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(dataset_arg, repo_type="dataset",
                                      allow_patterns=["objectron/*", "questions_objectron.csv", "images/*"])
        csv_path = os.path.join(local_dir, "objectron", "questions.csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(local_dir, "questions_objectron.csv")

    df = pd.read_csv(csv_path)
    if "answer_cm" in df.columns:  # release schema
        df = pd.DataFrame({
            "image": df["image"].str.replace("images/", "", regex=False),
            "query": df["question"],
            "answer": df["answer_cm"],
        })
    else:
        df = df[["image", "query", "answer"]].copy()
    df["category"] = df["image"].apply(lambda x: str(x).split("__")[0])

    unique_images = df.drop_duplicates(subset=["image"])[["image", "category"]]
    n_images = len(unique_images)
    questions_per_image = len(df) / n_images
    val_images_needed = int(val_size / questions_per_image)

    train_imgs, val_imgs = train_test_split(
        unique_images, test_size=val_images_needed,
        random_state=42, stratify=unique_images["category"],
    )

    train_df = df[df["image"].isin(train_imgs["image"])].reset_index(drop=True)
    val_df = df[df["image"].isin(val_imgs["image"])].reset_index(drop=True)

    print(f"Dataset: {csv_path}")
    print(f"Total: {len(df)} questions, {n_images} images, 9 categories")
    print(f"Train: {len(train_df)} questions ({len(train_imgs)} images)")
    print(f"Val:   {len(val_df)} questions ({len(val_imgs)} images)")
    print(f"Val category distribution: {val_df['category'].value_counts().to_dict()}")

    image_dir = os.path.dirname(csv_path)
    if os.path.isdir(os.path.join(image_dir, "images")):
        image_dir = os.path.join(image_dir, "images")

    return train_df, val_df, image_dir


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for metric VQA")
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", default="bmeivar/MetricVQA",
                        help="HF dataset ID or path to local CSV")
    parser.add_argument("--image_dir", default=None,
                        help="Directory of Objectron frames (bike__0000.png, ...). Required for "
                             "bmeivar/MetricVQA, which does not redistribute them; see README.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for LoRA adapters")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max sequence length (auto per model if not set)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", choices=["all", "language_only"], default="all",
                        help="'all' = vision+language, 'language_only' = language only")
    parser.add_argument("--val_size", type=int, default=500,
                        help="Approximate number of validation questions")
    args = parser.parse_args()

    family = get_model_family(args.model_id)
    max_length = args.max_length or MAX_LENGTH_DEFAULTS[family]
    output_dir = args.output_dir or f"./lora_{args.model_id.split('/')[-1]}"

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-Tuning: {args.model_id}")
    print(f"  Target: {args.lora_target} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # --- Data ---
    train_df, val_df, auto_image_dir = load_objectron_data(args.dataset, args.val_size)
    image_dir = args.image_dir or auto_image_dir
    first = os.path.join(image_dir, str(train_df["image"].iloc[0]))
    if not os.path.isfile(first):
        raise SystemExit(f"Objectron image not found: {first}\nPass --image_dir <dir of Objectron frames>.")

    # --- Model ---
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": int(os.environ.get("LOCAL_RANK", 0))},
        "attn_implementation": "sdpa",
    }

    if family == "paligemma":
        model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    elif family == "qwen":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, trust_remote_code=True, **load_kwargs)

    # --- LoRA ---
    target_modules = LORA_TARGETS[family][args.lora_target]
    print(f"LoRA target modules ({args.lora_target}): {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # --- Datasets ---
    train_dataset = MetricVQADataset(train_df, image_dir)
    val_dataset = MetricVQADataset(val_df, image_dir)
    collator = MetricVQACollator(processor, family)

    # --- Trainer ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    # --- Logging callback ---
    log_file = os.path.join(output_dir, "train_log.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    class JSONLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                entry = {"step": state.global_step, "epoch": state.epoch, **logs}
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[JSONLogCallback()],
    )

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save last model (current state after final step)
    trainer.save_model(f"{output_dir}/last_model")
    print(f"\nLast model saved to {output_dir}/last_model")

    # Save best model (lowest eval_loss checkpoint)
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        import shutil
        shutil.copytree(best_ckpt, f"{output_dir}/best_model", dirs_exist_ok=True)
        print(f"Best model saved to {output_dir}/best_model (from {best_ckpt})")
    else:
        trainer.save_model(f"{output_dir}/best_model")
        print(f"Best model saved to {output_dir}/best_model (same as last)")


if __name__ == "__main__":
    main()
