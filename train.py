#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
import subprocess
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen2_5_VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration
)
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# Hugging Face token retrieval
# The script will attempt to read the token from the environment variable.
if "HF_TOKEN" not in os.environ:
    print("WARNING: HF_TOKEN environment variable not set.")
    print("Please run `export HF_TOKEN='your_token'` before executing or use `huggingface-cli login`.")


# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# Centralized configuration block for reproducible methodology and fast iteration.
# ==============================================================================
CONFIG = {
    # Model Selection
    # Supported: "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct",
    #            "google/paligemma-3b-mix-448", "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf", 
    
    # Data Paths
    "questions_file": "Objectron/questions_fixed.csv",
    "image_dir": "Objectron/images/",
    "output_dir": "./llava_lora_metric_vqa",
    "eval_output_dir": "./llava_lora_metric_vqa/objectron_eval_results",
    
    # Training Hyperparameters
    "max_length": 2048,             # 1024 for Qwen, 2048 for LLaVA, 1200 for PaliGemma
    "batch_size": 1,                # Real batch size (scale based on VRAM)
    "grad_accum_steps": 4,          # Effective Batch Size = batch_size * grad_accum_steps
    "learning_rate": 1e-4,          # 1e-4 for 3B models, 5e-6 for 7B models
    "epochs": 1,
    
    # LoRA Capacity (Constant across evaluations for controlled comparison)
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
# ==============================================================================


class UnifiedMetricVQADataset(Dataset):
    """
    Generalized dataset handler supporting Qwen-VL, LLaVA, and PaliGemma architectures.
    Dynamically routes formatting based on the processor type and model family.
    """
    def __init__(self, data_df, image_dir, processor, model_id, max_length):
        self.data_df = data_df
        self.image_dir = image_dir
        self.processor = processor
        self.model_id = model_id.lower()
        self.max_length = max_length

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = str(self.data_df.iloc[idx, 0])
        img_path = os.path.join(self.image_dir, img_name)
        
        query = str(self.data_df.iloc[idx, 1])
        answer = str(self.data_df.iloc[idx, 2])

        # ---------------------------------------------------------
        # ARCHITECTURE-SPECIFIC TOKENIZATION & FORMATTING
        # ---------------------------------------------------------
        if "llava" in self.model_id:
            raw_image = Image.open(img_path).convert("RGB")
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            inputs = self.processor(
                images=raw_image,
                text=[text],
                padding="max_length",
                max_length=self.max_length,
                truncation=False,
                return_tensors="pt"
            )
        
        elif "qwen" in self.model_id:
            messages = [
                {"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

        elif "paligemma" in self.model_id:
            raw_image = Image.open(img_path).convert("RGB")
            # PaliGemma requires explicit <image> tag and handles masking internally via 'suffix'
            query_text = "<image> " + query
            
            inputs = self.processor(
                text=query_text,
                images=raw_image,
                suffix=answer,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_id}")

        # ---------------------------------------------------------
        # TENSOR SQUEEZING & DYNAMIC LABEL MASKING
        # ---------------------------------------------------------
        result = {}
        for key, value in inputs.items():
            # Processor returns batched tensors (1, seq_len); squeeze to match Dataset expectation
            result[key] = value.squeeze(0)
            
        # Qwen and LLaVA processors do not auto-generate labels or mask padding tokens.
        # Masking pad tokens with -100 is mathematically required for batch_size > 1 to prevent exploding loss.
        if "labels" not in result:
            labels = result["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            result["labels"] = labels
            
        return result


def data_collator(data):
    """
    Dynamically stacks all tensor keys returned by the dataset __getitem__.
    Supports disparate keys across architectures (e.g., token_type_ids, pixel_values).
    """
    batch = {}
    keys = data[0].keys()
    for key in keys:
        batch[key] = torch.stack([item[key] for item in data])
    return batch


def main():
    print(f"\n{'='*50}\nINITIALIZING UNIFIED VLM TRAINING PIPELINE\n{'='*50}")
    print(f"Target Architecture: {CONFIG['model_id']}\n")

    processor = AutoProcessor.from_pretrained(CONFIG["model_id"], trust_remote_code=True)
    
    # ==========================================================================
    # MODEL ROUTING & INSTANTIATION
    # ==========================================================================
    model_lower = CONFIG["model_id"].lower()
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "attn_implementation": "sdpa"
    }

    if "paligemma" in model_lower:
        model = PaliGemmaForConditionalGeneration.from_pretrained(CONFIG["model_id"], **load_kwargs)
        
        # Empirical standard: Freeze vision tower and projector for PaliGemma fine-tuning
        if hasattr(model, "vision_tower"):
            for param in model.vision_tower.parameters(): param.requires_grad = False
            for param in model.multi_modal_projector.parameters(): param.requires_grad = False
        elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            for param in model.model.vision_tower.parameters(): param.requires_grad = False
            for param in model.model.multi_modal_projector.parameters(): param.requires_grad = False

    elif "llava" in model_lower:
        if "onevision" in model_lower:
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(CONFIG["model_id"], **load_kwargs)
        elif "next" in model_lower:
            model = LlavaNextForConditionalGeneration.from_pretrained(CONFIG["model_id"], **load_kwargs)
        else:
            model = LlavaForConditionalGeneration.from_pretrained(CONFIG["model_id"], **load_kwargs)

    elif "qwen" in model_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(CONFIG["model_id"], **load_kwargs)

    else:
        raise ValueError(f"Model ID {CONFIG['model_id']} not recognized in handler registry.")

    # ==========================================================================
    # LoRA ADAPTATION
    # ==========================================================================
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ==========================================================================
    # DATA PREPARATION & STRATIFIED SPLITTING
    # ==========================================================================
    if not os.path.exists(CONFIG["questions_file"]) or not os.path.exists(CONFIG["image_dir"]):
        print(f"ERROR: Data paths not found. Ensure {CONFIG['questions_file']} and {CONFIG['image_dir']} exist.")
        exit(1)

    full_df = pd.read_csv(CONFIG["questions_file"])
    filename_col = full_df.columns[0]
    full_df['class_name'] = full_df[filename_col].apply(lambda x: str(x).split('__')[0])
    
    # Establish a holdout baseline strictly on unique images to prevent data leakage.
    # Split the data into train/val/test sets (80/10/10%) and stratify by class_name 
    # to maintain proportional class distribution.
    unique_images = full_df.drop_duplicates(subset=[filename_col])[[filename_col, 'class_name']]
    
    train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42, stratify=unique_images['class_name'])
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42, stratify=temp_imgs['class_name'])

    train_df = full_df[full_df[filename_col].isin(train_imgs[filename_col])].reset_index(drop=True)
    val_df = full_df[full_df[filename_col].isin(val_imgs[filename_col])].reset_index(drop=True)
    test_df = full_df[full_df[filename_col].isin(test_imgs[filename_col])].reset_index(drop=True)

    print("\n--- Data Split Verification ---")
    print(f"Total samples: {len(full_df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(full_df):.1%})")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(full_df):.1%})")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(full_df):.1%})")
    
    # Integrity Check
    missing_val_images = sum([1 for img_name in val_df.iloc[:, 0] if not os.path.exists(os.path.join(CONFIG["image_dir"], str(img_name)))])
    if missing_val_images > 0:
        print(f"WARNING: {missing_val_images} missing validation images. Check data pipeline.")
    print("-----------------------------\n")

    # Preserve test set for downstream vlm_eval.py execution
    test_set_path = "test_set.csv"
    eval_test_df = test_df.rename(columns={test_df.columns[0]: "image"})
    eval_test_df[["image", "query", "answer"]].to_csv(test_set_path, index=False)

    train_dataset = UnifiedMetricVQADataset(train_df, CONFIG["image_dir"], processor, CONFIG["model_id"], CONFIG["max_length"])
    val_dataset = UnifiedMetricVQADataset(val_df, CONFIG["image_dir"], processor, CONFIG["model_id"], CONFIG["max_length"])

    # ==========================================================================
    # HUGGING FACE TRAINER INSTANTIATION
    # ==========================================================================
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        bf16=True,
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        gradient_checkpointing=True,
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        report_to="tensorboard",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint:
        print(f"*** Resuming training from checkpoint: {last_checkpoint} ***")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    final_model_path = f"{CONFIG['output_dir']}/best_model"
    trainer.save_model(final_model_path)
    print(f"\nModel and adapter weights successfully saved to {final_model_path}")

    # Flush VRAM before launching the evaluation subprocess
    del trainer
    del model
    torch.cuda.empty_cache()

    # ==========================================================================
    # FINAL EVALUATION INVOCATION
    # ==========================================================================
    print("\n" + "="*50)
    print("      STARTING FINAL EVALUATION ON 10% TEST SET")
    print("="*50 + "\n")

    if os.path.exists(CONFIG["eval_output_dir"]):
        shutil.rmtree(CONFIG["eval_output_dir"])

    eval_command = [
        "python", "vlm_eval.py",
        "--model", CONFIG["model_id"],
        "--csv", test_set_path,
        "--image-dir", CONFIG["image_dir"],
        "--output-dir", CONFIG["eval_output_dir"],
        "--lora_path", final_model_path,
        "--use_lora",
        "--device", "0"
    ]

    try:
        subprocess.run(eval_command, check=True)
        print("Evaluation script completed successfully.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Evaluation script failed with exit code {e.returncode}.")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")


if __name__ == '__main__':
    main()