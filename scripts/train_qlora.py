from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """识别图片中所有可以可靠确认的食材。
只输出符合 VisionResult schema 的 JSON，字段只包含 name、normalized_name 和 confidence。
不要识别日期、数量、包装规格或非食材；无法确认的物体不要猜测。"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud QLoRA training for Qwen2.5-VL-3B")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen-vl-fridge-lora"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quantization, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    model.add_adapter(
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
    )
    processor = AutoProcessor.from_pretrained(args.model)
    dataset = load_dataset(
        "json", data_files={"train": str(args.train), "validation": str(args.eval)}
    )

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        messages_batch = []
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            target = json.dumps(row["target"], ensure_ascii=False)
            messages_batch.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "识别并输出 JSON。"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ]
            )
        texts = [processor.apply_chat_template(item, tokenize=False) for item in messages_batch]
        image_inputs, video_inputs = zip(
            *(process_vision_info(item) for item in messages_batch), strict=True
        )
        images = [image for group in image_inputs for image in group]
        videos = [video for group in video_inputs for video in group]
        batch = processor(
            text=texts,
            images=images,
            videos=videos or None,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
