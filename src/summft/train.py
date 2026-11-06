import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from summft.data import build_train_dataset, make_collator

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 83
TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None)  # None -> 2 x rank
    parser.add_argument("--targets", choices=["attn", "all"], default="all")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    alpha = args.alpha or 2 * args.rank
    on_gpu = torch.cuda.is_available()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "summ-ft")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = build_train_dataset(tokenizer=tokenizer, split="train", limit=args.limit)
    val_ds = build_train_dataset(
        tokenizer=tokenizer, split="validation", limit=args.limit
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        target_modules=TARGETS[args.targets],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dir = Path("models") / f"lora_r{args.rank}_{args.targets}_epochs{args.epochs:g}"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16 // (args.batch_size * world_size),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=on_gpu,
        train_sampling_strategy="group_by_length",
        remove_unused_columns=False,
        report_to="mlflow",
        run_name=out_dir.name,
        seed=SEED,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
    )

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = trainer.train()
    minutes = (time.time() - start) / 60
    eval_loss = trainer.evaluate()["eval_loss"]

    trainer.save_model(str(out_dir))
    metrics = {
        "method": "lora",
        "rank": args.rank,
        "alpha": alpha,
        "targets": args.targets,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_train": len(train_ds),
        "trainable_params": trainable,
        "train_loss": round(result.training_loss, 4),
        "eval_loss": round(eval_loss, 4),
        "best_step": trainer.state.best_global_step,
        "minutes": round(minutes, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if on_gpu
        else None,
        "batch_size": args.batch_size,
        "world_size": world_size,
    }

    if trainer.is_world_process_zero():
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out_dir / "log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(metrics)

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
