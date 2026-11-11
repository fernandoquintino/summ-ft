import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from summft.mmlu import mmlu_correct
from summft.models import load_model

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 83


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--tokenizer", default=BASE_MODEL)
    parser.add_argument("--name", default="base")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    args = parser.parse_args()

    mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")

    idx = random.Random(SEED).sample(range(len(mmlu_dataset)), args.n)
    questions = mmlu_dataset.select(idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args.model, args.quant)

    model_mem_gb = round(model.get_memory_footprint()/1e9, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    correct = mmlu_correct(questions, model, tokenizer)
    minutes = (time.time() - start) / 60

    peak_mem_gb = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None
    )
    accuracy = sum(correct) / len(correct)

    out_dir = Path("results") / f"mmlu_{args.name}_{args.n}questions"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_question.jsonl", "w") as f:
        for i, q, c in zip(idx, questions, correct):
            f.write(
                json.dumps({"index": i, "subject": q["subject"], "correct": c}) + "\n"
            )

    metrics = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "n": len(correct),
        "accuracy": round(100 * accuracy, 1),
        "minutes": round(minutes, 1),
            "quant": args.quant,
        "model_mem_gb": model_mem_gb,
        "peak_mem_gb": peak_mem_gb,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
