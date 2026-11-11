import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from summft.generate import generate_summaries
from summft.metrics import rouge_scores
from summft.models import load_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "knkarthick/samsum"
SEED = 83


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--name", default="base")
    parser.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    args = parser.parse_args()

    dataset = load_dataset(DATASET_NAME)
    examples = dataset[args.split]
    train = dataset["train"]
    if args.limit:
        examples = examples.select(range(args.limit))

    shots_idx = random.Random(SEED).sample(range(len(train)), args.shots)
    shots = train.select(shots_idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(args.model, args.quant)
    model_mem_gb = round(model.get_memory_footprint() / 1e9, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    predictions = generate_summaries(examples, model, tokenizer, shots=shots)
    minutes = (time.time() - start) / 60

    peak_mem_gb = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None
    )
    references = examples["summary"]
    scores = rouge_scores(predictions, references)

    out_dir = Path("results") / f"{args.name}_{args.shots}shot_{args.split}"
    if args.limit:
        out_dir = out_dir.with_name(out_dir.name + f"_limit{args.limit}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w") as f:
        for ex, pred in zip(examples, predictions):
            f.write(
                json.dumps(
                    {"id": ex["id"], "reference": ex["summary"], "prediction": pred}
                )
                + "\n"
            )
    metrics = {
        "model": args.model,
        "shots": args.shots,
        "split": args.split,
        "n": len(examples),
        "minutes": round(minutes, 1),
        **scores,
        "quant": args.quant,
        "model_mem_gb": model_mem_gb,
        "peak_mem_gb": peak_mem_gb,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
