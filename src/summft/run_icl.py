import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from summft.generate import generate_summaries
from summft.metrics import rouge_scores

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "knkarthick/samsum"
SEED = 83


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = load_dataset(DATASET_NAME)
    examples = dataset[args.split]
    train = dataset["train"]
    if args.limit:
        examples = examples.select(range(args.limit))

    shots_idx = random.Random(SEED).sample(range(len(train)), args.shots)
    shots = train.select(shots_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    model.eval()

    start = time.time()
    predictions = generate_summaries(examples, model, tokenizer, shots=shots)
    minutes = (time.time() - start) / 60

    references = examples["summary"]
    scores = rouge_scores(predictions, references)

    out_dir = Path("results") / f"icl_{args.shots}shot_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w") as f:
        f.writelines(
            json.dumps({"id": ex["id"], "reference": ex["summary"], "prediction": pred})
            + "\n"
            for ex, pred in zip(examples, predictions)
        )
    metrics = {
        "shots": args.shots,
        "split": args.split,
        "n": len(examples),
        "minutes": round(minutes, 1),
        **scores,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
