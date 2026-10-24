from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def len_stats():
    dataset = load_dataset("knkarthick/samsum")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    train_dataset = dataset["train"]
    for col in ["dialogue", "summary"]:
        texts = [t or "" for t in train_dataset[col]]
        lengths = np.array([len(ids) for ids in tokenizer(texts)["input_ids"]])
        print(
            f"--------------{col}--------------"
            f"\nmean = {lengths.mean():.1f} "
            f"\nmedian = {np.median(lengths):.1f}"
            f"\n95th = {np.percentile(lengths, 95):.1f}"
            f"\n99th = {np.percentile(lengths,99):.1f}"
            f"\nmax = {np.max(lengths)}"
        )
    print(f"Number of samples: {len(train_dataset)}")
