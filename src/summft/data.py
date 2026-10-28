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


SYSTEM_PROMPT = "Summarize this dialogue in one or two sentences."


def build_messages(example, with_summary=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["dialogue"]},
    ]
    if with_summary:
        messages.append(
            {"role": "assistant", "content": example["summary"]}
        )
    return messages


def tokenize_example(example, tokenizer):
    prompt_ids = tokenizer.apply_chat_template(
        build_messages(example, with_summary=False),
        add_generation_prompt=True,
        tokenize=True,
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        build_messages(example),
        tokenizer=True,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + full_ids[prompt_len:-1] + [-100]
    return {"input_ids": full_ids, "labels": labels}

