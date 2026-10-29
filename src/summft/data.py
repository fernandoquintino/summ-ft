import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


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
            f"\n99th = {np.percentile(lengths, 99):.1f}"
            f"\nmax = {np.max(lengths)}"
        )
    print(f"Number of samples: {len(train_dataset)}")


SYSTEM_PROMPT = "Summarize this dialogue in one or two sentences."


def build_messages(example, with_summary=True, shots=()):
    """
    Builds a list of dict for system/user/assistant messages.
    Optional: few-shots, summary.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots:
        messages.append({"role": "user", "content": shot["dialogue"]})
        messages.append({"role": "assistant", "content": shot["summary"]})
    messages.append({"role": "user", "content": example["dialogue"]})
    if with_summary:
        messages.append({"role": "assistant", "content": example["summary"]})
    return messages


def tokenize_example(example, tokenizer):
    """
    Tokenizes one example for training with prompt (and trailing '\n')
    out of the loss.

    Args:
        example: dict with "dialogue" and "summary.
        tokenizer: the Qwen tokenizer.
    Returns:
        dict:
            - input_ids: prompt and summary
            - labels: Has same length as input_ids. Labels are set to
                -100 for the prompt and trailing '\n'. Only the summary
                and <|im_end|> are graded.
    """
    prompt_ids = tokenizer.apply_chat_template(
        build_messages(example, with_summary=False),
        add_generation_prompt=True,
        tokenize=True,
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        build_messages(example),
        tokenize=True,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + full_ids[prompt_len:-1] + [-100]
    return {"input_ids": full_ids, "labels": labels}
