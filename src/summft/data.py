from datasets import load_dataset
from transformers import AutoTokenizer

def len_stats():
    dataset = load_dataset("Samsung/samsum")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    #train_dataset = dataset["train"]