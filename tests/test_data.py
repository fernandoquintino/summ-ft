from transformers import AutoTokenizer
from summft.data import tokenize_example

MODEL_NAME = model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
example = {
    "dialogue":
    """Amanda: I baked  cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)""",
    "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
}


def test_same_length():
    out = tokenize_example(example, tokenizer)
    assert len(out["input_ids"]) == len(out["labels"])


def test_only_summary_is_graded():
    out = tokenize_example(example, tokenizer)
    graded = [token for token, label in zip(out["input_ids"], out["labels"]) if label != -100]
    assert tokenizer.decode(graded) == example["summary"] + "<|im_end|>"