from transformers import AutoTokenizer

from summft.data import build_messages, tokenize_example

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
example = {
    "dialogue": """Amanda: I baked  cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)""",
    "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
}


def test_same_length():
    out = tokenize_example(example, tokenizer)
    assert len(out["input_ids"]) == len(out["labels"])


def test_only_summary_is_graded():
    out = tokenize_example(example, tokenizer)
    graded = [
        token for token, label in zip(out["input_ids"], out["labels"]) if label != -100
    ]
    assert tokenizer.decode(graded) == example["summary"] + "<|im_end|>"


def test_few_shot_order():
    shots = [{"dialogue": "d1", "summary": "s1"}, {"dialogue": "d2", "summary": "s2"}]
    msgs = build_messages({"dialogue": "real"}, with_summary=False, shots=shots)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"] == "d1" and msgs[2]["content"] == "s1"
    assert msgs[-1]["content"] == "real"
