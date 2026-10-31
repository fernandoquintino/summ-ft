from transformers import AutoTokenizer

from summft.data import (
    build_messages,
    tokenize_example,
    MAX_LENGTH,
    build_train_dataset,
    make_collator,
)

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


def test_build_train_dataset():
    ds = build_train_dataset(tokenizer, split="validation", limit=5)
    assert ds.column_names == ["input_ids", "labels"]
    assert all(len(r["input_ids"]) == len(r["labels"]) for r in ds)


def test_make_collator_right():
    collate = make_collator(pad_token_id=0)
    batch = [
        {"input_ids": [5, 6, 7, 8], "labels": [-100, -100, 7, 8]},
        {"input_ids": [9, 10], "labels": [-100, 10]},
    ]
    batch = collate(batch)
    assert batch["input_ids"].tolist() == [[5, 6, 7, 8], [9, 10, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 7, 8], [-100, 10, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]