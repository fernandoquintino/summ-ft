import torch

LETTERS = ["A", "B", "C", "D"]


def format_question(row):
    """Question followed by four lettered choices."""
    lines = [row["question"]]
    for letter, choice in zip(LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def mmlu_correct(questions, model, tokenizer):
    """
    For each multiple choice question (A/B/C/D), the answer is whichever
    the model predicts the highest. Returns correctness as a list of
    bools.
    """
    model.eval()
    letter_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in LETTERS]
    results = []
    for row in questions:
        messages = [{"role": "user", "content": format_question(row)}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        # Pick logits of first token after prompt.
        last_position = logits[0, -1]
        letter_logits = last_position[letter_ids]
        prediction = letter_logits.argmax()
        results.append(prediction.item() == row["answer"])
    return results
