from summft.data import build_messages


def generate_summaries(examples, model, tokenizer, max_new_tokens=100):
    summaries = []
    for example in examples:
        messages = build_messages(example, with_summary=False)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        new_tokens= output[0][inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
        summaries.append(summary)
    return summaries