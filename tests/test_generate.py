from transformers import AutoModelForCausalLM, AutoTokenizer
from summft.generate import generate_summaries

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
examples = [
    {
        "dialogue":
    """Amanda: I baked  cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)""",
    },
    {
        "dialogue":
        """Olivia: Who are you voting for in this election?
Oliver: Liberals as always.
Olivia: Me too!!
Oliver: Great"""
    }
]

def test_one_summary_per_example():
    out = generate_summaries(examples, model, tokenizer, max_new_tokens=40)
    assert len(out) == len(examples)
    assert all(isinstance(s, str) and s for s in out)
    assert not any("assistant" in s or "system" in s for s in out)