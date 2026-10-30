from transformers import AutoModelForCausalLM, AutoTokenizer
from summft.mmlu import mmlu_correct

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

questions = [
    {"question": "What is 2 + 2?", "choices": ["2", "4", "88", "97"], "answer": 1},
    {
        "question": "What color is the sky on a clear day?",
        "choices": ["orange", "black", "red", "blue"],
        "answer": 3,
    },
]


def test_returns_one_bool_per_question():
    output = mmlu_correct(questions, model, tokenizer)
    assert len(output) == 2 and all(isinstance(x, bool) for x in output)
