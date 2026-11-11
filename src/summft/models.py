from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def load_model(model_path, quant="none"):
    """
    Loads a model for evaluation, optionally with its base quantized.

    Args:
        model_path: the model id of a pretrained model hosted inside
            a model repo on huggingface.co, a full-model folder, or a
            LoRA adapter folder.
        quant: "none" (float16 on GPU / float32 on CPU), "8bit", or
            "4bit".
    Returns:
        the model on the GPU if available and in eval mode.
    """
    on_gpu = torch.cuda.is_available()
    is_adapter = (Path(model_path) / "adapter_config.json").exists()
    base_path = BASE_MODEL if is_adapter else model_path

    if quant == "none":
        dtype = torch.float16 if on_gpu else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
        model = model.to("cuda" if on_gpu else "cpu")
    else:
        if quant == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            raise ValueError(f"quant must be 'none', '8bit', or '4bit', got {quant!r}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            dtype=torch.float16,
        )

    if is_adapter:
        model = PeftModel.from_pretrained(model, model_path)

    model.eval()
    return model