import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

lora_model_path = "/Users/megi/Documents/Other/LLM/readlike-me/utils/lora-alpaca-final"

peft_config = PeftConfig.from_pretrained(lora_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

model = PeftModel.from_pretrained(base_model, lora_model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
def format_prompt(instruction, input_text=None):
    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        return f"""### Instruction:
{instruction}

### Response:"""

def infer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = format_prompt("Describe a sunny day. Mimic the style of Dostoyevsky.")

resp = infer(prompt)
print(resp)
