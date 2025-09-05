import torch
import streamlit as st
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

st.title("LLM Style Mimic App")
instruction = st.text_input("Instruction", "Describe a sunny day. Mimic the writing style of Mikhail Bulgakov.")
input_text = st.text_area("Input (optional)", "")

if st.button("Generate Response"):
    prompt = format_prompt(instruction, input_text if input_text.strip() else None)
    with st.spinner("Generating..."):
        resp = infer(prompt)
    st.write("### Response")
    st.write(resp)