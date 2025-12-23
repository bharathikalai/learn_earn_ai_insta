import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = """### Instruction:
Explain Ubuntu

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))
