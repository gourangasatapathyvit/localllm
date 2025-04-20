import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from transformers import (AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig)
from peft import PeftModel


tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# 1. Configure 4-bit loading
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)

# 3. Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "./phi2-jobs/checkpoint-62",
    device_map="auto"
)

# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/phi-2",
#     torch_dtype=torch.float16,
#     device_map="auto",
#     llm_int8_enable_fp32_cpu_offload=True ,
#     load_in_4bit=True  # Critical for 100K samples
# )

# 4. Query function
def query_jobs(question):
    inputs = tokenizer(
        f"Question: {question}\nAnswer:",
        return_tensors="pt",
        return_attention_mask=False
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test it
print(query_jobs("about microsoft"))