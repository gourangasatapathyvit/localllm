import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Add this line to suppress OMP warning
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_fine_tuned_model():
    base_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    adapter_path = "./sql-adapters/final-model"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the fine-tuned model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

def generate_sql_query(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=2,
        early_stopping=True
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(full_output.split("### Output:"))
    
    # First try to find the actual SQL query after "### Output:"
    if "### Output:" in full_output:
        sql_part = full_output.split("### Output:")[-1].strip()
        # Clean up any remaining ### markers
        sql_part = sql_part.replace("###", "").strip()
        # Look for SQL keywords in this part
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']
        for keyword in sql_keywords:
            if keyword in sql_part:
                sql_start = sql_part.find(keyword)
                sql_end = sql_part.find(';', sql_start) + 1
                return sql_part[sql_start:sql_end] if sql_end > sql_start else sql_part[sql_start:]
    
    # Fallback to searching the entire output
    for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']:
        if keyword in full_output:
            sql_start = full_output.find(keyword)
            sql_end = full_output.find(';', sql_start) + 1
            sql_part = full_output[sql_start:sql_end] if sql_end > sql_start else full_output[sql_start:]
            # Clean up any ### markers in the fallback case too
            return sql_part.replace("###", "").strip()
    
    return full_output.replace("###", "").strip()  # final fallback

def main():
    # Load the model and tokenizer
    print("Loading fine-tuned model...")
    model, tokenizer = load_fine_tuned_model()
    
    # Load table schemascls
    with open('table_schemas.json', 'r') as f:
        table_schemas = json.load(f)
    
    # Test cases
    test_cases = [
        # {
        #     "instruction": "who is modi?",
        #     "table_info": "Tables: customers (columns: customer_id, first_name, last_name, email), orders (columns: order_id, customer_id, order_date, total_amount)"
        # },
        # {
        #     "instruction": "find the total value of all orders for each product along with product names whose order_id=1",
        #     "table_info": "Tables: products (columns: product_id, name, price), order_items (columns: order_item_id, order_id, product_id, quantity, unit_price)"
        # },
        # {
        #     "instruction": "Write a SQL query to insert a new supplier",
        #     "table_info": f"Table suppliers has columns: {', '.join(table_schemas['suppliers']['columns'])}"
        # },
            {
            "instruction": "where is china",
            "table_info": "Tables: products (columns: product_id, name, price)"
        },
    ]
    
    # print("\nGenerating SQL queries...\n")
    for case in test_cases:
        prompt = format_prompt(case['instruction'], case['table_info'])
        sql_query = generate_sql_query(model, tokenizer, prompt)
        # print(f"Instruction: {case['instruction']}")
        print(f"Generated SQL: {sql_query}\n")


def format_prompt(instruction, table_info):
    return f"### Instruction: {instruction}\n### Input: {table_info}\n### Output:"

if __name__ == "__main__":
    main()
