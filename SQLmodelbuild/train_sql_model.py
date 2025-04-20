import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
import json

def load_model_and_tokenizer():
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True  # Critical for large samples
    )
    return model, tokenizer

def prepare_for_training(model):
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    return model

def create_sql_dataset(table_schemas):
    training_data = []
    for table_name, schema in table_schemas.items():
        # SELECT queries
        cols = ", ".join(schema['columns'])
        training_data.append({
            "instruction": f"Write a SQL query to select all columns from {table_name}",
            "input": f"Table {table_name} has columns: {cols}",
            "label": f"SELECT * FROM {table_name};"
        })
        # INSERT queries
        vals = ", ".join(["?" for _ in schema['columns']])
        training_data.append({
            "instruction": f"Write a SQL query to insert data into {table_name}",
            "input": f"Table {table_name} has columns: {cols}",
            "label": f"INSERT INTO {table_name} ({cols}) VALUES ({vals});"
        })
        # UPDATE queries
        training_data.append({
            "instruction": f"Write a SQL query to update records in {table_name}",
            "input": f"Table {table_name} has columns: {cols}",
            "label": f"UPDATE {table_name} SET {schema['columns'][1]} = ? WHERE {schema['columns'][0]} = ?;"
        })
    return Dataset.from_pandas(pd.DataFrame(training_data))

def format_training_prompt(example):
    return f'''### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Output:{example['label']} '''

def main():
    # Initialize model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    model = prepare_for_training(model)
    
    # Example table schemas - modify according to your actual tables
    table_schemas = {
        "customers": {
            "columns": ["customer_id", "first_name", "last_name", "email", "phone", "address", "created_at"]
        },
        "products": {
            "columns": ["product_id", "name", "description", "price", "stock_quantity", "category", "supplier_id"]
        },
        "orders": {
            "columns": ["order_id", "customer_id", "order_date", "total_amount", "status", "shipping_address"]
        },
        "order_items": {
            "columns": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "subtotal"]
        },
        "suppliers": {
            "columns": ["supplier_id", "company_name", "contact_name", "email", "phone", "address"]
        }
    }
    
    # Create dataset
    dataset = create_sql_dataset(table_schemas)
    
    # Save table schemas for future reference
    with open('table_schemas.json', 'w') as f:
        json.dump(table_schemas, f, indent=2)
    
    def tokenize_function(examples):
        prompts = [
            format_training_prompt({
                "instruction": instr,
                "input": inp,
                "label": lbl
            })
            for instr, inp,lbl in zip(examples["instruction"], examples["input"], examples["label"])
        ]
        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        # Tokenize the labels (outputs)
        input_prompts = [
            f"### Instruction: {instr}\n### Input: {inp}\n### Output:"
            for instr, inp in zip(examples["instruction"], examples["input"])
        ]
    
        # Tokenize just the input prompts to find their length
        tokenized_prompts = tokenizer(
            input_prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Mask the input part by setting labels to -100
        labels = model_inputs["input_ids"].clone()
        input_length = tokenized_prompts["input_ids"].shape[1]
        labels[:, :input_length] = -100
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=8,remove_columns=dataset.column_names)
    print(tokenized_dataset[0].keys())
    # tokenized_dataset = tokenized_dataset.remove_columns(["label"])
    
    training_args = TrainingArguments(
        output_dir="./sql-adapters",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        remove_unused_columns=False
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Helps with GPU efficiency
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    print("Starting training...")
    trainer.train()
    model.save_pretrained("./sql-adapters/final-model")
    tokenizer.save_pretrained("./sql-adapters/final-model")

if __name__ == "__main__":
    main()