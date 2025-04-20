# Local LLM SQL Fine-tuning Project

This project sets up DeepSeek-Coder 1.3B for local SQL query generation through fine-tuning.

## Requirements

- Python 3.8+
- CUDA-compatible GPU with 8GB+ VRAM
- 16GB+ RAM

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Modify the table schemas in `train_sql_model.py` to match your database structure:
```python
table_schemas = {
    "table_a": {"columns": ["your", "actual", "columns"]},
    "table_b": {"columns": ["your", "actual", "columns"]}
}
```

3. Run the training script:
```bash
python train_sql_model.py
```

## Usage

The model will be fine-tuned on your specific SQL tables and can generate queries based on natural language instructions. The training script prepares the model with LoRA adapters for efficient fine-tuning while maintaining the base model's capabilities.

## Model Details

- Base Model: DeepSeek-Coder 1.3B
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Task: SQL Query Generation
- Supported Operations: SELECT, INSERT, UPDATE


- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

- pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui


