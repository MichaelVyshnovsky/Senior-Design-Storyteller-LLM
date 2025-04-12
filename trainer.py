import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    set_seed
)

torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduces fragmentation
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Helps debug OOMs

# Set seed for reproducibility
set_seed(42)

model_name = "Qwen/Qwen2.5-Math-7B"
data_dir = "./SDdata/"

# Configure GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device_count = torch.cuda.device_count()
print(f"Available GPUs: {device_count}")

# Check CUDA and torch compatibility
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

def flatten(doc):
    """Flatten document data into a single string"""
    lines = []
    for key, value in doc.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, str):
                    lines.append(f"{key} - {subkey}: {subvalue}")
                elif isinstance(subvalue, dict):
                    for subsubkey, subsubvalue in subvalue.items():
                        if isinstance(subsubvalue, str):
                            lines.append(f"{key} - {subkey} - {subsubkey}: {subsubvalue}")
    return "\n".join(lines)

def load_json_files(data_dir):
    """Load and process JSON files into training examples"""
    data = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith(".json"):
                with open(os.path.join(root, file_name), "r", encoding="utf-8") as f:
                    content = json.load(f)
                    doc = content.get("document_data", {})
                    
                    # Create input text
                    input_text = flatten(doc)
                    
                    # Get mainbody text (or use empty string if not present)
                    output_text = doc.get("mainbody", "")
                    
                    if output_text:  # Only add if we have output text
                        data.append({
                            "input": input_text,
                            "output": output_text
                        })
    return data

# Load and prepare data
print("Loading data...")
data = load_json_files(data_dir)
print(f"Loaded {len(data)} examples")

# Create dataset
dataset = Dataset.from_dict({
    "input": [x["input"] for x in data],
    "output": [x["output"] for x in data]
})

# Initialize tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """Tokenize both inputs and labels"""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize outputs (labels)
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=8,
    remove_columns=["input", "output"],  # Remove original columns after tokenization
    num_proc=os.cpu_count()
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model_name,
    padding="longest",
    return_tensors="pt"
)

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="offload"
)
model.gradient_checkpointing_enable()

# Check if tf32 is supported
tf32_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
print(f"TF32 supported: {tf32_supported}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=True,  # Disabled FP16
    bf16=tf32_supported,  # Use BF16 if supported
    tf32=tf32_supported,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="adafactor",
    report_to="tensorboard",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    local_rank=int(os.environ.get("LOCAL_RANK", -1))
)

# Load model with appropriate precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # Uses FP16 to save memory
    offload_folder="offload",
    low_cpu_mem_usage=True
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets.select(range(min(100, len(tokenized_datasets)))) if len(tokenized_datasets) > 100 else tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train and save
print("Starting training...")
trainer.train()
print("Training complete!")

print("Saving model...")
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
print("Model saved successfully!")