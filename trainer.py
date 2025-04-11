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

# Set seed for reproducibility
set_seed(42)

model_name = "Qwen/Qwen2.5-Math-7B"
data_dir = "./data/"

# Configure GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Specify which GPUs to use
device_count = torch.cuda.device_count()
print(f"Available GPUs: {device_count}")

# Check CUDA and torch compatibility
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

def format_example(doc):
    # Extract and flatten the document data
    input_parts = []
    
    # Add basic fields
    basic_fields = ["name", "level of", "level number", "location", "mainbody"]
    for field in basic_fields:
        if doc.get(field):
            input_parts.append(f"{field}: {doc[field]}")
    
    # Add nested Geography fields
    if doc.get("Geography"):
        geo = doc["Geography"]
        input_parts.append("\nGeography:")
        if geo.get("Rooms"):
            input_parts.append("Rooms:")
            for room, desc in geo["Rooms"].items():
                input_parts.append(f"- {room}: {desc}")
        if geo.get("Traffic"):
            input_parts.append(f"Traffic: {geo['Traffic']}")
    
    # Add Inhabitants
    if doc.get("Inhabitants"):
        input_parts.append("\nInhabitants:")
        for group, desc in doc["Inhabitants"].items():
            input_parts.append(f"- {group}: {desc}")
    
    # Combine all input parts
    input_text = "\n".join(input_parts)
    
    # Create the final example
    return {
        "input": input_text,
        "output": doc.get("mainbody", ""),  # Using mainbody as target output
        "full_text": input_text + "\n\n" + doc.get("mainbody", "")  # For causal LM training
    }

# Flatten nested dictionary fields into strings
def flatten(doc):
    lines = []
    for key, value in doc.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, str):
                    lines.append(f"{key} - {subkey}: {subvalue}")
    return "\n".join(lines)

def load_json_files(data_dir):
    data = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith(".json"):
                with open(os.path.join(root, file_name), "r", encoding="utf-8") as f:
                    content = json.load(f)
                    doc = content.get("document_data", {})
                    input_text = flatten(doc)
                    data.append({
                        "input": input_text + "\nDescription:",
                        "output": doc.get("mainbody", "")
                    })
    return data

# Load and prepare data
data = load_json_files(data_dir)
dataset = Dataset.from_list(data)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Tokenize the full text for causal language modeling
    tokenized = tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Calculate where the input ends and output begins
    input_tokenized = tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create labels (mask input portion with -100)
    input_length = len(input_tokenized["input_ids"][0])
    labels = tokenized["input_ids"].clone()
    labels[:, :input_length] = -100
    
    tokenized["labels"] = labels
    return tokenized

# Parallelize dataset processing
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    batch_size=8,
    num_proc=os.cpu_count()
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model_name,
    padding="longest",
    return_tensors="pt"
)

# Load model with automatic multi-GPU distribution
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Check if tf32 is supported
tf32_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
print(f"TF32 supported: {tf32_supported}")

# Training arguments optimized for multi-GPU
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=True,
    bf16=False,
    tf32=tf32_supported,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    report_to="none",
    ddp_find_unused_parameters=False,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    remove_unused_columns=False,  # Add this line
)

# Initialize Trainer with updated parameters
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets.select(range(min(100, len(tokenized_datasets)))),
    data_collator=data_collator,
    # No longer passing tokenizer directly as it's deprecated
)

# Train and save
print("Starting training...")
trainer.train()
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

print("Training complete!")