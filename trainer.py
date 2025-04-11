import json
import os
import torch
from datasets import Dataset, load_dataset
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
data_dir = "./SDdata/"

# Enable multi-GPU training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Specify which GPUs to use
torch.backends.cuda.matmul.allow_tf32 = True  # Enable tf32 for matrix multiplications
device_count = torch.cuda.device_count()
print(f"Available GPUs: {device_count}")

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

data = load_json_files(data_dir)
dataset = Dataset.from_list(data)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    labels = tokenizer(
        examples["output"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Parallelize dataset processing
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    batch_size=8,  # Increased batch size for better GPU utilization
    num_proc=os.cpu_count()  # Use all available CPU cores
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model_name,
    padding="longest",
    return_tensors="pt"
)

# Load model with device_map="auto" for automatic multi-GPU distribution
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distributes across available GPUs
    torch_dtype=torch.bfloat16,  # Uses bfloat16 for better memory efficiency
    # low_cpu_mem_usage=True,
    # attn_implementation="flash_attention_2"  # If available
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Training arguments optimized for multi-GPU
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Batch size per GPU
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Accumulates gradients before optimization step
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=True,  # Mixed precision training
    bf16=False,  # Use if your GPUs support bfloat16
    tf32=True,  # Use if your GPUs support tf32
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster data transfer to GPU
    gradient_checkpointing=True,
    optim="adamw_torch_fused",  # Fused optimizer for better performance
    report_to="tensorboard",
    ddp_find_unused_parameters=False,  # For distributed training
    deepspeed=None,  # Can specify deepspeed config file if using DeepSpeed
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # For distributed training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets.select(range(100)),  # Small subset for evaluation
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train and save
trainer.train()
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

print("Training complete!")