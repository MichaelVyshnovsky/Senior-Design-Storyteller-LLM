import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
from datetime import datetime

# Configuration - CHANGED TO GPT-2
MODEL_NAME = "gpt2"  # Using GPT-2 which is designed for causal LM
DATA_ROOT = "./SDdata"              
RESULTS_DIR = "./results"         
BATCH_SIZE = 4                    # Reduced for GPT-2 which is larger
NUM_EPOCHS = 3                    
LEARNING_RATE = 5e-5              # Adjusted learning rate

# Set up environment
os.makedirs(RESULTS_DIR, exist_ok=True)
accelerator = Accelerator()
device = accelerator.device

# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs")
if num_gpus < 5:
    print("Warning: Expected 5 GPUs but found fewer. Performance may be impacted.")

# Custom Dataset for JSON files
class JSONDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            if isinstance(json_data, dict) and 'document_data' in json_data:
                                text = str(json_data['document_data'])
                                self.data.append(text)
                            else:
                                print(f"File {file_path} doesn't contain 'document_data' or is not a dict")
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error reading {file_path}: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error with {file_path}: {str(e)}")
        
        if not self.data:
            raise ValueError(f"No valid training data found in {data_dir}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return inputs

# Initialize model and tokenizer - ADDED PAD_TOKEN
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for GPT-2

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Prepare dataset
try:
    full_dataset = JSONDataset(DATA_ROOT, tokenizer)
    print(f"Loaded {len(full_dataset)} samples")
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
except ValueError as e:
    print(f"Data loading error: {str(e)}")
    exit(1)

# Data collator - CHANGED FOR CAUSAL LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{RESULTS_DIR}/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=min(4, os.cpu_count()),
    report_to="none",
    # ADDED TO PREVENT WARNINGS
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
try:
    trainer.train()
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{RESULTS_DIR}/model_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")
except Exception as e:
    print(f"Training failed: {str(e)}")
    exit(1)