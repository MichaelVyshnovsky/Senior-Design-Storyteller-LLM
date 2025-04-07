import json
import os
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hides GPUs from PyTorch 

model = "Qwen/Qwen2.5-Math-7B"
data_dir = "./SDdata/Dungeon"

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
                    input_text = flatten(doc)  # Using the flatten function here
                    data.append({
                        "input": input_text + "\nDescription:",  # Input to the model
                        "output": doc.get("mainbody", "")  # Output for the model (e.g., mainbody)
                    })
    return data

data = load_json_files(data_dir)
dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=512  # Reduce this further if needed
    )
    labels = tokenizer(
        examples["output"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=4, num_proc=1)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

model = AutoModelForCausalLM.from_pretrained(model, device_map="cpu")

torch.cuda.empty_cache()  # Clears GPU memory
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # Updated deprecated field
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=False  # Enables mixed precision to save GPU memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer  # Fix the deprecated field
)

trainer.train()
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
