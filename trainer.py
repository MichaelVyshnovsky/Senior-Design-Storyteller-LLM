import json
import os
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

data_dir = "path_to_json_folder"  # Change this to your dataset directory

def load_json_files(data_dir):
    data = []
    for root, _, files in os.walk(data_dir):  # Walk through subdirectories
        for file_name in files:
            if file_name.endswith(".json"):
                with open(os.path.join(root, file_name), "r", encoding="utf-8") as f:
                    content = json.load(f)
                    doc = content.get("document_data", {})
                    input_text = "\n".join([f"{key}: {value}" for key, value in doc.items() if isinstance(value, str)])
                    data.append({
                        "input": input_text + "\nDescription:",
                        "output": doc.get("mainbody", "")
                    })
    return data

data = load_json_files(data_dir)
dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input"], padding="max_length", truncation=True)
    labels = tokenizer(examples["output"], padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=16, num_proc=4)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="Qwen/Qwen2.5-Math-7B")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  # Avoid error by setting evaluation off unless eval_dataset is provided
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
