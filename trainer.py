import json
import os
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

data_dir = "./SDdata"  # Change this to your dataset directory

def load_json_files(data_dir):
    data = []
    for root, _, files in os.walk(data_dir):  # Walk through subdirectories
        for file_name in files:
            if file_name.endswith(".json"):
                with open(os.path.join(root, file_name), "r", encoding="utf-8") as f:
                    content = json.load(f)
                    doc = content.get("document_data", {})
                    data.append({
                        "input": f"Title: {doc.get('title', '')}\nSetting: {doc.get('setting', '')}\nStory Arc: {doc.get('story arc', '')}\nDescription:",
                        "output": doc.get("mainbody", "")
                    })
    return data

data = load_json_files(data_dir)
dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1:7b")

def tokenize_function(examples):
    return tokenizer(examples["input"], text_target=examples["output"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-r1:7b")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
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
    tokenizer=tokenizer
)

trainer.train()
