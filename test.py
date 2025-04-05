import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model and tokenizer
model_path = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Verify model loading
print("Model loaded successfully")
print(f"Model device: {next(model.parameters()).device}")  # Should show 'cpu' or 'cuda'

# Sample input data
sample_input = {
    "image": "NewLevel.jpg",
    "name": "New Level",
    "level of": "[[Undermountain]]",
    "level number": "7",
    "location": "[[Waterdeep]]",
    "denizens": "[[Drow]], [[gargoyle]]s"
}

# Flatten function (must match training)
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

# Prepare input text
input_text = flatten(sample_input) + "\nDescription:"
print("\nInput text:")
print(input_text)

# Tokenize
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
print("\nInput tokens shape:", inputs["input_ids"].shape)

# Generation parameters
gen_config = {
    "max_length": 512,
    "min_length": 20,  # Ensure at least some output
    "num_beams": 5,
    "early_stopping": True,
    "no_repeat_ngram_size": 2,
    "temperature": 0.7,
    "do_sample": True  # More creative output
}

# Generate
with torch.no_grad():
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_config
        )
        print("\nGeneration successful!")
    except Exception as e:
        print(f"\nGeneration failed: {str(e)}")
        exit()

# Decode
full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_part = full_output[len(input_text):].strip()

print("\nFull output:")
print(full_output)
print("\nGenerated part only:")
print(generated_part)