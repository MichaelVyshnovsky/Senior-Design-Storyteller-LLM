import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model and tokenizer
model_path = "./results"  # Path where you saved your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Sample input data (formatted similarly to your training data)
sample_input = {
    "image": "NewLevel.jpg",
    "name": "New Level",
    "level of": "[[Undermountain]]",
    "level number": "7",
    "location": "[[Waterdeep]]",
    "denizens": "[[Drow]], [[gargoyle]]s"
}

# Flatten the input (using the same function from your training script)
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

input_text = flatten(sample_input) + "\nDescription:"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,  # Same as during training
        num_beams=5,  # Beam search for better quality
        early_stopping=True
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# The output will include both input and generation, so we'll split to get just the new part
generated_part = generated_text[len(input_text):].strip()

print("Generated Description:")
print(generated_part)