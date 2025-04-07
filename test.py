from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load your trained model 
model_path = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Or set to a specific token

# 2. Define a prompt wrapper (matches your training format)
def format_prompt(user_input):
    return f"""user_input: {user_input}
"document_data":"""  # Mimics your training data's structure

# 3. Interactive generation
while True:
    user_prompt = input("\nEnter your prompt (or 'quit' to exit):")
    if user_prompt.lower() == 'quit':
        break
    
    # Format the input to match training data style
    formatted_input = format_prompt(user_prompt)
    
    # Tokenize with attention mask
    inputs = tokenizer(
        formatted_input, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True,
        padding=True  # This ensures attention_mask is generated
    )
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Pass the attention mask
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id  # Explicitly set pad token
    )
    
    # Decode and clean output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_text[len(formatted_input):].strip()  # Remove input from output
    
    print("\nGenerated:")
    print(generated_part)