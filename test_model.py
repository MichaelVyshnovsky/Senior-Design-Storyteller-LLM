from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your trained model and tokenizer
model_path = "./results/model_20250417_153158"  # Replace with your actual model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate text from the model given a prompt
    
    Args:
        prompt (str): Input text to start generation
        max_length (int): Maximum length of generated text
        temperature (float): Controls randomness (lower = more deterministic)
        top_k (int): Top-k sampling parameter
        top_p (float): Nucleus sampling parameter
        
    Returns:
        str: Generated text
    """
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length + len(input_ids[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = """###Guidelines

- You are a fantasy storyteller tasked with designing a new character to fit in this world
- Using the provided context make the character integrated into the world
- Before generating new people, places, and concepts use those from the provided context if possible
- Use the provided template without markdown syntax
- Only return HTML. Do not include explanations or extra narration outside the structured format.
- Use as much provided context as possible
- Format the response using clean, raw HTML. Do NOT use markdown (like **bold**, ### headers, etc.).


###Template 
Respond using this clean and consistent HTML structure: (No Bold)

<h2>Character Name: {{Character Name}}</h2>

<h3>Background</h3>
<p>Provide a concise summary of the character\'s origin, beliefs, and general personality. Talk about their homeland as well.</p>

<h3>Abilities</h3>
<ul>
  <li>Describe unique skills or magical abilities.</li>
  <li>Focus on gameplay-relevant traits or powers.</li>
</ul>

<h3>Relationships</h3>
<ul>
  <li>Important allies and enemies, with 1-2 sentences of context.</li>
</ul>

<h3>Personality</h3>
<p>Key psychological traits, including behavior and motivation.</p>

<h3>Appearance</h3>
<p>Physical features, clothing, and notable items.</p>

<h3>Faction Role</h3>
<p>Explain their role within their faction or organization.</p>

named Mike from GreenLand who is a Bard and is a part of awesome bandits In addition the character knows of sword of death
"""
generated_text = generate_text(prompt)
print("Prompt:", prompt)
print("Generated text:", generated_text)