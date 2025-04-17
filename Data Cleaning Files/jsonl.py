import os
import re
import json
from bs4 import BeautifulSoup

def clean_wiki_text(text):
    """Remove wiki templates, citations, and markup."""
    # Remove {{Cite...}}, {{Dungeon|...}}, etc.
    text = re.sub(r'\{\{(Cite|Dungeon|Fq|YearlinkName|SI)[^}]*\}\}', '', text)
    # Remove [[links]] and ''italics''
    text = re.sub(r'\[\[([^|]*?\|)?([^]]*?)\]\]', r'\2', text)
    text = re.sub(r'\'\'(.*?)\'\'', r'\1', text)
    # Remove HTML tags (if any)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def infer_entity_type(text):
    """Guess if the content describes a person, place, item, or concept."""
    text_lower = text.lower()
    if " was a " in text_lower or " were " in text_lower:
        return "concept"
    elif " built by " in text_lower or " located in " in text_lower:
        return "location"
    elif " wielded " in text_lower or " made of " in text_lower:
        return "item"
    elif " was born " in text_lower or " is a " in text_lower:
        return "person"
    else:
        return "unknown"

def process_file(file_path):
    """Extract and structure text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = clean_wiki_text(content)
    entity_type = infer_entity_type(cleaned)
    
    # Extract the first sentence as a summary
    first_sentence = re.split(r'[.!?]', cleaned)[0].strip()
    
    return {
        "text": cleaned,
        "summary": first_sentence,
        "type": entity_type,
        "source": os.path.basename(file_path)
    }

def process_folder(input_folder, output_file):
    """Convert all files in a folder to JSONL."""
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_folder):
            if filename.endswith(('.html', '.txt', '.md')):
                file_path = os.path.join(input_folder, filename)
                try:
                    data = process_file(file_path)
                    json.dump(data, out_f, ensure_ascii=False)
                    out_f.write('\n')
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Usage
input_folder = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"  # Folder with HTML/wiki files
output_file = "output.jsonl"         # Output JSONL file
process_folder(input_folder, output_file)
print(f"Converted files saved to {output_file}")