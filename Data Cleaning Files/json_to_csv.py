import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional

def extract_text_from_section(section: Dict[str, Any] or str, indent_level: int = 0) -> str:
    """
    Recursively extract text from a section of the JSON structure.
    """
    text_parts = []
    indent = "  " * indent_level
    
    if isinstance(section, str):
        if section.strip():
            text_parts.append(f"{indent}{section.strip()}")
    elif isinstance(section, dict):
        for key, value in section.items():
            # Skip image references and empty fields
            if key.lower() in ["image", "caption"] or not value:
                continue
                
            # Handle special formatting for headers
            if key.lower() in ["name", "title"]:
                text_parts.append(f"\n{indent}== {value} ==")
            else:
                text_parts.append(f"\n{indent}=== {key} ===")
                
            # Recursively process the content
            text_parts.append(extract_text_from_section(value, indent_level + 1))
    elif isinstance(section, list):
        for item in section:
            text_parts.append(extract_text_from_section(item, indent_level))
    
    return "\n".join(filter(None, text_parts))

def process_wiki_json(data: Dict[str, Any]) -> str:
    """
    Process a wiki-style JSON document into formatted text.
    """
    text_parts = []
    
    # Extract main document info
    if "document_data" in data:
        doc_data = data["document_data"]
        
        # Add title/name
        if "name" in doc_data and doc_data["name"]:
            text_parts.append(f"== {doc_data['name']} ==")
            
            # Add location/level info if available
            location_parts = []
            if "level of" in doc_data and doc_data["level of"]:
                location_parts.append(doc_data["level of"])
            if "level number" in doc_data and doc_data["level number"]:
                location_parts.append(f"Level {doc_data['level number']}")
            if "location" in doc_data and doc_data["location"]:
                location_parts.append(doc_data["location"])
                
            if location_parts:
                text_parts.append("Location: " + ", ".join(location_parts))
        
        # Add main body text
        if "mainbody" in doc_data and doc_data["mainbody"]:
            text_parts.append("\n" + doc_data["mainbody"])
        
        # Process all other sections recursively
        for section_name, section_content in doc_data.items():
            if section_name.lower() in ["image", "caption", "name", "level of", 
                                      "level number", "location", "mainbody"]:
                continue
                
            if section_content:  # Only process non-empty sections
                text_parts.append(extract_text_from_section({section_name: section_content}))
    
    return "\n".join(filter(None, text_parts))

def json_to_csv(input_folder: str, output_file: str = "dnd_wiki_training.csv"):
    """
    Convert D&D wiki JSON files to training CSV.
    """
    records = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    processed_text = process_wiki_json(data)
                    if processed_text.strip():
                        records.append({
                            "text": processed_text,
                            "source": file  # Keep track of source file
                        })
                    else:
                        print(f"No valid text content in: {file}")
                
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    if not records:
        print("No valid records created.")
        return
    
    df = pd.DataFrame(records)
    
    # Basic cleaning
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() > 50]  # Remove very short entries
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Successfully created training CSV with {len(df)} entries at: {output_file}")

# Example usage
input_folder = r'C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\Data Cleaning Files'
json_to_csv(input_folder)