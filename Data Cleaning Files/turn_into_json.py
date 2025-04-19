import json
import re
import os
from collections import OrderedDict

# Configuration - CHANGE THESE
INPUT_FOLDER = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"
OUTPUT_FOLDER = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"

def parse_content(content):
    """Parse both template patterns and document structure"""
    result = OrderedDict()
    
    # 1. Parse template patterns between {{...}}
    template_data = OrderedDict()
    template_blocks = re.findall(r'\{\{(.*?)\}\}', content, re.DOTALL)
    
    for block in template_blocks:
        kv_pairs = re.findall(r'\|([^=\s]+)\s*=\s*([^\n\|}]+)', block.strip())
        for key, value in kv_pairs:
            clean_key = key.strip()
            clean_value = value.strip()
            if clean_key in template_data:
                counter = 1
                while f"{clean_key}_{counter}" in template_data:
                    counter += 1
                clean_key = f"{clean_key}_{counter}"
            template_data[clean_key] = clean_value
    
    if template_data:
        result["template_data"] = template_data
    
    # 2. Parse document structure
    doc_data = OrderedDict()
    
    # Key-value pairs (|key=value) - only at document level
    kv_pairs = re.findall(r'^\|([^=]+)=([^\n]+)', content, re.MULTILINE)
    for key, value in kv_pairs:
        doc_data[key.strip()] = value.strip()
    
    # Main body content (after last key-value pair)
    last_kv_match = list(re.finditer(r'^\|([^=]+)=([^\n]+)', content, re.MULTILINE))
    mainbody_start = last_kv_match[-1].end() if last_kv_match else 0
    first_section_match = re.search(r'^==([^=]+)==', content[mainbody_start:], re.MULTILINE)
    mainbody_end = first_section_match.start() + mainbody_start if first_section_match else len(content)
    doc_data['mainbody'] = content[mainbody_start+1:mainbody_end].strip()
    
    # Sections (==section==) and subsections
    sections = re.finditer(r'^==([^=]+)==\n(.*?)(?=^==[^=]+==|\Z)', content, re.MULTILINE | re.DOTALL)
    for section in sections:
        section_name = section.group(1).strip()
        section_content = section.group(2).strip()
        
        # Initialize section dictionary
        section_dict = OrderedDict()
        
        # Check for subsections (===subsection===)
        subsections = re.finditer(r'^===([^=]+)===\n(.*?)(?=^===|\Z)', section_content, re.MULTILINE | re.DOTALL)
        has_subsections = False
        
        for subsection in subsections:
            has_subsections = True
            subsection_name = subsection.group(1).strip()
            subsection_content = subsection.group(2).strip()
            
            # Parse ;[[subsection]]: format (for both Geography and Inhabitants)
            if re.search(r'^;(\[\[[^\]]+\]]|[^:\n]+):', subsection_content, re.MULTILINE):
                subsubsection_dict = OrderedDict()
                subsubsections = re.finditer(r'^;(\[\[([^\]]+)\]\]|([^:\n]+)):\s*(.*?)(?=\n^;|\Z)', subsection_content, re.MULTILINE | re.DOTALL)
                for subsubsection in subsubsections:
                    # Get either the part inside [[ ]] or the whole match
                    subsubsection_name = subsubsection.group(2) or subsubsection.group(3)
                    subsubsection_value = subsubsection.group(4).strip()
                    subsubsection_dict[subsubsection_name.strip()] = subsubsection_value
                
                if subsubsection_dict:
                    section_dict[subsection_name] = subsubsection_dict
                else:
                    section_dict[subsection_name] = subsection_content
            else:
                section_dict[subsection_name] = subsection_content
        
        if has_subsections:
            doc_data[section_name] = section_dict
        else:
            # Parse ;[[subsection]]: or ;subsection: format (directly under section)
            if re.search(r'^;(\[\[[^\]]+\]]|[^:\n]+):', section_content, re.MULTILINE):
                subsubsection_dict = OrderedDict()
                subsubsections = re.finditer(r'^;(\[\[([^\]]+)\]\]|([^:\n]+)):\s*(.*?)(?=\n^;|\Z)', section_content, re.MULTILINE | re.DOTALL)
                for subsubsection in subsubsections:
                    # Get either the part inside [[ ]] or the whole match
                    subsubsection_name = subsubsection.group(2) or subsubsection.group(3)
                    subsubsection_value = subsubsection.group(4).strip()
                    subsubsection_dict[subsubsection_name.strip()] = subsubsection_value
                
                if subsubsection_dict:
                    doc_data[section_name] = subsubsection_dict
                else:
                    doc_data[section_name] = section_content
            else:
                doc_data[section_name] = section_content
    
    if doc_data:
        result["document_data"] = doc_data
    
    return result if (template_data or doc_data) else None

def process_file(input_path, output_path):
    """Process a single file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        json_data = parse_content(content)
        
        if not json_data:
            print(f"No parsable data found in: {input_path}")
            return None
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        
        print(f"Processed: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_folder():
    """Process all files in the input folder"""
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder does not exist - {INPUT_FOLDER}")
        return
    
    print(f"Processing files from {INPUT_FOLDER} to {OUTPUT_FOLDER}")
    stats = {'processed': 0, 'errors': 0, 'no_data': 0}
    
    for root, _, files in os.walk(INPUT_FOLDER):
        rel_path = os.path.relpath(root, INPUT_FOLDER)
        output_root = os.path.join(OUTPUT_FOLDER, rel_path)
        
        for file in files:
            if file.lower().endswith('.txt'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_root, os.path.splitext(file)[0] + '.json')
                
                result = process_file(input_path, output_path)
                if result is True:
                    stats['processed'] += 1
                elif result is None:
                    stats['no_data'] += 1
                else:
                    stats['errors'] += 1
    
    print("\nProcessing complete!")
    print(f"Files successfully processed: {stats['processed']}")
    print(f"Files with no parsable data: {stats['no_data']}")
    print(f"Files with errors: {stats['errors']}")

if __name__ == "__main__":
    process_folder()