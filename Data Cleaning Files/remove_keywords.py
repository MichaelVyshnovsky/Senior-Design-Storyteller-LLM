import os
import re
from typing import Dict, List, Union

def remove_substrings_from_files(folder_path: str, 
                               patterns_dict: Dict[str, Union[str, re.Pattern]], 
                               file_extensions: List[str] = ['.txt']):
    """
    Recursively processes all files with given extensions in a folder and its subfolders,
    removing any text that matches patterns in the provided dictionary.
    """
    print(f"Starting processing in folder: {folder_path}")
    file_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext.lower()) for ext in file_extensions):
                file_path = os.path.join(root, file)
                print(f"Found matching file: {file_path}")  # Debug print
                file_count += 1
                process_file(file_path, patterns_dict)
    
    print(f"Total files found: {file_count}")

def process_file(file_path: str, patterns_dict: Dict[str, Union[str, re.Pattern]]):
    try:
        # Read file with encoding fallback
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            print(f"File is empty: {file_path}")
            return
            
        modified_content = content
        
        # First pass: Remove all templates and their contents
        modified_content = re.sub(r'\{\{.*?\}\}', '', modified_content, flags=re.DOTALL)
        modified_content = re.sub(r'\{\|.*?\|\}', '', modified_content, flags=re.DOTALL)
        
        # Second pass: Process links and formatting
        modified_content = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', modified_content)
        modified_content = re.sub(r"''+", '', modified_content)
        
        # Third pass: Clean up HTML and special cases
        modified_content = re.sub(r'<ref[^>]*>.*?</ref>', '', modified_content, flags=re.DOTALL)
        modified_content = re.sub(r'<[^>]+>', '', modified_content)
        modified_content = re.sub(r'^=+.*?=+\s*$', '', modified_content, flags=re.MULTILINE)
        
        # Process all patterns from the dictionary
        for pattern, replacement in patterns_dict.items():
            modified_content = re.sub(pattern, replacement, modified_content, flags=re.DOTALL)
        
        # Final cleanup
        modified_content = re.sub(r'\n\s*\n', '\n', modified_content)  # Empty lines
        modified_content = modified_content.strip()
        
        if modified_content != content:
            # Create backup if it doesn't exist
            backup_path = file_path + '.bak'
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"Processed and modified: {file_path}")
        else:
            print(f"No changes needed: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Configuration - modify these values
    FOLDER_TO_PROCESS = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"
    print(FOLDER_TO_PROCESS)
    
    # Dictionary of patterns to remove and their replacements
    PATTERNS_TO_REMOVE = {
        r'\{\{': "",  # Remove opening double braces
        r'\}\}': "",  # Remove closing double braces
        r'<ref name=".*?">': "",
        r'<ref name="*?">': "",  # Remove all reference tags
        r'</ref>': "",
        r'<ref name=".*?"/>': "",
        r'<ref name=".*?" />': "",
        r'{{DEFAULTSORT:.*?}}': "",
        r'{{Otheruses4.*?}}': "",
        '<small>': "",
        '</small>': "",
        r'{{Cite book/.*?}}': "",
        r'{{Cite game/.*?}}': "",
        r'{{Cite Game/.*?}}': "",
        r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]': r'\1',  # Clean wikilinks
        r"'''": "",   
        r"''": "",    
        r"'": "",
        r'\{': "",
        r'\}': "",
        r'<br />': "",
        r';': "",
        r':': ""
    }

    # File extensions to process
    FILE_EXTENSIONS = ['.txt']  # Add more as needed
    
    # Run the processing
    remove_substrings_from_files(FOLDER_TO_PROCESS, PATTERNS_TO_REMOVE, FILE_EXTENSIONS)
    print("Processing complete!")