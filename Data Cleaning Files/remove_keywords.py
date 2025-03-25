import os
import re
from typing import Dict, List, Union

def remove_substrings_from_files(folder_path: str, 
                               patterns_dict: Dict[str, Union[str, re.Pattern]], 
                               file_extensions: List[str] = ['.txt']):
    """
    Recursively processes all files with given extensions in a folder and its subfolders,
    removing any text that matches patterns in the provided dictionary.
    
    Args:
        folder_path: Path to the root folder to process
        patterns_dict: Dictionary of patterns to remove (key) and their replacements (value)
                      Keys can be either plain strings or compiled regex patterns
        file_extensions: List of file extensions to process (default: ['.txt'])
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                process_file(file_path, patterns_dict)

def process_file(file_path: str, patterns_dict: Dict[str, Union[str, re.Pattern]]):
    """
    Processes a single file, removing all text that matches the patterns.
    
    Args:
        file_path: Path to the file to process
        patterns_dict: Dictionary of patterns and their replacements
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified_content = content
        for pattern, replacement in patterns_dict.items():
            if isinstance(pattern, re.Pattern):
                # If it's already a compiled pattern
                modified_content = pattern.sub(replacement, modified_content)
            else:
                # If it's a string, compile it as regex
                modified_content = re.sub(pattern, replacement, modified_content)
        
        if modified_content != content:
            # Create backup if it doesn't exist
            backup_path = file_path + '.bak'
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"Processed: {file_path}")
        else:
            print(f"No changes needed: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Configuration - modify these values
    FOLDER_TO_PROCESS = r""
    
    # Dictionary of patterns to remove and their replacements
    # Keys can be either plain strings or regex patterns
    PATTERNS_TO_REMOVE = {
    r'<ref name=".*?">': "",
    r'<ref name="*?">': "", # Remove all reference tags
    r'</ref>': "",
    r'<ref name=".*?"/>': "",
    r'<ref name=".*?" />': "",
    '<small>': "",
    '</small>': "",
    r'{{Cite book/.*?}}': "",
    r'{{Cite game/.*?}}': "",
    r'{{Cite Game/.*?}}': ""          
}
    
    # File extensions to process
    FILE_EXTENSIONS = ['.txt']  # Add more as needed
    
    # Run the processing
    remove_substrings_from_files(FOLDER_TO_PROCESS, PATTERNS_TO_REMOVE, FILE_EXTENSIONS)
    print("Processing complete!")