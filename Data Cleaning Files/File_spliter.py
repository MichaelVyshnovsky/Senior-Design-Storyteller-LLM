import re
import os

def split_html_from_xml(xml_file):
    with open(xml_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Regex pattern to match <title>Section Name</title>
    sections = re.split(r'<title>(.*?)</title>', content)
    
    # Ensure the output directory exists
    output_dir = "output_sections"
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the sections, skipping the first empty split
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip()
        section_content = sections[i + 1].strip()
        
        # Skip sections that start with Forum, User, File, or Template
        if section_name.startswith(("Forum", "User", "File", "Template")):
            print(f"Skipping: {section_name}")
            continue
        
        # Create a filename-friendly version of section name
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', section_name)
        file_path = os.path.join(output_dir, f"{safe_name}.html")
        
        with open(file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(section_content)
        
        print(f"Created: {file_path}")

# Example usage
xml_filename = "forgottenrealms_pages_current.xml"  # Change this to the actual file name
split_html_from_xml(xml_filename)
