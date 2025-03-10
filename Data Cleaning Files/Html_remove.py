""""""
import os
from bs4 import BeautifulSoup
import re

def html_to_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Use BeautifulSoup to parse and extract text
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text()
    
    # Save the extracted text to an output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(plain_text)
    
    print(f"Text extracted and saved to {output_file}")

def process_folder(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        target_folder = os.path.join(output_folder, relative_path)
        
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        for filename in files:
            if filename.endswith(".html"):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(target_folder, filename.replace(".html", ".txt"))
                html_to_text(input_path, output_path)

def clean_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove text before and including 'text/x-wiki' and any resulting newlines
    content = re.sub(r'.*?text/x-wiki\n*', '', content, flags=re.DOTALL)
    
    # Remove text after and including '==Appendix=='
    content = re.sub(r'==Appendix==.*', '', content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                clean_text_file(file_path)
                print(f"Processed: {file_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\micha\Documents\School\Senior\CSE 4940 Senior Design\SDdata"  # Replace with your input folder
    output_folder = r"C:\Users\micha\Documents\School\Senior\CSE 4940 Senior Design\SDdata"  # Replace with your output folder
    if os.path.isdir(input_folder):
        #process_folder(input_folder, output_folder)
        process_directory(input_folder)
        print("Processing complete.")
    else:
        print("Invalid folder path.")
