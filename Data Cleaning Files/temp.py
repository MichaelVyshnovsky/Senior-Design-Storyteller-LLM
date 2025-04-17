import os

def fix_double_txt(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.txt.txt'):  # Target the messed-up files
                old_path = os.path.join(foldername, filename)
                new_name = filename[:-8] + '.txt'  # Removes the last `.txt` (8 chars: .txt.txt → .txt)
                new_path = os.path.join(foldername, new_name)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Fixed: {old_path} → {new_path}")
                except Exception as e:
                    print(f"Error fixing {old_path}: {e}")

# Usage:
folder_path = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"  # Replace with your folder
fix_double_txt(folder_path)