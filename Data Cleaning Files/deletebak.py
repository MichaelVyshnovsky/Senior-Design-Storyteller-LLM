import os

def delete_bak_files(folder_path: str):
    """
    Recursively deletes all .bak files in a folder and its subfolders.
    
    Args:
        folder_path: Path to the root folder to process
    """
    deleted_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
                    error_count += 1
    
    print(f"\nDeletion complete! Removed {deleted_count} .bak files.")
    if error_count > 0:
        print(f"Encountered {error_count} errors during deletion.")

if __name__ == "__main__":
    # Configuration - modify this value
    FOLDER_TO_PROCESS = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"
    
    # Ask for confirmation before deleting
    confirm = input(f"WARNING: This will permanently delete all .bak files in {FOLDER_TO_PROCESS} and its subfolders.\n"
                   f"Do you want to continue? (y/n): ").strip().lower()
    
    if confirm == 'y':
        delete_bak_files(FOLDER_TO_PROCESS)
    else:
        print("Operation cancelled.")