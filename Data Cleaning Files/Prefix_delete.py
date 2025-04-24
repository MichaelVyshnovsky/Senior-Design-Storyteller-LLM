import os

def delete_files_by_substring(folder_path, substring):
    """
    Deletes files in the specified folder (and subfolders) that contain the given substring in their names.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    deleted_count = 0
    for root, dirs, files in os.walk(folder_path):  # Traverse all subfolders
        for filename in files:
            if substring in filename:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except PermissionError:
                    print(f"Permission denied: {file_path} (file may be in use)")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print(f"Total deleted: {deleted_count}")

# Example usage
folder_path = r"C:\Users\micha\Documents\School\Senior\Spring\CSE 4940 Senior Design\SeniorDesign Codebase\SDdata"
substring = "Category"  # Case-sensitive match
delete_files_by_substring(folder_path, substring)