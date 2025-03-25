import os

def delete_files_by_substring(folder_path, substring):
    """
    Deletes files in the specified folder that contain the given substring in their names.
    
    :param folder_path: Path to the folder containing the files.
    :param substring: Substring to search for in file names.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        if substring in filename:  # Check if the substring is anywhere in the file name
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Example usage
folder_path = r""  # Replace with your folder path
substring = "Module_"  # Replace with the substring you want to target
delete_files_by_substring(folder_path, substring)
