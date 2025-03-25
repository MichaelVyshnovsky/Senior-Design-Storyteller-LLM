import os
import shutil

def search_and_delete_files(folder_path, keyword):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            try:
                # Open and read the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Check if the keyword is in the file content
                if keyword in content:
                    print(f"Keyword '{keyword}' found in file: {filename}")
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted file: {filename}")
            except Exception as e:
                print(f"Error reading or deleting file {filename}: {e}")

def search_and_move_files(folder_path, keyword, destination_folder):
    # Check if the source folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            try:
                # Open and read the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Check if the keyword is in the file content
                if keyword in content:
                    print(f"Keyword '{keyword}' found in file: {filename}")
                    
                    # Move the file to the destination folder
                    shutil.move(file_path, os.path.join(destination_folder, filename))
                    print(f"Moved file: {filename} to {destination_folder}")
            except Exception as e:
                print(f"Error reading or moving file {filename}: {e}")

if __name__ == "__main__":
    folder_path = r""  # Replace with your folder path
    destination_folder = r""  # Replace with your destination folder
    keyword = "magical abilities"
    search_and_move_files(folder_path, keyword, destination_folder)
    #search_and_delete_files(folder_path, keyword)
