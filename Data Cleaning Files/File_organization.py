"""This File takes the results of 'File Spilter' and sorts them into folders based off the context of '{{Keyword' which most files follow"""
import os
import shutil

#Define where to find and store the results as well as the keywords that the program will be using
SOURCE_DIR = r""
BASE_DIR = r""

KEYWORDS = ["{{Item", 
            "{{Organization", 
            "{{Person", 
            "{{Spell", 
            "{{Location", 
            "{{Creature", 
            "{{BGitem", 
            "{{Book", 
            "{{Building", 
            "{{Computer game", 
            "{{Game", 
            "{{Roll of years", 
            "{{Organized play", 
            "{{Class", 
            "{{Deity", 
            "{{Real-world person", 
            "{{Substance", 
            "{{Mountain", 
            "{{Celestial body", 
            "{{Ethnicity", 
            "{{Road", 
            "{{Language", 
            "{{Body of water",
            "{{Comic",
            "{{roll of years",
            "{{Adventurers league",
            "{{Plane",
            "{{Plant",
            "{{State",
            "{{Dungeon",
            "{{Ship",
            "{{Disease",
            "{{building",
            "{{Conflict",
            "{{Portal",
            "{{Event",
            "{{MTG",
            "{{person",
            "{{Primordial",
            "{{Family tree",
            "{{Fungus",
            "{{Channel divinity",
            "{{Dragon magazine",
            "{{Vegetation/Plant",
            "{{Polyhedron magazine",
            "{{Roll of months",
            "{{book",
            "{{Dragon+"]

#Defines a Unprocessed directories to store everything that doesnt fit within the keywords
UNPROCESSED_DIR = os.path.join(BASE_DIR, "unprocessed") 

#Ensure all necessary directories exist in BASE_DIR
os.makedirs(UNPROCESSED_DIR, exist_ok=True)
for keyword in KEYWORDS:
    os.makedirs(os.path.join(BASE_DIR, keyword), exist_ok=True)

#Process each file in the source directory
for filename in os.listdir(SOURCE_DIR):
    file_path = os.path.join(SOURCE_DIR, filename)

    if not os.path.isfile(file_path):
        continue  # Skip directories
    
    moved = False  # Track if the file has been moved

    try:
        # Read file content and close it before moving
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()  # Read entire content into memory

        # Now process the content after the file is closed
        for keyword in KEYWORDS:
            if keyword in content:
                destination = os.path.join(BASE_DIR, keyword, filename)
                shutil.move(file_path, destination)
                print(f"Moved {filename} to {keyword}/ in {BASE_DIR}")
                moved = True
                break  # Stop searching once a keyword is found

        #If no keyword was found, move the file to the unprocessed folder
        if not moved:
            destination = os.path.join(UNPROCESSED_DIR, filename)
            shutil.move(file_path, destination)
            print(f"Moved {filename} to unprocessed/ in {BASE_DIR}")

    except PermissionError:
        print(f"PermissionError: Unable to move {filename}. It might be open in another program.")
