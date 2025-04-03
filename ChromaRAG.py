import chromadb
import os
import ollama
import json

class ChromaRAG():
    def __init__(self, doc_path="./docs", collection_path="./collection", model_name="deepseek-r1:7b"):
        self.doc_path = doc_path
        self.model_name = model_name
        self.collection_path = collection_path

        self.chroma = chromadb.PersistentClient(path=collection_path)
        
        self.collection = self.chroma.get_or_create_collection(name = "collection")
        self.add_to_collection()

    def add_to_collection(self):
        def _add_to_collection(path, time_dict):
            docs = list()
            ids = list()

            for files in os.listdir(path):
                # Explore all files and add .md and .txt files to the collection and are new or have been modified
                if (files[-3:] == ".md" or files[-4:] == ".txt") and time_dict.get(path+files, None) != os.stat(path+files).st_mtime:
                    # Add the file to the collection
                    ids.append(files)
                    docs.append(open(path + files, "r").read())
                    time_dict[path+files] = os.stat(path+files).st_mtime # Save last modified time to time dict
                
                if files[-5:] == ".html" and time_dict.get(path+files, None) != os.stat(path+files).st_mtime:
                    ids.append(files)
                    html_parse = html_ripper()
                    html_parse.feed(open(path + files, "r").read())
                    docs.append(html_parse.get_data())
                    time_dict[path+files] = os.stat(path+files).st_mtime

                # Use recursion to explore every subfolder
                if os.path.isdir(path+files):
                    sub_docs, sub_ids, time_dict = _add_to_collection(path + files + "/", time_dict)

                    docs += sub_docs
                    ids += sub_ids

            
            return docs, ids, time_dict

        # Load timestamp dictionary
        if os.path.exists(self.collection_path + "/timestamp.dat"):
            with open(self.collection_path + "/timestamp.dat", 'r') as f:
                time_dict = json.load(f)
        else:
            time_dict = dict()


        docs, ids, time_dict = _add_to_collection(self.doc_path + "/", time_dict)

        if ids:
            self.collection.add(documents=docs, ids=ids)

        # Save timestamp dictionary
        with open(self.collection_path + "/timestamp.dat", 'w') as f:
            json.dump(time_dict, f)

    def query_notes(self, in_prompt):
        """Method for querying the notes to get information about something
            such as a character or location, mostly used for testing RAG
                prompt = <string> Input of prompt to send to the model
                
                returns the output from the model
        """

        prompt = """###Task: 

    Respond to the user query using the provided context, incorporating inline citations in the format [name_of_source] where name_of_source is the title of the source the citation is from. 

###Guidelines: 

    If you don't know the answer, clearly state that. 

    If uncertain, ask the user for clarification. 

    If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding. 

    Cite sources using the filename before the context

    Ensure citations are concise and directly related to the information provided. 

###Example of Citation: 

    If the user asks about a specific topic and the information is found in \"whitepaper.pdf\" with a provided , the response should include the citation like so:  

    \"According to the study, the proposed method increases efficiency by 20% [whitepaper.pdf].\" If no context is present, the response should omit the citation. 
"""

        collection_results = self.collection.query(query_texts=[in_prompt], n_results = 3)
        
        context = context_to_string(collection_results["documents"][0], collection_results["ids"][0])

        if context:
            prompt += "###Context\n\n" + context
        prompt += "###Query\n\n" + in_prompt

        response = ollama.generate(model=MODEL_NAME, prompt=prompt)['response'].split("</think>")
        if len(response) > 1:
            return response[1]
        else:
            return response[0]


    

    def generate_character(self, prompt_dict):
        """Dictionary contains:
                person_name
                person_home
                profession
                faction
                relationships
                additional_info
        """
        prompt = "Generate a character "
        if prompt_dict['person_name']:
            prompt += "named " + prompt_dict['person_name'] + " "

        if prompt_dict['person_home']:
            prompt += "from " + prompt_dict['person_home'] + " "

        if prompt_dict['profession']:
            prompt += "who is a " + prompt_dict['profession'] + " "

        if prompt_dict['faction']:
            prompt += "and is a part of " + prompt_dict['faction'] + " "

        if prompt_dict['relationships']:
            prompt += "\nIn addition the character knows of:\n" + prompt_dict['relationships']

        if prompt_dict['additional_info']:
            prompt += "\nThe character also:\n" + prompt_dict['additional_info']

        

        collection_results = self.collection.query(query_texts=[prompt], n_results = 3)
        
        context = context_to_string(collection_results["documents"][0], collection_results["ids"][0])

        context = "###Context\n\n" + context
        
        prompt = format_character_prompt(prompt, context)

        

        return ollama.generate(self.model_name, prompt)['response']

class html_ripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.data = ""

    def handle_data(self, data):
        self.data += data

    def get_data(self):
        return self.data

def context_to_string(documents, ids):
    res = ""
    for x in range(len(ids)):
        res += "[" + ids[x] + "]:\n\n\"\"\""
        res += documents[x]
        res += "\"\"\"\n\n"
    return res

def format_character_prompt(in_prompt, context):
    return """###Task:

    Generate a character for a fantasy world described by the given context under the conditions provided by the user in query with an output formatted the same way as the Template

###Guidelines: 

    Connect the generated content to the given context if possible

    Keep content to a generic fantasy style

###Template:

\"\"\"
Name
---
[Name of the character]

Description
---
[Brief overview of the character]

Skills
---
[Things the character is good or bad at]

Personality
---
[The personality of the character]

Backstory
---
[A backstory of where the character comes from]

Relationships
---
[Relationships with other characters and factions in the world]

\"\"\"

###Query

""" + context + in_prompt 


MODEL_NAME = "deepseek-r1:7b"
TEMPLATE_PATH = "./templates"
DOC_PATH = "./test_docs"
COLLECTION_NAME = "test_collection"

if __name__ == "__main__":
   
    RAG = ChromaRAG()

    user_input = input("Enter prompt here: ")

    while user_input != "exit":
        if user_input == "test1":
            test_char_dict = {
                "person_name" : "Big man",
                "person_home" : "Kasei",
                "profession" : "Explorer",
                "faction" : "",
                "relationships" : "Allies with Hirroko",
                "additional_info" : "Was created by Akreor"}

            print(RAG.generate_character(test_char_dict))
        else:
            print(RAG.query_notes(user_input))

        print("\n")
        user_input = input("Enter prompt here: ") 
