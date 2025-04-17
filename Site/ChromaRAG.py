import chromadb
import os
import ollama
import json
from html.parser import HTMLParser

class ChromaRAG():
    def __init__(self, doc_path="./docs", collection_path="./collection", template_path="./templates", model_name="deepseek-r1:7b", ollama_url=None):
        self.doc_path = doc_path
        self.model_name = model_name
        self.collection_path = collection_path
        self.template_path = template_path
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

    def add_note_to_RAG(self, file_path):
        if (file_path[-3:] == ".md" or file_path[-4:] == ".txt"):
                    # Add the file to the collection
                    doc = [open(file_path, "r").read()]
                
        if file_path[-5:] == ".html":
            html_parse = html_ripper()
            html_parse.feed(open(file_path, "r").read())
            doc = html_parse.get_data()

        self.collection.add(documents=doc,ids=[file_path.split("/")[-1]])


    def query_notes(self, in_prompt):
        """Method for querying the notes to get information about something
            such as a character or location, mostly used for testing RAG
                prompt = <string> Input of prompt to send to the model
                
                returns the output from the model
        """

        collection_results = self.collection.query(query_texts=[in_prompt], n_results = 3)
        
        context = context_to_string(collection_results["documents"][0], collection_results["ids"][0])

        prompt = self.format_prompt(in_prompt, context, "query")

        response = ollama.generate(model=MODEL_NAME, prompt=prompt)['response'].split("</think>")
        if len(response) > 1:
            return response[1]
        else:
            return response[0]


    

    def generate_character(self, prompt_dict, use_url = None):
        """Dictionary contains:
                person_name
                person_home
                profession
                faction
                relationships
                additional_info
        """
        prompt = "Generate a character "
        if prompt_dict['person_name'] != "":
            prompt += "named " + prompt_dict['person_name'] + " "

        if prompt_dict['person_home'] != "":
            prompt += "from " + prompt_dict['person_home'] + " "

        if prompt_dict['profession'] != "":
            prompt += "who is a " + prompt_dict['profession'] + " "

        if prompt_dict['faction'] != "":
            prompt += "and is a part of " + prompt_dict['faction'] + " "

        if prompt_dict['relationships'] != "":
            prompt += "\nIn addition the character knows of:\n" + prompt_dict['relationships']

        if prompt_dict['additional_info'] != "":
            prompt += "\nThe character also:\n" + prompt_dict['additional_info']

        

        collection_results = self.collection.query(query_texts=[prompt], n_results = 1)
        
        context = context_to_string(collection_results["documents"][0], collection_results["ids"][0])

        context = "###Context\n\n" + context
        
        template = open(self.template_path + "/" + "character" + ".template").read()
        #prompt = self.format_prompt(prompt, context, "character")
        
        if use_url:
            client = ollama.Client(host=use_url)
        else:
            client = ollama.Client()

        response = client.chat(model=self.model_name, messages=[{'role':'system', 'content': template + context }, {'role':'user', 'content':prompt}])

               
        response = response['message']['content'].split("</think>")

        if len(response) > 1:
            return response[1]
        else:
            return response[0]



    def generate_location(self, prompt_dict, use_url=None):
        """Dictionary contains:
                place_name
                nearby
                faction
                location_type
                sub_locations
                additional_info
        """
        prompt = "Generate a place "
        if prompt_dict['place_name'] != "":
            prompt += "named " + prompt_dict['place_name'] + " "

        if prompt_dict['nearby'] != "":
            prompt += "near " + prompt_dict['nearby'] + " "

        if prompt_dict['location_type'] != "":
            prompt += "which is a " + prompt_dict['location_type'] + " "

        if prompt_dict['faction'] != "":
            prompt += "ruled by " + prompt_dict['faction'] + " "

        if prompt_dict['sub_locations'] != "":
            prompt += "\nIn addition within this place there are:\n" + prompt_dict['sub_locations']

        if prompt_dict['additional_info'] != "":
            prompt += "\nThe place also:\n" + prompt_dict['additional_info']

        

        collection_results = self.collection.query(query_texts=[prompt], n_results = 3)
        
        context = context_to_string(collection_results["documents"][0], collection_results["ids"][0])

        context = "###Context\n\n" + context
        
        #prompt = self.format_prompt(prompt, context, "location")

        template = open(self.template_path + "/" + "location" + ".template").read()
        
        if use_url:
            client = ollama.Client(host=use_url)
        else:
            client = ollama.Client()

        response = client.chat(model=self.model_name, messages=[{'role':'system', 'content': template + context }, {'role':'user', 'content':prompt}])

               
        response = response['message']['content'].split("</think>")

        if len(response) > 1:
            return response[1]
        else:
            return response[0]

    def generate_campaign(self, prompt_dict, use_url=None):
        """Generates a full DnD campaign using all wiki entries and a structured HTML template.
            
        """
        
        prompt = "Create a story about:\n\n"+ prompt_dict["prompt"]
        collection_results = self.collection.query(query_texts=[prompt], n_results = 5)
        template = open(self.template_path + "/" + "campaign.template").read()

        messages = [ {'role':'system', 'content': template}]
        for context in context_to_list(collection_results["documents"][0], collection_results["ids"][0]):
            messages.append({'role':'system', 'content':context})
        messages.append({'role':'user', 'content':prompt})

        if use_url:
            client = ollama.Client(host=use_url)
        else:
            client = ollama.Client()

        response = client.chat(model=self.model_name, messages=messages)
        
        response = response['message']['content'].split("</think>")

        if len(response) > 1:
            return response[1]
        else:
            return response[0]


    def format_prompt(self, in_prompt, context, template):
        template = open(self.template_path + "/" + template + ".template").read()
        return template + context + "###Query\n\n" + in_prompt




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

def context_to_list(documents, ids):
    res = []
    for x in range(len(ids)):
        res.append("###Context " + str(x) + ": " + ids[x] + "\n\n\n")
        res.append(documents[x])
    return res

MODEL_NAME = "deepseek-r1:7b"
TEMPLATE_PATH = "./templates"
DOC_PATH = "./test_docs"
COLLECTION_NAME = "test_collection"

if __name__ == "__main__":
   
    RAG = ChromaRAG("./data/wiki_entries")

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

            print(RAG.generate_character(test_char_dict, None))
        else:
            print(RAG.query_notes(user_input))

        print("\n")
        user_input = input("Enter prompt here: ")