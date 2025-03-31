import chromadb
import os
import ollama

class ChromaRAG():
    def __init__(self, doc_path="./docs", collection_path="./collection", template_path="./templates", model_name="deepseek-r1:7b"):
        self.doc_path = doc_path
        self.template_path = template_path
        self.model_name = model_name

        self.collection = chroma.get_or_create_collection(name = collection_path)
        add_to_collection(self.collection, doc_path)

   # def 



def add_to_collection(collection, path):
    # Find all of the documents in the collection but not in the file location and remove them
    # from the collection (i.e. the file has been deleted and should no longer be referenced)
    in_dir = _add_to_collection(collection, path)
    missing_in_dir = [x for x in collection.get(include=["uris"])["ids"] if x not in in_dir]
    if missing_in_dir:
        print("Removing " + str(missing_in_dir))
        collection.delete(missing_in_dir)

def _add_to_collection(collection, path):
    documents = list()
    ids = list()
    in_dir = [x for x in os.listdir(path) if x[-3:]==".md" or x[-4:]==".txt"]

    for files in os.listdir(path):
        # Explore all files and add .md and .txt files to the collection
        if (files[-3:] == ".md" or files[-4:] == ".txt"): #and collection.get(files)["ids"] == []:
            ids.append(files)
            documents.append(open(path + "/" + files, "r").read())
        
        # Use recursion to explore every subfolder
        if os.path.isdir(path+"/"+files):
            in_dir.extend(_add_to_collection(collection, path+"/"+files))

    if ids:
        collection.add(documents=documents, ids=ids)

    return in_dir

def context_to_string(documents, ids):
    res = ""
    for x in range(len(ids)):
        res += "[" + ids[x] + "]:\n\n\"\"\""
        res += documents[x]
        res += "\"\"\"\n\n"
    return res

def generate_prompt(collection, input_prompt, template = None):
    prompt = """###Task: 

    Respond to the user query using the provided context, incorporating inline citations in the format [name_of_source] where name_of_source is the title of the source the citation is from. 

###Guidelines: 

    If you don't know the answer, clearly state that. 

    If uncertain, ask the user for clarification. 

    If the context is unreadable or of poor quality, inform the user and provide the best possible answer. 

    If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding. 

    Only include inline citations using [source_id] when a tag is explicitly provided in the context.  

    Do not cite if the tag is not provided in the context.  

    Ensure citations are concise and directly related to the information provided. 

###Example of Citation: 

    If the user asks about a specific topic and the information is found in \"whitepaper.pdf\" with a provided , the response should include the citation like so:  

    \"According to the study, the proposed method increases efficiency by 20% [whitepaper.pdf].\" If no context is present, the response should omit the citation. 

###Output: 

    Provide a clear and direct response to the user's query, including inline citations in the format [source_id] only when the tag is present in the context. 


"""
    collection_results = collection.query(query_texts=[input_prompt], n_results = 3)
    context = context_to_string(collection_results["documents"][0], collection_results["ids"][0])
    if template:
        template_prompt = "###Template\n\nFormat your response using the following template\n\n"
        if template == "character":
            template_prompt += open(TEMPLATE_PATH + "/character.template", "r").read()
        
        prompt += template_prompt

    if context:
        prompt += "###Context\n\n" + context
    prompt += "###Query\n\n" + input_prompt
    return prompt

def query_model_chat(collection, input_prompt):
    return ollama.chat(model = MODEL_NAME, messages = [{'role': 'user', 'content':generate_prompt(collection, input_prompt)}], stream=True)

def query_model_oneshot(collection, input_prompt):
    return ollama.generate(model=MODEL_NAME, prompt=generate_prompt(collection, input_prompt))['response']

def query_model_chat_cleaned(collection, input_prompt, include_think=False):
    past_think = False
    for chunk in ollama.chat(model = MODEL_NAME, messages = [{'role': 'user', 'content':generate_prompt(collection, input_prompt)}], stream=True):
        if include_think or past_think:
            yield chunk['message']['content']
        else:
            if "</think>" in chunk['message']['content']: past_think = True


MODEL_NAME = "deepseek-r1:7b"
TEMPLATE_PATH = "./templates"
DOC_PATH = "./test_docs"
COLLECTION_NAME = "test_collection"

chroma = chromadb.PersistentClient(path=DOC_PATH)

collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

add_to_collection(collection, DOC_PATH)



#for chunk in query_model_chat_cleaned(collection, "who is luminos"):
#    print(chunk)

if __name__ == "__main__":
    user_input = input("Enter prompt here: ")
    while user_input != "exit":
        for chunk in query_model_chat_cleaned(collection, user_input):
            print(chunk, end='', flush=True)
        
        print("\n")
        user_input = input("Enter prompt here: ") 
