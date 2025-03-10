from flask import Flask, request, jsonify
import requests
import os
import datetime
import re

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ensure Ollama is running
MODEL_NAME = "deepseek-r1:7b"  # Change to your desired model
WIKI_FILE = "ollama_wiki.html"

def ask_ollama(prompt):
    """Sends a query to Ollama and retrieves the response."""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        answer = response.json().get("response", "").strip()
        return clean_response(answer)
    return f"Error: {response.status_code} - {response.text}"

def clean_response(text):
    """Removes unnecessary HTML-like tags from the response."""
    return re.sub(r"<.*?>", "", text).strip()

def format_wiki_entry(question, answer):
    """Formats the response into an HTML wiki entry."""
    return f"<h2>{question}</h2>\n<p>{answer}</p>\n<hr>\n"

def update_wiki(question, answer):
    """Writes the formatted response to the wiki HTML file."""
    entry = format_wiki_entry(question, answer)
    
    # Create file if it doesn't exist
    if not os.path.exists(WIKI_FILE):
        reset_wiki()
    
    with open(WIKI_FILE, "a") as f:
        f.write(entry)
    
    return "✅ Wiki updated"

def reset_wiki():
    """Resets the HTML wiki file."""
    with open(WIKI_FILE, "w") as f:
        f.write("<html><head><title>Ollama Wiki</title></head><body><h1>Ollama Wiki</h1>\n")
    return "✅ Wiki reset"

@app.route('/ask', methods=['POST'])
def ask():
    """Flask route to ask Ollama and store response in the wiki."""
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    answer = ask_ollama(question)
    update_wiki(question, answer)
    
    return jsonify({"question": question, "answer": answer})

@app.route('/wiki', methods=['GET'])
def get_wiki():
    """Flask route to retrieve the current wiki contents as HTML."""
    if not os.path.exists(WIKI_FILE):
        return jsonify({"message": "Wiki file is empty or does not exist."}), 404
    
    with open(WIKI_FILE, "r") as f:
        wiki_content = f.read()
    
    return wiki_content, 200, {'Content-Type': 'text/html'}

@app.route('/reset_wiki', methods=['POST'])
def reset_wiki_route():
    """Flask route to reset the HTML wiki file."""
    reset_wiki()
    return jsonify({"message": "Wiki has been reset."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4990, debug=True)
