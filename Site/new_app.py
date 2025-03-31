from flask import Flask, request, jsonify, render_template
import requests
import os
import re

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ensure Ollama is running
MODEL_NAME = "deepseek-r1:7b"
WIKI_FILE = "data/ollama_wiki.html"

def ask_deepseek(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return clean_response(response.json().get("response", "").strip())
    return f"Error: {response.status_code} - {response.text}"

def clean_response(text):
    return re.sub(r"<.*?>", "", text).strip()

def format_wiki_entry(question, answer):
    return f"<h2>{question}</h2>\n<p>{answer}</p>\n<hr>\n"

def update_wiki(question, answer):
    entry = format_wiki_entry(question, answer)
    if not os.path.exists(WIKI_FILE):
        reset_wiki()
    with open(WIKI_FILE, "a") as f:
        f.write(entry)

def reset_wiki():
    with open(WIKI_FILE, "w") as f:
        f.write("<html><head><title>Ollama Wiki</title></head><body><h1>Ollama Wiki</h1>\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_page', endpoint='ask_page')
def serve_ask_page():
    return render_template('ask.html')

@app.route('/character_creation', methods=['GET', 'POST'])
def character_creation():
    if request.method == 'POST':
        character_data = request.form.to_dict()
        prompt = ("Create a detailed DnD character with the following attributes:\n"
                  f"Name: {character_data.get('person_name', 'Unknown')}\n"
                  f"Homeland: {character_data.get('person_home', 'Unknown')}\n"
                  f"Class/Profession: {character_data.get('profession', 'Unknown')}\n"
                  f"Guild or Faction: {character_data.get('faction', 'None')}\n"
                  f"Allies & Rivals: {character_data.get('relationships', 'None')}\n"
                  f"Backstory & Traits: {character_data.get('additional_info', 'None')}\n"
                  "Please generate a compelling backstory and detailed traits for this character.")
        generated_character = ask_deepseek(prompt)
        return render_template('results.html', generated_content=generated_character, custom_style='css/creation.css')
    return render_template('person_form.html', custom_style='css/creation.css')

@app.route('/town_creation', methods=['GET', 'POST'])
def town_creation():
    if request.method == 'POST':
        town_data = request.form.to_dict()
        prompt = ("Generate a detailed fantasy town description using these attributes:\n"
                  f"Town Name: {town_data.get('place_name', 'Unknown')}\n"
                  f"Nearby Landmarks: {town_data.get('nearby', 'Unknown')}\n"
                  f"Ruling Faction: {town_data.get('faction', 'Unknown')}\n"
                  f"Type of Settlement: {town_data.get('location_type', 'Unknown')}\n"
                  f"Notable Places: {town_data.get('sub_locations', 'None')}\n"
                  f"History & Lore: {town_data.get('additional_info', 'None')}\n"
                  "Please generate a rich and immersive description of this town, its people, and its history.")
        generated_town = ask_deepseek(prompt)
        return render_template('results.html', generated_content=generated_town, custom_style='css/creation.css')
    return render_template('place_form.html', custom_style='css/creation.css')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4990, debug=True)