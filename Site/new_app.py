from flask import Flask, request, render_template, redirect
import requests
import os
import re
from ChromaRAG import ChromaRAG

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ensure Ollama is running
GPU_URL = "http://137.99.227.35:11434"
MODEL_NAME = "deepseek-r1:1.5b"
WIKI_FILE = "data/ollama_wiki.html"
docPath = "./data/wiki_entries"

COLLECTION_NAME = "./collection"  # Must follow ChromaDB rules
Chroma = ChromaRAG(docPath, COLLECTION_NAME)

def ask_deepseek(prompt, use_gpu_server=False):
    """Sends the prompt to DeepSeek and returns a cleaned response."""
    url = "http://137.99.227.35:11434/api/generate" if use_gpu_server else OLLAMA_URL
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        raw = response.json().get("response", "").strip()
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        return clean_response(raw)

    return f"Error: {response.status_code} - {response.text}"


def clean_response(text):
    """Removes unwanted HTML tags while keeping essential formatting."""
    text = re.sub(r"<(?!p|br|strong|em|h\d)[^>]+>", "", text)  # Keeps <p>, <br>, etc.
    return text.strip()


def character_creation_prompt(character_data):
    """Generates a structured prompt for AI to ensure clarity."""
    return (
        "Create a well-organized DnD character with the following structured sections:\n\n"
        f"Name: {character_data.get('person_name', 'Unknown')}\n"
        f"Homeland: {character_data.get('person_home', 'Unknown')}\n"
        f"Class/Profession: {character_data.get('profession', 'Unknown')}\n"
        f"Guild or Faction: {character_data.get('faction', 'None')}\n"
        f"Allies & Rivals: {character_data.get('relationships', 'None')}\n"
        f"Backstory & Traits: {character_data.get('additional_info', 'None')}\n\n"
        "Format the response with **HTML-friendly headings** as follows:\n"
        "1. **<h3>Introduction</h3>**\n"
        "   - A short overview of the character.\n"
        "2. **<h3>Abilities & Skills</h3>**\n"
        "   - Bullet points listing unique abilities, spells, and talents.\n"
        "3. **<h3>Allies & Rivals</h3>**\n"
        "   - Mention key allies and rival characters with a short description.\n"
        "4. **<h3>Backstory</h3>**\n"
        "   - 2-3 paragraphs detailing their history and character motivations.\n"
        "Ensure the text is structured and easy to read."
    )


def town_creation_prompt(town_data):
    """Generates a structured prompt for AI to ensure clarity."""
    return (
        "Create a fantasy town with structured details:\n\n"
        f"Town Name: {town_data.get('place_name', 'Unknown')}\n"
        f"Nearby Landmarks: {town_data.get('nearby', 'Unknown')}\n"
        f"Ruling Faction: {town_data.get('faction', 'Unknown')}\n"
        f"Type of Settlement: {town_data.get('location_type', 'Unknown')}\n"
        f"Notable Places: {town_data.get('sub_locations', 'None')}\n"
        f"History & Lore: {town_data.get('additional_info', 'None')}\n\n"
        "Format the response as follows:\n"
        "1. **Overview**: Briefly describe the town.\n"
        "2. **Key Features**: List unique aspects.\n"
        "3. **Notable Locations**: Describe important places.\n"
        "4. **History & Lore**: Provide a background story.\n"
        "Ensure readability by using sections and avoid dense text."
    )

def format_wiki_entry(title, content, category):
    """Formats the wiki entry with structured HTML for clarity."""
    return f"""
    <div class="wiki-entry {'character-entry' if category == 'Character' else 'town-entry'}">
        <h2>{title}</h2>
        <h3>{'Character Profile' if category == 'Character' else 'Town Overview'}</h3>
        <div class="wiki-content">
            {content}
        </div>
        <hr>
        <a href="/wiki">Back to Wiki</a>
    </div>
    """


def update_wiki(title, content, category):
    """Saves each entry as its own HTML file and updates the main wiki index."""
    wiki_dir = "data/wiki_entries/"
    os.makedirs(wiki_dir, exist_ok=True)  # Ensure the directory exists

    # Generate filename-friendly title (replace spaces with underscores)
    filename = title.replace(" ", "_") + ".html"
    file_path = os.path.join(wiki_dir, filename)

    # Save the detailed entry as its own page
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head><title>{title}</title></head>
        <body>
        <h2>{title}</h2>
        <h3>{'Character Profile' if category == 'Character' else 'Town Overview'}</h3>
        <div class="wiki-container">
            <div class="wiki-content">{content}</div>
        </div>
        <hr>
        <a href="/wiki">Back to Wiki</a>
        </body>
        </html>
        """)

    # Update the main index with a hyperlink to this entry
    index_path = "data/wiki_index.html"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(f'<li><a href="/wiki_entry/{filename}">{title} ({category})</a></li>\n')

    return filename

def update_location_wiki(place_name, content):
    """Saves a location entry as its own HTML file with proper title and structure."""
    wiki_dir = "data/wiki_entries/"
    os.makedirs(wiki_dir, exist_ok=True)

    # Generate a filename-safe version
    filename = place_name.replace(" ", "_") + ".html"
    file_path = os.path.join(wiki_dir, filename)

    # Save the entry with the correct <title> and content
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head><title>{place_name}</title></head>
        <body>
        <div class="wiki-container">
            <h2>{place_name}</h2>
            <h3>Town Overview</h3>
            <div class="wiki-content">{content}</div>
            <hr>
            <a href="/wiki">Back to Wiki</a>
        </div>
        </body>
        </html>
        """)

    # Update the main wiki index
    index_path = "data/wiki_index.html"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(f'<li><a href="/wiki_entry/{filename}">{place_name} (Location)</a></li>\n')

def update_campaign_wiki(campaign_title, content):
    """Saves a campaign entry as its own HTML file and updates the wiki index."""
    wiki_dir = "data/wiki_entries/"
    os.makedirs(wiki_dir, exist_ok=True)

    filename = campaign_title.replace(" ", "_") + ".html"
    file_path = os.path.join(wiki_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head><title>{campaign_title}</title></head>
        <body>
        <div class="wiki-container">
            <h2>{campaign_title}</h2>
            <h3>Campaign Overview</h3>
            <div class="wiki-content">{content}</div>
            <hr>
            <a href="/wiki">Back to Wiki</a>
        </div>
        </body>
        </html>
        """)

    index_path = "data/wiki_index.html"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(f'<li><a href="/wiki_entry/{filename}">{campaign_title} (Campaign)</a></li>\n')




def reset_wiki():
    """Creates an empty wiki file if it doesn't exist"""
    with open(WIKI_FILE, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Ollama Wiki</title></head><body><h1>Ollama Wiki</h1>\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/character_creation', methods=['GET', 'POST'])
def character_creation():
    if request.method == 'POST':
        character_data = request.form.to_dict()
        #use_gpu = character_data.get("ollama_source") == "gpu"
        #prompt = character_creation_prompt(character_data)
        #generated_character = ask_deepseek(prompt, use_gpu_server=use_gpu)
        if character_data.get("ollama_source") == "gpu":
            url = GPU_URL
        else:
            url = None
        generated_character = Chroma.generate_character(character_data, url)
        filename = update_wiki(character_data.get("person_name", "Unnamed Character"), generated_character, "Character")
        Chroma.add_note_to_RAG("./data/wiki_entries/"+filename)
        return redirect('/wiki')
    return render_template('person_form.html', custom_style='css/creation.css')

@app.route('/town_creation', methods=['GET', 'POST'])
def town_creation():
    if request.method == 'POST':
        location_data = request.form.to_dict()
        use_gpu = location_data.get("ollama_source") == "gpu"
        url = GPU_URL if use_gpu else None

        # Generate raw location content
        generated_location = Chroma.generate_location(location_data, url)

        # Fill in {{Location Name}}
        location_name = location_data.get("place_name", "Unnamed Location")
        generated_location = generated_location.replace("{{Location Name}}", location_name)

        # Convert **Title:** sections into HTML <p><strong>Title:</strong> ...</p>
        formatted_location = format_location_output(generated_location)

        # Save to file
        filename = location_name.replace(" ", "_") + ".html"
        update_location_wiki(location_name, formatted_location)

        # Add to RAG
        Chroma.add_note_to_RAG(f"./data/wiki_entries/{filename}")

        return redirect('/wiki')

    return render_template('place_form.html', custom_style='css/creation.css')


def format_location_output(raw_text):
    """Convert markdown-style bold headers into HTML paragraph blocks."""
    # Break into sections based on **Title:**
    parts = re.split(r"\*\*(.*?)\*\*:", raw_text)
    
    # Start building formatted HTML
    formatted = ""
    
    # If the first part before any **Title:** is non-empty, add it
    if parts[0].strip():
        formatted += f"<p>{parts[0].strip()}</p>"

    # Then pair each Title/Body into a <p><strong>Title:</strong> Body</p>
    for i in range(1, len(parts)-1, 2):
        title = parts[i].strip()
        body = parts[i+1].strip()
        formatted += f"<p><strong>{title}:</strong> {body}</p>\n"

    return formatted.strip()



@app.route('/generate_story', methods=['GET', 'POST'])
def generate_story():
    if request.method == 'POST':
        prompt_text = request.form.get('prompt', 'A mysterious event in a fantasy world')
        use_gpu = request.form.get("ollama_source") == "gpu"
        url = GPU_URL if use_gpu else None

        story = Chroma.generate_campaign({"prompt": prompt_text}, use_url=url)

        # Extract actual title from the story body (e.g., **Title: NAME**)
        # In /generate_story route:
        title_match = re.search(r"[#]+\s*\*\*Title:\*\*\s*(?:\*{0,2})(.*?)(?:\*{0,2})\s*(?:#*|$)", story)
        campaign_title = title_match.group(1).strip() if title_match else "Unnamed Campaign"


        # Remove title line from content if found
        if title_match:
            story = story.replace(title_match.group(0), "").strip()

        # Improve spacing between acts
        story = re.sub(r"---\s*", "<br><br>", story)

        update_campaign_wiki(campaign_title, story)
        return redirect('/wiki')

    return render_template("story_prompt.html")




@app.route('/wiki')
def view_wiki():
    index_path = "data/wiki_index.html"
    
    if not os.path.exists(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("<html><head><title>Wiki</title></head><body><h1>Wiki</h1>\n<ul>\n</ul></body></html>")

    with open(index_path, "r", encoding="utf-8") as f:
        wiki_index = f.read()

    CSS_STYLE = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 20px;
            padding: 20px;
        }
        .wiki-container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            padding: 8px;
            font-size: 18px;
        }
        a {
            text-decoration: none;
            color: #007bff;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """

    return f"""
    <html>
    <head>
        <title>Wiki</title>
        {CSS_STYLE}
    </head>
    <body>
        <div class="wiki-container">
            <h1>Ollama Wiki</h1>
            <p>Click an entry below to view details:</p>
            <ul>
                {wiki_index}
            </ul>
            <form action="/generate_story" method="get">
            <button type="submit" style="...">Generate Story</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.route('/wiki_entry/<entry_name>')
def view_wiki_entry(entry_name):
    """Loads and displays a specific wiki entry with clean formatting"""
    wiki_dir = "data/wiki_entries/"
    file_path = os.path.join(wiki_dir, entry_name)  # Securely join the path
    
    if not os.path.exists(file_path):
        return "<h1>Entry Not Found</h1><a href='/wiki' class='back-link'>Back to Wiki</a>"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # CSS for clean formatting
    CSS_STYLE = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 20px;
            padding: 20px;
        }
        .wiki-container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #333;
            font-weight: bold;
        }
        h3 {
            margin-top: 20px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        .wiki-content {
            line-height: 1.6;
        }
        .wiki-content ul {
            padding-left: 20px;
        }
        .wiki-content li {
            margin-bottom: 8px;
        }
        .wiki-content p {
            margin-bottom: 10px;
        }
        .back-link {
            display: block;
            margin-top: 20px;
            text-decoration: none;
            color: #007bff;
            font-size: 18px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
    </style>
    """

    return f"""
    <html>
    <head>
        <title>{entry_name.replace("_", " ").replace(".html", "")}</title>
        {CSS_STYLE}  <!-- Inject the improved CSS -->
    </head>
    <body>
        <div class="wiki-container">
            {content}
        </div>
    </body>
    </html>
    """



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4990, debug=True)
