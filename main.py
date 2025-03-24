import os
import openai
import time
from flask import Flask, render_template, request, session
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session handling

# Initialize OpenAI client with SambaNova credentials
client = openai.OpenAI(
    api_key="726c299f-664e-451d-8316-94e68e232718",
    base_url="https://api.sambanova.ai/v1",
)

# Prompt template for parsing
template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

# Function to scrape website
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return str(e)

# Extract body content from HTML
def extract_body_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.body.get_text(separator=' ') if soup.body else ""

# Clean extracted content
def clean_body_content(content):
    return ' '.join(content.split())

# Split DOM content into chunks
def split_dom_content(dom_content, chunk_size=3000):
    words = dom_content.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Parse content using SambaNova API
def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt.format(dom_content=chunk, parse_description=parse_description)}
            ]

            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",  # Or your desired SambaNova model
                messages=messages,
                temperature=0.1,
                top_p=0.1
            )

            result = response.choices[0].message.content
            print(f"Parsed batch: {i} of {len(dom_chunks)}")
            parsed_results.append(result)

        except Exception as e:
            print(f"Error parsing chunk {i}: {e}")
            time.sleep(2)  # Optional retry after short delay
            continue

    return "\n".join(parsed_results)

@app.route('/', methods=['GET', 'POST'])
def index():
    parsed_result = None
    dom_content = None

    if request.method == 'POST':
        # Handle Scraping
        if 'scrape' in request.form:
            url = request.form.get('url')
            if url:
                html_content = scrape_website(url)
                body_content = extract_body_content(html_content)
                cleaned_content = clean_body_content(body_content)

                session['dom_content'] = cleaned_content  # Store scraped content in session
                dom_content = cleaned_content
        
        # Handle Parsing
        elif 'parse' in request.form:
            parse_description = request.form.get('parse_description')
            if 'dom_content' in session and parse_description:
                dom_chunks = split_dom_content(session['dom_content'])
                parsed_result = parse_with_ollama(dom_chunks, parse_description)

    return render_template('index.html', dom_content=session.get('dom_content'), parsed_result=parsed_result)

if __name__ == '__main__':
    app.run(debug=True)
