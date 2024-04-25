from flask import Flask, request, render_template, redirect, url_for
import PyPDF2
import re
import json

app = Flask(__name__)

# Helper function to process text
def process_text(text, file_type):
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into chunks by sentences, respecting a maximum chunk size
    sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    
    with open("vault.txt", "a", encoding="utf-8") as vault_file:
        for chunk in chunks:
            vault_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks

    return f"{file_type} file content appended to vault.txt with each chunk on a separate line."

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file_type = request.form['file_type']
        file = request.files['file']
        if file:
            if file_type == 'pdf':
                reader = PyPDF2.PdfReader(file.stream)
                text = ''.join(page.extract_text() + " " for page in reader.pages if page.extract_text())
            elif file_type == 'text':
                text = file.stream.read().decode('utf-8')
            elif file_type == 'json':
                data = json.load(file.stream)
                text = json.dumps(data, ensure_ascii=False)
            
            message = process_text(text, file_type)
            return message
        return 'No file uploaded'

    return '''
    <!doctype html>
    <title>Upload File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=hidden name=file_type value="pdf">
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
