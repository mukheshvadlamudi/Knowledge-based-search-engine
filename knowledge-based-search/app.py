from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the embedding model (lightweight and fast)
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCoRt79lLRyQ7D5PgTuY6gZ_-D0EQ_qmNs"  # Replace with your API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Global storage for documents
document_chunks = []
chunk_embeddings = None

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_txt(txt_path):
    """Extract text from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

def split_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings(chunks):
    """Create embeddings for all chunks"""
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

def find_relevant_chunks(query, top_k=3):
    """Find most relevant chunks for a query"""
    global chunk_embeddings, document_chunks
    
    if chunk_embeddings is None or len(document_chunks) == 0:
        return []
    
    query_embedding = embedder.encode([query])[0]
    
    # Calculate cosine similarity
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = [document_chunks[i] for i in top_indices]
    return relevant_chunks

def generate_answer(query, context_chunks):
    """Generate answer using Gemini"""
    context = "\n\n".join(context_chunks)
    
    prompt = f"""Using the following documents, answer the user's question succinctly and accurately.

Documents:
{context}

Question: {query}

Answer (be concise and directly address the question):"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_documents():
    """Handle document uploads"""
    global document_chunks, chunk_embeddings
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    all_text = ""
    processed_files = []
    
    for file in files:
        if file.filename == '':
            continue
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(filepath)
        else:
            continue
        
        all_text += text + "\n\n"
        processed_files.append(filename)
    
    if not all_text.strip():
        return jsonify({'error': 'No text could be extracted from files'}), 400
    
    # Process documents
    print("Splitting text into chunks...")
    document_chunks = split_into_chunks(all_text)
    
    print(f"Creating embeddings for {len(document_chunks)} chunks...")
    chunk_embeddings = create_embeddings(document_chunks)
    
    return jsonify({
        'message': 'Documents processed successfully',
        'files': processed_files,
        'chunks': len(document_chunks)
    })

@app.route('/query', methods=['POST'])
def query_documents():
    """Handle user queries"""
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    if len(document_chunks) == 0:
        return jsonify({'error': 'No documents uploaded yet'}), 400
    
    # Find relevant chunks
    print(f"Finding relevant chunks for: {query}")
    relevant_chunks = find_relevant_chunks(query, top_k=3)
    
    # Generate answer
    print("Generating answer...")
    answer = generate_answer(query, relevant_chunks)
    
    return jsonify({
        'query': query,
        'answer': answer,
        'sources': relevant_chunks[:2]  # Return top 2 source chunks
    })

@app.route('/reset', methods=['POST'])
def reset_system():
    """Reset the system (clear all documents)"""
    global document_chunks, chunk_embeddings
    
    document_chunks = []
    chunk_embeddings = None
    
    # Clear uploads folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing file: {e}")
    
    return jsonify({'message': 'System reset successfully'})

if __name__ == '__main__':
    print("Starting RAG Knowledge Base Search Engine...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)