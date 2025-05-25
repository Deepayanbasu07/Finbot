from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import fitz
import traceback
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
import datetime
import faiss
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

# Setup and init
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nltk.download('punkt')
nltk.download('stopwords')

# Initialize models
llm = None
embedding_model = None
vector_store = None

import re
from datetime import datetime

def get_quarter_from_text(text):
    # Try to find quarter info like "Q1 2024"
    match = re.search(r'Q([1-4])\s*20(\d{2})', text)
    if match:
        q = match.group(1)
        year = '20' + match.group(2)
        return f"{year}-Q{q}"
    return None

def date_to_quarter(date_obj):
    # Converts a datetime.date to string quarter e.g. '2024-Q1'
    quarter = (date_obj.month - 1) // 3 + 1
    return f"{date_obj.year}-Q{quarter}"

from collections import defaultdict

@app.route('/upload-and-analyze-quarterly', methods=['POST'])
def upload_and_analyze_quarterly():
    if 'files' not in request.files:
        return jsonify({'message': 'No files part in request'}), 400

    start_date_str = request.form.get('startDate')
    end_date_str = request.form.get('endDate')
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None
    except Exception:
        return jsonify({'message': 'Invalid date format'}), 400

    files = request.files.getlist('files')
    quarterly_texts = defaultdict(str)

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            text = ""

        quarter = get_quarter_from_text(text)
        if not quarter and start_date and end_date:
            # fallback: assign file to all quarters between start and end
            # or approximate by assigning entire text to all quarters (simple approach)
            # Here, we assign all to a single quarter derived from start_date for simplicity:
            quarter = date_to_quarter(start_date)

        if not quarter:
            quarter = "Unknown"

        quarterly_texts[quarter] += " " + text

    quarterly_scores = {}
    for quarter, text in quarterly_texts.items():
        sentiment_summary = run_sentiment_analysis_from_text(text, start_date_str, end_date_str)
        quarterly_scores[quarter] = round(sentiment_summary['afinn_adjusted'], 2)

    return jsonify({"quarterly_scores": quarterly_scores})


def initialize_models():
    global llm, embedding_model
    try:
        llm = ChatOllama(model="llama3", temperature=0)
        embedding_model = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
        print("✅ LLM and Embeddings initialized")
    except Exception as e:
        print("❌ Model init failed:", e)
        llm = None
        embedding_model = None

initialize_models()

def extract_text_from_files(files):
    all_text = ""
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            doc = fitz.open(filepath)
            for page in doc:
                all_text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Failed extracting text from {filename}: {e}")
    return all_text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

def run_sentiment_analysis_from_text(raw_text, start_date, end_date):
    clean_text = preprocess_text(raw_text)
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(clean_text)
    afinn = Afinn(language='en')
    afinn_score = afinn.score(clean_text)
    word_count = len(clean_text.split())
    afinn_adjusted = (afinn_score / word_count) * 100 if word_count else 0

    revenue_data = {}
    for line in raw_text.splitlines():
        try:
            if "revenue" in line.lower():
                parts = line.lower().split()
                year = None
                amount = None
                for p in parts:
                    if p.isdigit() and len(p) == 4:
                        year = int(p)
                    if p.replace(',', '').isdigit():
                        amount = float(p.replace(',', ''))
                if year and amount:
                    revenue_data[year] = amount
        except:
            continue

    revenue_growth = None
    years_sorted = sorted(revenue_data.keys())
    if len(years_sorted) >= 2:
        first_year = years_sorted[0]
        last_year = years_sorted[-1]
        first_rev = revenue_data[first_year]
        last_rev = revenue_data[last_year]
        if first_rev > 0:
            revenue_growth = round(((last_rev - first_rev) / first_rev) * 100, 2)

    summary = {
        'sentiment': sentiment_scores,
        'afinn_adjusted': afinn_adjusted,
        'word_count': word_count,
        'revenue_data': revenue_data,
        'revenue_growth_percent': revenue_growth,
        'period_start': start_date,
        'period_end': end_date
    }
    return summary

# Routes: chatbot and upload/analyze

@app.route('/')
def index():
    return send_from_directory('frontend', 'chatbot.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('frontend', path)

@app.route('/upload', methods=['POST'])
def upload():
    global vector_store, embedding_model
    if embedding_model is None:
        return jsonify({'response': 'Embedding model not initialized'}), 500
    if 'documents' not in request.files:
        return jsonify({'response': 'No files provided.'}), 400
    files = request.files.getlist('documents')
    saved_files = []
    all_docs = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        saved_files.append(filename)
        try:
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                full_text += f"=== PAGE {page.number + 1} ===\n{page.get_text()}\n"
            doc.close()
        except Exception as e:
            print(f"Failed to extract text from {filename}: {e}")
            full_text = ""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            page_num = 1
            if "=== PAGE" in chunk:
                try:
                    page_num = int(chunk.split("=== PAGE ")[1].split(" ===")[0])
                except:
                    pass
            doc_obj = Document(page_content=chunk, metadata={'filename': filename, 'chunk': i, 'page': page_num})
            all_docs.append(doc_obj)
    if vector_store is None:
        dummy_embedding = embedding_model.embed_query("test")
        dim = len(dummy_embedding)
        index = faiss.IndexFlatL2(dim)
        vector_store = FAISS(embedding_function=embedding_model, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    try:
        vector_store.add_documents(all_docs)
        return jsonify({'response': f'Uploaded and processed {len(saved_files)} files.', 'files': saved_files})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f'Error adding documents: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search_query():
    global vector_store, llm
    if llm is None:
        return jsonify({'response': 'LLM not initialized'}), 500
    if vector_store is None:
        return jsonify({'response': 'No documents uploaded yet'}), 400
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'response': "Please enter a valid query."}), 400
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return jsonify({'response': f"No results found for: '{query}'", 'suggestions': []})
    suggestions = []
    seen_docs = set()
    for doc in docs:
        filename = doc.metadata.get('filename', 'Unknown Document')
        page_num = doc.metadata.get('page', doc.metadata.get('chunk', 0) + 1)
        doc_key = f"{filename}_{page_num}"
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            excerpt = doc.page_content.replace(f"=== PAGE {page_num} ===", "").strip()
            excerpt = excerpt.split("\n")[0][:120] + "..."
            suggestions.append({'document': filename, 'page': page_num, 'excerpt': excerpt})
    context_parts = []
    for doc in docs:
        filename = doc.metadata.get('filename', 'Unknown')
        page_num = doc.metadata.get('page', doc.metadata.get('chunk', 0) + 1)
        content = doc.page_content.replace(f"=== PAGE {page_num} ===", "").strip()
        context_parts.append(f"📄 {filename} (Page {page_num}):\n{content}")
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""
You are a financial document analyst. Please answer the question using ONLY the provided context.
If the answer isn't in the context, say "I couldn't find this information in the documents."

CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

ANSWER (be concise and mention document/page references when possible):
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f'LLM Error: {str(e)}', 'suggestions': suggestions}), 500
    return jsonify({'response': answer, 'suggestions': suggestions})

@app.route('/ask', methods=['POST'])
def ask():
    global llm
    if llm is None:
        return jsonify({'response': 'LLM not initialized'}), 500
    try:
        data = request.get_json(force=True)
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'response': 'No question provided'}), 400
        response = llm.invoke([HumanMessage(content=question)])
        return jsonify({'response': response.content})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/upload-and-analyze', methods=['POST'])
def upload_and_analyze():
    if 'files' not in request.files:
        return jsonify({'message': 'No files part in request'}), 400
    files = request.files.getlist('files')
    start_date = request.form.get('startDate')
    end_date = request.form.get('endDate')
    if not start_date or not end_date:
        return jsonify({'message': 'Missing startDate or endDate'}), 400
    raw_text = extract_text_from_files(files)
    if not raw_text:
        return jsonify({'message': 'Failed to extract any text from uploaded files'}), 500
    try:
        sentiment_summary = run_sentiment_analysis_from_text(raw_text, start_date, end_date)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Error analyzing sentiment: {str(e)}'}), 500
    return jsonify({'sentiment_summary': sentiment_summary})

if __name__ == '__main__':
    print("🚀 Running FinQuery backend on http://localhost:5000")
    app.run(debug=True)
