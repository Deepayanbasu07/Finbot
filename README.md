# FinQuery Chat 

FinQuery Chat is an intelligent financial document analysis tool designed to help users extract insights from financial reports and conference call transcripts. It features a conversational interface for querying PDF documents and a dedicated analyzer for sentiment analysis of transcripts.

## ✨ Features

* 📄 **Document Upload:** Easily upload multiple financial documents (PDFs) for analysis.
* 💬 **Conversational Q&A (RAG):** Chat with an AI to get insights and answers directly from the content of your uploaded documents.
* 🔍 **Smart Search:** Search across documents by company, quarter, or keyword. The system provides relevant excerpts and page references from the source documents.
* 📊 **Conference Call Sentiment Analysis:** The 'FinDoc Analyzer' tool processes uploaded conference call transcripts (PDFs) to provide overall sentiment scores using Vader and AFINN libraries.
* 🧠 **LLM Powered:** Leverages Ollama with the Llama3 model for response generation and Nomic Embed Text for creating document embeddings.
* 🚀 **Efficient Retrieval:** Employs FAISS (Facebook AI Similarity Search) for fast and relevant information retrieval from documents.
* 🎨 **Modern UI:** Clean and intuitive user interface built with HTML, TailwindCSS, and JavaScript.

## 🛠️ Tech Stack

* **Backend:** Python, Flask, Langchain
* **LLM & Embeddings:** Ollama (Llama3, Nomic Embed Text)
* **Vector Store:** FAISS
* **PDF Processing:** PyMuPDF (Fitz)
* **NLP & Sentiment Analysis:** NLTK, VaderSentiment, AFINN
* **Frontend:** HTML, TailwindCSS, JavaScript

## 📸 Screenshots

Here's a glimpse of FinQuery Chat in action:

**Main Chat Interface (FinQueryChat):**

1.  **Landing Page:** The main interface for uploading documents and interacting with the chatbot.
   ![Image](https://github.com/user-attachments/assets/f542d12b-b5a1-4756-82eb-f9ce7fc28de1)

2.  **Uploading & Searching:** Users can upload multiple PDF files and see them listed. The search bar initiates queries about the documents.
    ![Image](https://github.com/user-attachments/assets/434eae3a-55b2-465f-a587-01e17be554b6)
    ![Image](https://github.com/user-attachments/assets/1528441e-125a-43c4-b89b-283ea8900cfc)
    ![Image](https://github.com/user-attachments/assets/d7ac2343-07d2-4987-a36b-e875446f6566)

4.  **Search Suggestions:** As you type in the search bar, relevant suggestions with excerpts from documents appear.
    ![Image](https://github.com/user-attachments/assets/5a5c7aa8-7cb1-45b5-a253-feb1a7690533)

5.  **Chatbot Responses:** The AI provides answers based on the document content, citing the source document and page.
    ![Image](https://github.com/user-attachments/assets/d4582824-4568-4830-b690-9fdd2582bcd0)


**Document Analyzer (FinDoc Analyzer):**

5.  **Analyzer Interface:** The 'FinDoc Analyzer' page for sentiment analysis of conference call transcripts.
    ![Image](https://github.com/user-attachments/assets/64ad9b8e-8196-46ae-be9d-d7317e1b339b)

6.  **Sentiment Analysis Results:** Displays Vader sentiment scores (Positive, Neutral, Negative, Compound) and an AFINN adjusted score.
    ![Image](https://github.com/user-attachments/assets/00250d1e-bd93-4d09-8aaa-e271d0f7a9cd)

## ⚙️ Setup and Running

1.  **Prerequisites:**
    * Python 3.8+
    * Ollama installed and running.
    * Pull the required Ollama models:
        ```bash
        ollama pull llama3
        ollama pull nomic-embed-text
        ```

2.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd FinQueryChat # Or your repository name
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    Create a `requirements.txt` file with the following content (or add any other specific versions you used):
    ```txt
    Flask
    Flask-CORS
    Werkzeug
    PyMuPDF
    nltk
    vaderSentiment
    afinn
    faiss-cpu # or faiss-gpu if you have a compatible GPU and setup
    langchain
    langchain-ollama
    langchain-community
    langchain-text-splitters
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    You might need to download NLTK data for the first time:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```
    (You can run this in a Python interpreter or add it to your script's initialization).

5.  **Configure Upload Folder:**
    The application uses an `uploads` folder. Ensure it's created or the script has permissions to create it.

6.  **Run the Flask application:**
    ```bash
    flask run
    ```
    The application should be accessible at `http://localhost:5000`.
    The main chat interface is at `/` or `/app.html` (or `/chatbot.html` as per your Flask routes).
    The Document Analyzer is at `/findocanalyser.html`.

## 🚀 Usage

**FinQuery Chat (Main Interface - `app.html` or `chatbot.html`):**

1.  Navigate to the main page (`http://localhost:5000`).
2.  Click on "Choose Files" to select one or more financial PDF documents.
3.  The selected files will be listed below the "Choose Files" button after a brief moment.
4.  Click "Upload Documents". A confirmation message will appear in the chat window.
5.  Use the search bar on the left sidebar (placeholder: "Search by company, quarter, or keyword...") to find specific information. Click on suggestions to populate the chat or trigger a search.
6.  Alternatively, type your questions directly into the chat input field at the bottom of the main panel and press Enter or click the send icon.
7.  The chatbot will process your query and respond with answers based on the content of the uploaded documents, including references to the document name and page number.

**FinDoc Analyzer (`findocanalyser.html`):**

1.  Navigate to the "Doc Analyser" page using the link in the top navigation bar (or go to `http://localhost:5000/findocanalyser.html`).
2.  Click "Choose Files" to upload conference call transcripts (PDFs are supported for text extraction).
3.  Enter the "Start Date" and "End Date" relevant to the transcripts.
4.  Click the "Analyze Sentiment" button.
5.  After processing, the "Analysis Results" section will display:
    * Vader Sentiment Scores (Positive, Negative, Neutral, Compound)
    * AFINN Adjusted Score
    * Word Count

## 🧑‍💻 Author

Built with ❤️ by **Deepayan Basu**
* BTech @ IIT Jodhpur
* **Email:** [deepayanbasu5@gmail.com](mailto:deepayanbasu5@gmail.com)
* **LinkedIn:** [linkedin.com/in/deepayan-basu-06a5b123b](https://www.linkedin.com/in/deepayan-basu-06a5b123b/)
* **GitHub:** [github.com/Deepayanbasu07](https://github.com/Deepayanbasu07)

---

Generated with assistance from an AI.
