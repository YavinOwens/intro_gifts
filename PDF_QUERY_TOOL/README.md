# gifts

## AI-Powered PDF Financial Report Analyzer

This project provides a modern Streamlit web app for extracting, embedding, and querying financial PDF reports using advanced AI and database tools. It is designed for robust extraction of tables and text (including OCR for scanned PDFs), persistent vector storage, and natural language querying with state-of-the-art models.

---

## Features
- **Upload one or more PDF files** via the web UI
- **Extracts tables and text** from each page using `pdfplumber` (with OCR fallback via `pytesseract`)
- **Saves tables as markdown in context** for better chunking and traceability
- **Embeds all extracted text and tables** using `nomic-embed-text` via the Ollama SDK
- **Stores embeddings in ChromaDB** for persistent, efficient semantic search
- **Natural language querying** using Microsoft Phi (via Ollama) with context retrieval from ChromaDB
- **Session state** for fast, interactive exploration
- **Expander UI** for model "thoughts" (shows the full answer in a <think>...</think> block)
- **Warnings for OCR usage** to alert users to possible extraction inaccuracies

---

## Setup Instructions

1. **Clone the repository and enter the project directory**

2. **Create and activate a Python virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR (required for scanned/image-based PDFs)**
- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install tesseract-ocr
  ```

5. **Install Ollama and ensure the `nomic-embed-text` and `phi4-mini-reasoning` models are available**
- See: https://ollama.com/

6. **(Optional) Install ChromaDB server for persistent storage**
- The app uses a local persistent directory by default (`autonomus/chromadb_store`).

---

## Running the App

From the project root (with venv activated):
```bash
python -m streamlit run autonomus/app.py
```

- The app will be available at [http://localhost:8501](http://localhost:8501) (or another port if in use).

---

## Usage
1. **Upload one or more PDF files** using the sidebar.
2. **Click "Process PDFs and Create Embeddings"** to extract, chunk, and embed the content.
3. **Query the PDFs** using natural language in the main input box.
4. **Review the answer and top relevant chunks.**
5. **Expand the "Think" section** to see the model's full reasoning in a <think>...</think> block.

---

## Tech Stack
- **Streamlit** (UI)
- **pdfplumber** (PDF extraction)
- **pytesseract** (OCR)
- **Ollama SDK** (embeddings + chat)
- **nomic-embed-text** (embeddings)
- **Microsoft Phi** (reasoning/QA)
- **ChromaDB** (vector database)
- **NumPy, etc.**

---

## Troubleshooting
- **pytesseract not found:**
  - Ensure you are running Streamlit from the activated venv.
  - Ensure Tesseract binary is installed and in your PATH.
- **Ollama or model errors:**
  - Make sure Ollama is running and the required models are downloaded.
- **ChromaDB errors:**
  - The app will create a local persistent store by default. Check permissions if you see file errors.
- **General:**
  - If you see `ModuleNotFoundError`, check your venv and installed packages.

---

## License
See [LICENSE](LICENSE).