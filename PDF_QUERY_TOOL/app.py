import streamlit as st
import os
import tempfile
from typing import List
import pdfplumber
import numpy as np
import ollama
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pytesseract

st.set_page_config(page_title="PDF Information Extractor", layout="wide")
st.title("PDF Information Extractor (Ollama + Phi)")

# Constants
CHUNK_SIZE = 1000  # characters
MAX_CHUNKS = 100  # Limit number of chunks per session
EMBED_BATCH_SIZE = 10  # Number of chunks per embedding batch
EMBED_MODEL = "nomic-embed-text"
PHI_MODEL = "phi4-mini-reasoning"
CHROMA_DIR = os.path.join(os.getcwd(), "autonomus", "chromadb_store")

# Session state for uploaded files, embeddings, and model resources
if 'pdf_files' not in st.session_state:
    st.session_state['pdf_files'] = []
if 'pdf_chunks' not in st.session_state:
    st.session_state['pdf_chunks'] = []
if 'chunk_sources' not in st.session_state:
    st.session_state['chunk_sources'] = []
if 'chunk_metadata' not in st.session_state:
    st.session_state['chunk_metadata'] = []
if 'chroma_collection' not in st.session_state:
    st.session_state['chroma_collection'] = None
if 'ocr_used' not in st.session_state:
    st.session_state['ocr_used'] = False

st.sidebar.header("Upload PDF(s)")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)

def table_to_markdown(table):
    # Format a table (list of lists) as markdown
    if not table or not table[0]:
        return ""
    header = '| ' + ' | '.join([str(cell) if cell is not None else '' for cell in table[0]]) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(table[0])) + ' |'
    rows = ['| ' + ' | '.join([str(cell) if cell is not None else '' for cell in row]) + ' |' for row in table[1:]]
    return '\n'.join([header, sep] + rows)

def extract_text_chunks(pdf_file):
    # Use pdfplumber for robust extraction, with OCR fallback and markdown table formatting
    with pdfplumber.open(pdf_file) as pdf:
        chunks = []
        chunk_sources = []
        chunk_metadata = []
        ocr_used = False
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # OCR fallback if no text found
            if not text.strip():
                ocr_text = page.to_image(resolution=300).ocr(layout=True, lang='eng')
                if ocr_text:
                    text = ocr_text
                    ocr_used = True
            # Extract tables as markdown
            tables = page.extract_tables()
            for table in tables:
                table_md = table_to_markdown(table)
                if table_md.strip():
                    text += f"\n[Table on page {page_num}]:\n{table_md}"
            # Chunk the text for this page
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i:i+CHUNK_SIZE]
                if chunk.strip():
                    chunk_with_meta = f"[Page {page_num}] {chunk.strip()}"
                    chunks.append(chunk_with_meta)
                    chunk_sources.append(pdf_file.name)
                    chunk_metadata.append({"page": page_num, "source": pdf_file.name})
                if len(chunks) >= MAX_CHUNKS:
                    break
            if len(chunks) >= MAX_CHUNKS:
                break
    return chunks, chunk_sources, chunk_metadata, ocr_used

# Helper: Get embeddings from nomic-embed-text via Ollama, in batches
def get_embeddings_batched(chunks, batch_size=EMBED_BATCH_SIZE):
    all_embs = []
    for chunk in chunks:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        all_embs.append(response['embedding'])
    return np.array(all_embs)

# Helper: Query Phi model via Ollama
def query_phi(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model=PHI_MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Initialize ChromaDB client and collection
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection("pdf_chunks")
    return collection

if uploaded_files:
    st.session_state['pdf_files'] = uploaded_files
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

if st.session_state['pdf_files']:
    st.write("### Uploaded PDFs:")
    for f in st.session_state['pdf_files']:
        st.write(f"- {f.name}")
    if st.button("Process PDFs and Create Embeddings"):
        st.info("Extracting text and creating embeddings (with OCR, table formatting, batching, and ChromaDB)...")
        all_chunks = []
        chunk_sources = []
        chunk_metadata = []
        ocr_used = False
        for f in st.session_state['pdf_files']:
            chunks, sources, metadata, ocr_flag = extract_text_chunks(f)
            all_chunks.extend(chunks)
            chunk_sources.extend(sources)
            chunk_metadata.extend(metadata)
            ocr_used = ocr_used or ocr_flag
        st.session_state['pdf_chunks'] = all_chunks
        st.session_state['chunk_sources'] = chunk_sources
        st.session_state['chunk_metadata'] = chunk_metadata
        st.session_state['ocr_used'] = ocr_used
        # Create or get ChromaDB collection
        collection = get_chroma_collection()
        all_ids = collection.get()['ids']
        if all_ids:
            collection.delete(ids=all_ids)
        for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch_chunks = all_chunks[i:i+EMBED_BATCH_SIZE]
            batch_embs = get_embeddings_batched(batch_chunks, batch_size=EMBED_BATCH_SIZE)
            ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
            metadatas = [chunk_metadata[i+j] for j in range(len(batch_chunks))]
            collection.add(
                embeddings=batch_embs.tolist(),
                documents=batch_chunks,
                ids=ids,
                metadatas=metadatas
            )
        st.session_state['chroma_collection'] = collection
        st.success(f"Processed {len(all_chunks)} chunks and stored in ChromaDB.")
        if ocr_used:
            st.warning("OCR was used for some pages. Extraction may be less accurate for scanned or image-based PDFs.")

if st.session_state.get('chroma_collection') is not None:
    st.write("### Query the PDF(s)")
    user_query = st.text_input("Ask a question about the uploaded PDF(s):")
    if user_query:
        st.info(f"Embedding query and retrieving relevant context from ChromaDB...")
        query_emb = get_embeddings_batched([user_query], batch_size=1)[0]
        results = st.session_state['chroma_collection'].query(
            query_embeddings=[query_emb.tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        context = "\n---\n".join(results['documents'][0])
        st.info("Querying Phi model with retrieved context...")
        answer = query_phi(context, user_query)
        st.write("#### Answer:")
        st.write(answer)
        with st.expander("Think", expanded=False):
            st.markdown(f"<think>{answer}</think>", unsafe_allow_html=True)
        st.write("\n**Top relevant chunks:**")
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            st.write(f"*From {meta['source']} (page {meta['page']})* (distance: {dist:.4f}):")
            st.write(doc[:500])
            # st.code(doc[:500])
else:
    st.info("Upload and process PDFs to enable querying.")

st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit, Ollama, ChromaDB, pdfplumber, pytesseract, and Microsoft Phi.") 