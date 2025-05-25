import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import tempfile
import chromadb
from chromadb.config import Settings
from ollama_rag import query_ollama
from ingest_data import list_supported_files
from utils import (
    load_and_split_documents,
    add_documents_to_qdrant,
    get_rag_chain,
    get_vector_store,
    should_use_web_search,
    run_web_search,
    COLLECTION_NAME,
    QDRANT_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    check_ollama_availability,
    list_loaded_documents
)

# --- ChromaDB Setup ---
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chromadb_store')
COLLECTION_NAME = 'my_documents'

# Initialize ChromaDB client and collection
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection(COLLECTION_NAME)

# --- Helper Functions ---
def embed_chunks_with_ollama(chunks):
    # Use Ollama to embed each chunk using the real embedding model
    embedded = []
    for chunk in chunks:
        try:
            embedding = query_ollama(chunk['content'], model=EMBEDDING_MODEL)
            # Ensure embedding is a list of floats
            if not isinstance(embedding, list):
                raise ValueError('Ollama did not return a list for embedding')
        except Exception as e:
            print(f"Embedding failed: {e}")
            embedding = [0.0] * 384  # Fallback to dummy embedding
        chunk['embedding'] = embedding
        embedded.append(chunk)
    return embedded

def add_documents_to_chromadb(documents, source_name):
    # Each document should be a dict with 'content', 'embedding', and 'metadata' (if available)
    # For demo, we'll assume 'content' and 'embedding' keys
    try:
        ids = [f"{source_name}_{i}" for i in range(len(documents))]
        texts = [doc['content'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents]
        metadatas = [doc.get('metadata', {'source': source_name}) for doc in documents]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return True, len(documents)
    except Exception as e:
        print(f"Error adding to ChromaDB: {e}")
        return False, 0

def list_loaded_documents():
    # Return a list of unique source names from metadatas
    try:
        all_metas = collection.get(include=['metadatas'])['metadatas']
        sources = set()
        for meta in all_metas:
            if meta and 'source' in meta:
                sources.add(meta['source'])
        return list(sources)
    except Exception as e:
        print(f"Error listing loaded documents: {e}")
        return []

def get_vector_store():
    # Return the ChromaDB collection
    return collection

# --- Page Configuration ---
st.set_page_config(page_title="Doc RAG Chat", layout="wide")
st.title("📄 RAG Chat: Files, URLs, DBs")
st.markdown("Upload documents (PDF, TXT, CSV, XLSX, JSON, SQLite), enter URLs, or provide a DB connection string (PostgreSQL). The system can also search the web.")
st.markdown("**Note:** For databases, only the schema (table names, columns) is loaded, not the data.")
st.markdown("---")

# --- One-time Ollama System Check ---
if 'ollama_checked' not in st.session_state:
    st.session_state.ollama_checked = False
    st.session_state.ollama_ok = False

if not st.session_state.ollama_checked:
    with st.spinner("Checking Ollama server and models..."):
        overall_ok, server_running, models_status, error_msg = check_ollama_availability()
        st.session_state.ollama_checked = True
        st.session_state.ollama_ok = overall_ok

        if overall_ok:
            st.success(f"Ollama check successful. Server running, models '{EMBEDDING_MODEL}' and '{LLM_MODEL}' confirmed.")
        else:
            st.error(f"Ollama Check Failed: {error_msg}")
            if server_running:
                st.info("Individual model status check:")
                for model, status in models_status.items():
                    if status:
                        st.info(f" -> Model '{model}' found.")
                    else:
                        st.warning(f" -> Model '{model}' missing or unconfirmed. Run `ollama pull {model}`")
            st.stop()

# --- Helper Functions ---

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Upload content, enter a URL/DB URI, and ask me questions."}]
    
    # We'll preserve the loaded_sources list but refresh it from Qdrant
    st.session_state.loaded_sources_updated = False
    
    # Clear the cached vector store to force reinitialization
    if hasattr(get_vector_store, 'clear'):
        get_vector_store.clear()
        
    # Don't set documents_added to False if we have documents in Qdrant
    # This allows chat to continue working with existing documents
    sources = list_loaded_documents()
    if not sources:
        st.session_state.documents_added = False
        st.session_state.loaded_sources = []
    else:
        # Keep documents_added as True and update sources
        st.session_state.loaded_sources = sources
    
    st.info("Chat history cleared. Documents in Qdrant remain available.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    # Initialize messages only if they don't exist (don't call clear_chat_history here)
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Upload content, enter a URL/DB URI, and ask me questions."}]
if "documents_added" not in st.session_state:
    st.session_state.documents_added = False
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = []
if "loaded_sources_updated" not in st.session_state:
    st.session_state.loaded_sources_updated = False
# We no longer need rag_chain in session state, as we access it via the cached function.

# --- Sidebar for Uploading --- 
st.sidebar.header("Add Content Source")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more documents",
    type=["pdf", "txt", "csv", "xlsx", "xls", "json", "db", "sqlite", "sqlite3"],
    help="Supports PDF, TXT, CSV, Excel, JSON, and SQLite files (schema only for SQLite).",
    accept_multiple_files=True
)

# Debug: Confirm uploader is rendered
if uploaded_files is not None:
    st.sidebar.success(f"Uploader ready. {len(uploaded_files)} file(s) selected.")
else:
    st.sidebar.info("Uploader ready. No files selected yet.")

# URL Input
url_input = st.sidebar.text_input(
    "Or enter a Web URL",
    placeholder="https://example.com"
)

# Database Connection String Input
db_uri_input = st.sidebar.text_input(
    "Or enter PostgreSQL DB URI (schema only)",
    placeholder="postgresql://user:pass@host:port/dbname"
)

# Process Button - Handles one source at a time
if st.sidebar.button("Process Content Source"):
    temp_dirs = []
    content_sources = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
        temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            content_sources.append({"type": "file", "path": file_path, "name": uploaded_file.name})
        st.sidebar.info(f"Processing file: {uploaded_file.name}")
    if url_input:
        content_sources.append({"type": "url", "url": url_input})
        st.sidebar.info(f"Processing URL: {url_input}")
    if db_uri_input:
        content_sources.append({"type": "db", "uri": db_uri_input})
        st.sidebar.info(f"Processing DB URI: {db_uri_input.split('@')[0]}...")
    if not content_sources:
        st.sidebar.warning("Please provide at least one source: upload files, enter a URL, or enter a DB URI.")
    for content_source_info in content_sources:
        source_type = content_source_info["type"]
        source_identifier = content_source_info.get("name") or content_source_info.get("url") or content_source_info.get("uri")
        with st.spinner(f"Loading and splitting content from {source_type}: {source_identifier}... This might take a moment."):
            documents, source_name = load_and_split_documents(
                file_path=content_source_info.get("path"),
                url=content_source_info.get("url"),
                db_uri=content_source_info.get("uri")
            )
        if documents:
            embedded_docs = embed_chunks_with_ollama(documents)
            display_name = source_name if source_name != "Unknown" else source_identifier
            with st.spinner(f"Adding {len(embedded_docs)} chunk(s) from {display_name} to vector store..."):
                success, count = add_documents_to_chromadb(embedded_docs, display_name)
                if success:
                    st.sidebar.success(f"Successfully added {count} chunk(s) from {display_name} to ChromaDB!")
                    st.session_state.documents_added = True
                    
                    # Add source to loaded sources list if not already there
                    if display_name not in st.session_state.loaded_sources:
                        st.session_state.loaded_sources.append(display_name)
                else:
                    st.sidebar.error(f"Failed to add content from {display_name}. Check console.")
        else:
            st.sidebar.error(f"Could not load or split content from {source_identifier}. Check console.")
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

# --- Batch Ingest Button ---
if st.sidebar.button("Batch Ingest from 'ingest' Folder"):
    ingest_folder = os.path.join(os.path.dirname(__file__))
    supported_files = list_supported_files(ingest_folder, ['.py', '.csv', '.txt', '.pdf', '.json', '.xlsx', '.xls'])
    for filename in supported_files:
        file_path = os.path.join(ingest_folder, filename)
        with st.spinner(f"Batch loading and splitting: {filename}"):
            documents, source_name = load_and_split_documents(file_path=file_path)
        if documents:
            embedded_docs = embed_chunks_with_ollama(documents)
            with st.spinner(f"Adding {len(embedded_docs)} chunk(s) from {filename} to vector store..."):
                success, count = add_documents_to_chromadb(embedded_docs, filename)
                if success:
                    st.sidebar.success(f"Batch: Added {count} chunk(s) from {filename} to ChromaDB!")
                    st.session_state.documents_added = True
                    if filename not in st.session_state.loaded_sources:
                        st.session_state.loaded_sources.append(filename)
                else:
                    st.sidebar.error(f"Batch: Failed to add content from {filename}. Check console.")
        else:
            st.sidebar.error(f"Batch: Could not load or split content from {filename}. Check console.")

# Display note about vector store persistence
st.sidebar.markdown("--- ")

# Fetch all existing sources from ChromaDB at startup and update session state
if 'loaded_sources_updated' not in st.session_state or not st.session_state.loaded_sources_updated:
    st.sidebar.text("Loading document sources...")
    docs_from_chromadb = list_loaded_documents()
    if docs_from_chromadb:
        # Update loaded sources in session state
        for source in docs_from_chromadb:
            if source not in st.session_state.loaded_sources:
                st.session_state.loaded_sources.append(source)
        
        # Set documents_added flag to true if we found documents
        if docs_from_chromadb and not st.session_state.documents_added:
            st.session_state.documents_added = True
    st.session_state.loaded_sources_updated = True

# Display loaded documents section
st.sidebar.subheader("Loaded Documents")
if st.session_state.loaded_sources:
    for i, source in enumerate(st.session_state.loaded_sources):
        st.sidebar.text(f"{i+1}. {source}")
else:
    st.sidebar.text("No documents loaded yet")
    # Add helpful message for new users
    st.sidebar.info("👆 Add content by uploading a file, entering a URL, or providing a database connection string.")

st.sidebar.markdown("--- ")
st.sidebar.caption(f"Vector data is managed by ChromaDB")

# --- Chat Interface --- 
st.subheader("Chat with your Content")

# Attempt to initialize vector store early if documents were added,
# so the RAG chain can be built when needed.
# This also provides earlier feedback if vector store init fails.
if st.session_state.documents_added:
    try:
        vector_store = get_vector_store() # Trigger initialization/retrieval
        if vector_store is None:
            st.warning("Vector store could not be initialized. Cannot proceed with chat.")
    except Exception as e:
         st.error(f"Error initializing vector store for chat: {e}")

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask a question..."):
    # Check if documents have been added (as a proxy for chain readiness)
    if not st.session_state.documents_added:
        st.warning("Please add a content source first using the sidebar.")
        st.stop()

    # Get the RAG chain (will be cached after first successful call)
    try:
        # RAG chain is no longer cached, built dynamically using cached vector store
        rag_chain = get_rag_chain()
        if rag_chain is None: # Check if the function failed internally
             # Error messages are now handled inside get_rag_chain or get_vector_store
             st.warning("RAG chain is not available. Check logs.")
             st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred getting RAG chain: {e}")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get response from RAG chain
    try:
        with st.spinner("Thinking based on documents..."):
            rag_response = rag_chain.invoke(prompt)
            
            # Check if the response indicates no relevant documents were found
            if "No relevant documents found" in rag_response:
                st.chat_message("assistant").warning("I couldn't find relevant information in the documents. Try adding more content sources or rephrasing your question.")
                st.session_state.messages.append({"role": "assistant", "content": "I couldn't find relevant information in the documents. Try adding more content sources or rephrasing your question."})
            else:
                # Display initial RAG response
                st.chat_message("assistant").write(f"**From Documents:**\n{rag_response}")
                st.session_state.messages.append({"role": "assistant", "content": f"**From Documents:**\n{rag_response}"})
    except Exception as e:
        error_msg = f"Error during document search: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.stop()

    # Decide whether to perform a web search
    if should_use_web_search(prompt, rag_response):
        try:
            with st.spinner("Performing web search..."):
                web_search_results = run_web_search(prompt)

            # Display web search results
            st.chat_message("assistant").info(f"**Web Search Results:**\n{web_search_results}")
            st.session_state.messages.append({"role": "assistant", "content": f"**Web Search Results:**\n{web_search_results}"})
        except Exception as e:
            st.error(f"Error during web search: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"I attempted to search the web but encountered an error: {str(e)}"})

# Button to clear chat history
st.sidebar.markdown("--- ")
st.sidebar.button("Clear Chat History", on_click=clear_chat_history) 