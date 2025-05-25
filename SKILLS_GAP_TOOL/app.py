import streamlit as st
import os
import json
from collections import Counter
import pandas as pd
from ollama_rag import query_ollama
import subprocess
from tools.ML.llama_skill_identifier import query_codellama_ollama, extract_skills_from_llama_response
from tools.web_search import search_duckduckgo
import time
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import chromadb
from chromadb.config import Settings
import sys
import tempfile
import importlib.util
import spacy
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ingest'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ingest_data import list_supported_files
from utils import load_and_split_documents, check_ollama_availability, EMBEDDING_MODEL, LLM_MODEL, parallel_embed_chunks

# Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Skills Gap Data Explorer", layout="wide")

# --- Ensure skillset_cache is always initialized ---
if "skillset_cache" not in st.session_state:
    st.session_state["skillset_cache"] = {}
skillset_cache = st.session_state["skillset_cache"]

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INGESTED_PATH = os.path.join(BASE_DIR, 'downstream', 'ingested_files.json')
CATEGORIZED_PATH = os.path.join(BASE_DIR, 'downstream', 'skills_categorized.json')
WEBREF_PATH = os.path.join(BASE_DIR, 'downstream', 'web_references.json')
WEBREF_FILTERED_PATH = os.path.join(BASE_DIR, 'downstream', 'web_references_filtered.json')
DOC_DIR = os.path.join(BASE_DIR, 'downstream', 'doc')
# Ensure DOC_DIR exists
os.makedirs(DOC_DIR, exist_ok=True)
MD_REPORT = os.path.join(DOC_DIR, 'skills_gap_report.md')
QMD_REPORT = os.path.join(DOC_DIR, 'skills_gap_report.qmd')
PDF_REPORT = os.path.join(DOC_DIR, 'skills_gap_report.pdf')
HTML_REPORT = os.path.join(DOC_DIR, 'skills_gap_report.html')
TOOLS_DIR = os.path.join(BASE_DIR, 'tools')

# --- ChromaDB and Embedding Setup ---
CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chromadb_store')
COLLECTION_NAME = 'my_documents'
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection(COLLECTION_NAME)

# Add chunk size option to sidebar
chunk_size = st.sidebar.number_input("Chunk size (characters)", min_value=200, max_value=5000, value=1000, step=100, help="Set the chunk size for splitting documents.")

# Check if spaCy en_core_web_sm is installed, if not, download it with a spinner
with st.spinner("Checking spaCy model (en_core_web_sm)..."):
    try:
        spacy.load("en_core_web_sm")
        spacy_model_ready = True
    except OSError:
        st.info("Downloading spaCy model: en_core_web_sm (first time only)...")
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("spaCy model en_core_web_sm downloaded!")
            spacy_model_ready = True
        else:
            st.error("Failed to download spaCy model en_core_web_sm. Some features may not work.")
            spacy_model_ready = False

# Replace embed_chunks_with_ollama to use fallback if Ollama is not running
def is_ollama_running():
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

ollama_available = is_ollama_running()

def embed_chunks_with_ollama(chunks):
    progress = st.progress(0, text="Embedding chunks...")
    fallback_used = False
    def progress_callback(done, total):
        progress.progress(done/total, text=f"Embedding chunk {done}/{total}")
    def embed_func(chunk):
        if ollama_available:
            try:
                embedding = query_ollama(chunk['content'], model=EMBEDDING_MODEL)
                if not isinstance(embedding, list):
                    print(f"[DEBUG] Ollama returned non-list: {embedding} (type: {type(embedding)})")
                    fallback_used = True
                    embedding = [random.uniform(-1, 1) for _ in range(384)]
            except Exception as e:
                print(f"Embedding failed: {e}")
                fallback_used = True
                embedding = [random.uniform(-1, 1) for _ in range(384)]
        else:
            fallback_used = True
            embedding = [random.uniform(-1, 1) for _ in range(384)]
        chunk['embedding'] = embedding
        return chunk
    results = parallel_embed_chunks(chunks, embed_func, max_workers=4, timeout=120, progress_callback=progress_callback)
    progress.empty()
    if fallback_used:
        st.warning("Ollama did not return valid embeddings for some or all chunks. Fallback embeddings were used. Check logs for details.")
    return [r for r in results if r is not None]

def add_documents_to_chromadb(documents, source_name):
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

def list_loaded_documents_chroma():
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

# --- Sidebar for Uploading and Batch Ingest ---
st.sidebar.header("Add Content Source (ChromaDB)")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more documents",
    type=["pdf", "txt", "csv", "xlsx", "xls", "json", "db", "sqlite", "sqlite3", "sql", "py"],
    help="Supports PDF, TXT, CSV, Excel, JSON, SQLite, SQL, and Python files.",
    accept_multiple_files=True
)

if uploaded_files is not None:
    st.sidebar.success(f"Uploader ready. {len(uploaded_files)} file(s) selected.")
else:
    st.sidebar.info("Uploader ready. No files selected yet.")

url_input = st.sidebar.text_input(
    "Or enter a Web URL",
    placeholder="https://example.com"
)

db_uri_input = st.sidebar.text_input(
    "Or enter PostgreSQL DB URI (schema only)",
    placeholder="postgresql://user:pass@host:port/dbname"
)

if st.sidebar.button("Process Content Source (ChromaDB)"):
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
                db_uri=content_source_info.get("uri"),
                chunk_size=chunk_size
            )
        if documents:
            embedded_docs = embed_chunks_with_ollama(documents)
            display_name = source_name if source_name != "Unknown" else source_identifier
            with st.spinner(f"Adding {len(embedded_docs)} chunk(s) from {display_name} to ChromaDB..."):
                success, count = add_documents_to_chromadb(embedded_docs, display_name)
                if success:
                    st.sidebar.success(f"Successfully added {count} chunk(s) from {display_name} to ChromaDB!")
                    # --- Automatic skill extraction ---
                    from tools.ml_nlp_skill_identifier import identify_skills_nlp
                    for doc in embedded_docs:
                        code = doc['content']
                        skills = identify_skills_nlp(code)
                        skillset_cache.setdefault(display_name, {"Hybrid (Regex+NLP)": set()})
                        skillset_cache[display_name]["Hybrid (Regex+NLP)"] |= skills
                    # --- Save code content for this file ---
                    st.session_state.setdefault('file_code_content', {})
                    # Concatenate all chunks for this file
                    file_code = ''.join([doc['content'] for doc in embedded_docs if doc.get('content')])
                    st.session_state['file_code_content'][display_name] = file_code
                else:
                    st.sidebar.error(f"Failed to add content from {display_name}. Check console.")
        else:
            st.sidebar.error(f"Could not load or split content from {source_identifier}. Check console.")
    # Save extracted skills to JSON
    categorized_skills_list = []
    for fname, model_skills in skillset_cache.items():
        # Add code content if available
        code_content = st.session_state.get('file_code_content', {}).get(fname, '')
        categorized_skills_list.append({
            "filename": fname,
            "skills": {k: list(v) for k, v in model_skills.items()},
            "content": code_content
        })
    with open(os.path.join(BASE_DIR, 'downstream', 'skills_categorized.json'), 'w', encoding='utf-8') as f:
        json.dump(categorized_skills_list, f, indent=2)
    st.sidebar.success("Skills extracted and saved to downstream/skills_categorized.json!")
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

if st.sidebar.button("Batch Ingest from 'ingest' Folder (ChromaDB)"):
    ingest_folder = os.path.join(os.path.dirname(__file__), 'ingest')
    supported_files = list_supported_files(ingest_folder, ['.py', '.csv', '.txt', '.pdf', '.json', '.xlsx', '.xls'])
    for filename in supported_files:
        file_path = os.path.join(ingest_folder, filename)
        with st.spinner(f"Batch loading and splitting: {filename}"):
            documents, source_name = load_and_split_documents(file_path=file_path, chunk_size=chunk_size)
        if documents:
            embedded_docs = embed_chunks_with_ollama(documents)
            with st.spinner(f"Adding {len(embedded_docs)} chunk(s) from {filename} to ChromaDB..."):
                success, count = add_documents_to_chromadb(embedded_docs, filename)
                if success:
                    st.sidebar.success(f"Batch: Added {count} chunk(s) from {filename} to ChromaDB!")
                else:
                    st.sidebar.error(f"Batch: Failed to add content from {filename}. Check console.")
        else:
            st.sidebar.error(f"Batch: Could not load or split content from {filename}. Check console.")

# Show loaded ChromaDB sources
st.sidebar.markdown("--- ")
st.sidebar.subheader("Loaded Documents (ChromaDB)")
loaded_sources_chroma = list_loaded_documents_chroma()
if loaded_sources_chroma:
    for i, source in enumerate(loaded_sources_chroma):
        st.sidebar.text(f"{i+1}. {source}")
else:
    st.sidebar.text("No documents loaded yet in ChromaDB")
    st.sidebar.info("👆 Add content by uploading a file, entering a URL, or using batch ingest.")
st.sidebar.markdown("--- ")
st.sidebar.caption(f"Vector data is managed by ChromaDB")

# --- Web search function using cached skills ---
def run_skill_driven_web_search(skills_by_file):
    results = []
    for file, model_skills in skills_by_file.items():
        for model, skills in model_skills.items():
            for skill in skills:
                try:
                    search_results = search_duckduckgo(f"learn {skill}", max_results=3)
                    for r in search_results:
                        results.append({
                            'file': file,
                            'model': model,
                            'skill': skill,
                            'title': r.get('title', ''),
                            'url': r.get('url', ''),
                            'snippet': r.get('snippet', '')
                        })
                    time.sleep(1)  # Add a delay to avoid rate limiting
                except DuckDuckGoSearchException as e:
                    results.append({
                        'file': file,
                        'model': model,
                        'skill': skill,
                        'title': 'Rate limit hit or search failed',
                        'url': '',
                        'snippet': str(e)
                    })
    return results

# --- Stub: Pass cached skill sets to webscraping pipeline for targeted search ---
# def run_webscraping_with_skills(skillset_cache):
#     pass  # To be implemented in a future sprint 

# Global cache for skill sets by file and model (now in session state)
if "skill_web_refs" not in st.session_state:
    st.session_state["skill_web_refs"] = {}

st.title("Skills Gap Data Explorer")

# Sidebar navigation
view_option = st.sidebar.radio(
    "Select View:",
    ("Analytics", "Ingested Files", "Skill Categorization", "RAG Q&A", "Web References", "Generate Report")
)

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

ingested_files = load_json(INGESTED_PATH)
categorized_skills = load_json(CATEGORIZED_PATH)
web_references = load_json(WEBREF_PATH)
web_references_filtered = load_json(WEBREF_FILTERED_PATH)

def file_download_link(filepath, label):
    if os.path.exists(filepath):
        # st.download_button(label, f, file_name=os.path.basename(filepath))
        st.info(f"{label}: {filepath}")

def file_url_info(filepath, label):
    if os.path.exists(filepath):
        file_url = f"file://{filepath}"
        st.info(f"[{label}]({file_url}) (opens in a new tab)", icon="🌐")

if view_option == "Analytics":
    # --- Live Analytics from ChromaDB ---
    file_sources = list_loaded_documents_chroma()
    all_docs = collection.get(include=['documents', 'metadatas'])

    # Number of files ingested
    num_files = len(file_sources)
    st.metric("Total Files Ingested", num_files)

    # File types breakdown (from metadata or file extension)
    file_types = [os.path.splitext(src)[1] for src in file_sources]
    if file_types:
        type_counts = pd.Series(file_types).value_counts()
        st.subheader("File Types Breakdown (Live)")
        type_df = pd.DataFrame(type_counts).reset_index()
        type_df.columns = ["File Type", "Count"]
        st.dataframe(type_df, use_container_width=True)
        st.bar_chart(type_df.set_index("File Type"))

    # Chunk counts per file
    chunk_counts = {}
    for meta in all_docs['metadatas']:
        if meta and 'source' in meta:
            chunk_counts[meta['source']] = chunk_counts.get(meta['source'], 0) + 1
    if chunk_counts:
        st.subheader("Chunk Counts per File (Live)")
        chunk_df = pd.DataFrame(list(chunk_counts.items()), columns=["Filename", "Chunks"])
        st.dataframe(chunk_df, use_container_width=True)
        st.bar_chart(chunk_df.set_index("Filename"))

    # (Optional) You can keep or comment out the old static analytics code below

elif view_option == "Ingested Files":
    st.header("Ingested Files")
    if not ingested_files:
        st.info("No ingested files found.")
    else:
        filenames = [f["filename"] for f in ingested_files]
        selected_file = st.selectbox("Select a file to view its content:", filenames)
        file_data = next((f for f in ingested_files if f["filename"] == selected_file), None)
        if file_data:
            st.subheader(f"File: {file_data['filename']} ({file_data.get('filetype', 'unknown')})")
            if file_data['filetype'] == '.csv' and isinstance(file_data['content'], list):
                st.write("Preview (first 10 rows):")
                st.table(file_data['content'][:10])
            else:
                st.code(file_data['content'], language=file_data.get('filetype', 'unknown').replace('.', ''))

elif view_option == "Skill Categorization":
    st.header("Skill Categorization Results")
    # --- Debug output ---
    st.write("DEBUG: categorized_skills =", categorized_skills)
    st.write("DEBUG: skillset_cache =", skillset_cache)
    if not categorized_skills:
        st.info("No skill categorization results found.")
    else:
        # --- New: Table view for file, skills, and libraries ---
        table_rows = []
        for entry in categorized_skills:
            filename = entry.get('filename', 'unknown')
            skills = []
            libraries = set()
            # Try to get all skills from all models
            if 'skills' in entry:
                # If skills is a dict of model->skills
                if isinstance(entry['skills'], dict):
                    for model, skill_list in entry['skills'].items():
                        if isinstance(skill_list, list):
                            skills.extend(skill_list)
                elif isinstance(entry['skills'], list):
                    skills.extend(entry['skills'])
            # --- Extract libraries from content for .py files ---
            if filename.endswith('.py'):
                code = entry.get('content', '')
                if code:
                    import re
                    # Find all import and from ... import ... statements
                    import_lines = re.findall(r'^(?:from|import)\s+([\w\.]+)', code, re.MULTILINE)
                    for lib in import_lines:
                        # Only take the top-level package
                        libraries.add(lib.split('.')[0])
            table_rows.append({
                'file': filename,
                'skills': ', '.join(sorted(set(skills))),
                'libraries': ', '.join(sorted(libraries)) if libraries else ''
            })
        if table_rows:
            import pandas as pd
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True)
        filenames = [f["filename"] for f in categorized_skills]
        model_options = ["Hybrid (Regex+NLP)", "phi4:mini (Ollama)", "Code Llama (Ollama)"]
        selected_models = st.multiselect("Select model(s) for skill extraction:", model_options, default=model_options)
        tab1, tab2 = st.tabs(["Detail View", "Tabular View"])
        with tab1:
            selected_files = st.multiselect("Select one or more files to view their skills:", filenames, default=filenames)
            if st.button("Extract and Cache Skills"):
                with st.spinner("Extracting and caching skills for selected files and models..."):
                    for selected_file in selected_files:
                        skill_data = next((f for f in categorized_skills if f["filename"] == selected_file), None)
                        if skill_data:
                            code = skill_data.get('content', '')
                            skillset_cache.setdefault(selected_file, {})
                            if "Hybrid (Regex+NLP)" in selected_models:
                                from tools.ml_nlp_skill_identifier import identify_skills_nlp
                                hybrid_skills = identify_skills_nlp(code)
                                skillset_cache[selected_file]["Hybrid (Regex+NLP)"] = hybrid_skills
                            if "phi4:mini (Ollama)" in selected_models:
                                from ollama_rag import query_ollama
                                phi4_prompt = f"Given the following code, list the main programming skills, libraries, and topics it demonstrates. Return the answer as a comma-separated list of skills/topics only (no explanation):\n\nCODE:\n{code}"
                                phi4_response = query_ollama(phi4_prompt, model='phi4-mini:latest')
                                phi4_skills = [s.strip() for s in phi4_response.split(',') if s.strip()]
                                skillset_cache[selected_file]["phi4:mini (Ollama)"] = phi4_skills
                            if "Code Llama (Ollama)" in selected_models:
                                llama_response = query_codellama_ollama(code)
                                llama_skills = extract_skills_from_llama_response(llama_response)
                                skillset_cache[selected_file]["Code Llama (Ollama)"] = llama_skills
                st.success("Skill extraction and caching complete! You can now use the web search functionality.")
            # Display cached skills for selected files/models
            for selected_file in selected_files:
                skill_data = next((f for f in categorized_skills if f["filename"] == selected_file), None)
                if skill_data:
                    st.subheader(f"File: {skill_data['filename']} ({skill_data.get('filetype', 'unknown')})")
                    for model in selected_models:
                        if model in skillset_cache.get(selected_file, {}):
                            st.markdown(f"**{model} Skills:**")
                            st.write(skillset_cache[selected_file][model])
        with tab2:
            selected_files = st.multiselect("Select files for table view:", filenames, default=filenames, key="table_view_select")
            # Prepare data for each section
            functions_data = {f: next((x['skills'].get('functions', []) for x in categorized_skills if x['filename'] == f), []) for f in selected_files}
            libraries_data = {f: next((x['skills'].get('libraries', []) for x in categorized_skills if x['filename'] == f), []) for f in selected_files}
            comments_data = {f: next((x['skills'].get('comments', []) for x in categorized_skills if x['filename'] == f), []) for f in selected_files}
            st.markdown("### Functions Table")
            st.dataframe({fn: functions_data[fn] for fn in selected_files})
            st.markdown("### Libraries Table")
            st.dataframe({fn: libraries_data[fn] for fn in selected_files})
            st.markdown("### Comments Table")
            st.dataframe({fn: comments_data[fn] for fn in selected_files})
            # --- New: File Name and Library Table ---
            library_rows = []
            for f in selected_files:
                for lib in libraries_data[f]:
                    library_rows.append({"Filename": f, "Library": lib})
            if library_rows:
                st.markdown("### File Name and Libraries Used (One per Row)")
                st.dataframe(pd.DataFrame(library_rows))

elif view_option == "RAG Q&A":
    st.header("RAG Q&A (Ollama)")
    st.write("Ask questions or request summaries using the local Ollama model (phi3:mini or phi4-mini-reasoning:latest). Optionally, select a file to provide context.")
    context = ""
    # --- Check available Ollama models ---
    available_models = []
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            data = r.json()
            available_models = [m['name'] for m in data.get('models', [])]
    except Exception as e:
        st.warning(f"Could not connect to Ollama: {e}")
    # Choose a default model that exists
    preferred_models = ['phi3:mini', 'phi4-mini-reasoning:latest']
    model_to_use = next((m for m in preferred_models if m in available_models), None)
    if not model_to_use:
        st.warning("No suitable Ollama model found (phi3:mini or phi4-mini-reasoning:latest). Please pull a supported model.")
        model_to_use = preferred_models[0]  # fallback for UI
    else:
        st.info(f"Using Ollama model: {model_to_use}")
    if categorized_skills:
        filenames = [f["filename"] for f in categorized_skills]
        selected_file = st.selectbox("Select a file for context (optional):", ["None"] + filenames)
        if selected_file != "None":
            skill_data = next((f for f in categorized_skills if f["filename"] == selected_file), None)
            if skill_data:
                skills = skill_data['skills']
                context = f"\n\nContext from {skill_data['filename']} ({skill_data.get('filetype', 'unknown')}):\n"
                context += f"Functions: {', '.join(skills.get('functions', [])) or 'None'}\n"
                context += f"Libraries: {', '.join(skills.get('libraries', [])) or 'None'}\n"
                context += f"Comments: {' | '.join(skills.get('comments', [])) or 'None'}\n"
    user_prompt = st.text_area("Enter your question or prompt:")
    if st.button("Ask Ollama") and user_prompt.strip():
        # --- RAG: Retrieve relevant context from ChromaDB ---
        rag_context = ""
        try:
            # Embed the user prompt
            from ollama_rag import query_ollama
            question_embedding = query_ollama(user_prompt, model=EMBEDDING_MODEL)
            if not isinstance(question_embedding, list):
                import random
                question_embedding = [random.uniform(-1, 1) for _ in range(384)]
            # Query ChromaDB for top 3 relevant chunks
            query_kwargs = {
                'query_embeddings': [question_embedding],
                'n_results': 3,
                'include': ["documents", "metadatas"]
            }
            if selected_file != "None":
                query_kwargs['where'] = {"source": selected_file}
            results = collection.query(**query_kwargs)
            top_chunks = results.get('documents', [[]])[0]
            if top_chunks:
                rag_context = "\n\nRAG Context from your files:\n" + "\n---\n".join(top_chunks)
        except Exception as e:
            st.warning(f"RAG retrieval failed: {e}")
        # --- Combine RAG context, static context, and user prompt ---
        full_prompt = user_prompt + context + rag_context
        with st.spinner(f"Querying Ollama ({model_to_use}) with RAG context..."):
            response = query_ollama(full_prompt, model=model_to_use)
        st.markdown("**Ollama Response:**")
        st.write(response)

elif view_option == "Web References":
    st.header("Web References: Gold Standard & Self-Learning Platforms")
    ref_type = st.radio("Show:", ["All References", "Filtered (High-Quality) References", "Skill-Driven Web References"])
    refs = web_references if ref_type == "All References" else web_references_filtered

    # --- Gold Standard Providers Filter ---
    gold_csv_path = os.path.join(DOC_DIR, 'gold_standard_providers.csv')
    gold_brands = set()
    gold_institutions = set()
    if os.path.exists(gold_csv_path):
        import csv
        with open(gold_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader((row for row in f if not row.startswith('#')))
            for row in reader:
                if row['brand']:
                    gold_brands.add(row['brand'].strip().lower())
                if row['institution']:
                    gold_institutions.add(row['institution'].strip().lower())

    if ref_type == "Skill-Driven Web References":
        st.markdown("**Run a targeted web search using cached skills from skill categorization.**")
        # Let user select files to use for skill-driven search
        if not skillset_cache:
            st.info("No cached skills found. Run skill categorization first.")
        else:
            files_with_skills = list(skillset_cache.keys())
            selected_files = st.multiselect("Select files to use for skill-driven web search:", files_with_skills, default=files_with_skills)
            cache_key = tuple(sorted(selected_files))
            if st.button("Run Skill-Driven Web Search"):
                selected_skills = {f: skillset_cache[f] for f in selected_files}
                skill_refs = run_skill_driven_web_search(selected_skills)
                st.session_state["skill_web_refs"][cache_key] = skill_refs
            # Use cached results if available
            skill_refs = st.session_state["skill_web_refs"].get(cache_key, [])
            if skill_refs:
                st.success(f"Found {len(skill_refs)} skill-driven web references.")
                skill_ref_df = pd.DataFrame(skill_refs)
                skill_ref_df['url'] = skill_ref_df['url'].apply(lambda x: f"[link]({x})" if x else "")
                st.write(skill_ref_df[['file', 'model', 'skill', 'title', 'snippet', 'url']].rename(columns={'title': 'Title', 'snippet': 'Snippet', 'url': 'URL'}), unsafe_allow_html=True)
            else:
                st.info("No skill-driven web references found.")
    elif ref_type == "Filtered (High-Quality) References":
        # --- Skill-based filtering ---
        # Gather all unique skills from skills_categorized.json
        all_skills = set()
        for entry in categorized_skills:
            if 'skills' in entry and isinstance(entry['skills'], dict):
                for skill_list in entry['skills'].values():
                    all_skills.update(skill_list)
        all_skills = sorted(all_skills)
        selected_skills = st.multiselect("Filter by skill:", all_skills, default=all_skills[:1] if all_skills else [])

        # --- Provider selection and type filter ---
        # Parse gold standard providers CSV and categorize by type
        provider_rows = []
        provider_type = None
        if os.path.exists(gold_csv_path):
            with open(gold_csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        if 'Education' in line:
                            provider_type = 'Education'
                        elif 'Technology' in line:
                            provider_type = 'Technology'
                        elif 'Consultancy' in line:
                            provider_type = 'Consultancy'
                        else:
                            provider_type = None
                        continue
                    if line and not line.startswith('institution'):
                        parts = line.split(',')
                        if len(parts) == 2:
                            provider_rows.append({'institution': parts[0], 'brand': parts[1], 'type': provider_type})
        provider_types = sorted(set([row['type'] for row in provider_rows if row['type']]))
        selected_type = st.selectbox("Filter providers by type:", ["All"] + provider_types)
        filtered_providers = [row for row in provider_rows if selected_type == "All" or row['type'] == selected_type]
        provider_options = [f"{row['institution']} ({row['brand']})" for row in filtered_providers]
        selected_providers = st.multiselect(
            "Select up to 5 providers:",
            provider_options,
            max_selections=5
        )
        # Extract selected brands and institutions
        selected_brands = set()
        selected_institutions = set()
        for opt in selected_providers:
            for row in filtered_providers:
                if f"{row['institution']} ({row['brand']})" == opt:
                    selected_brands.add(row['brand'].strip().lower())
                    selected_institutions.add(row['institution'].strip().lower())

        # --- Run Web Search Button ---
        run_search = st.button("Run Web Search for Selected Skills and Providers")
        filtered_refs = []
        if run_search and selected_skills and (selected_brands or selected_institutions):
            # Run DuckDuckGo search for each (skill, provider) pair
            from tools.web_search import search_duckduckgo
            import time
            new_refs = []
            for skill in selected_skills:
                for brand in selected_brands:
                    query = f"{brand} {skill}"
                    try:
                        results = search_duckduckgo(query, max_results=3)
                        for r in results:
                            r['query'] = query
                            r['skill'] = skill
                            r['provider'] = brand
                            new_refs.append(r)
                        time.sleep(1)
                    except Exception as e:
                        new_refs.append({'title': 'Search failed', 'snippet': str(e), 'url': '', 'query': query, 'skill': skill, 'provider': brand})
                for institution in selected_institutions:
                    query = f"{institution} {skill}"
                    try:
                        results = search_duckduckgo(query, max_results=3)
                        for r in results:
                            r['query'] = query
                            r['skill'] = skill
                            r['provider'] = institution
                            new_refs.append(r)
                        time.sleep(1)
                    except Exception as e:
                        new_refs.append({'title': 'Search failed', 'snippet': str(e), 'url': '', 'query': query, 'skill': skill, 'provider': institution})
            # Save to web_references_filtered.json
            with open(WEBREF_FILTERED_PATH, 'w', encoding='utf-8') as f:
                import json
                json.dump(new_refs, f, indent=2)
            filtered_refs = new_refs
        else:
            # Load from file and filter by selected skills/providers
            if os.path.exists(WEBREF_FILTERED_PATH):
                import json
                with open(WEBREF_FILTERED_PATH, 'r', encoding='utf-8') as f:
                    all_refs = json.load(f)
                def is_gold_standard_and_skill_and_provider(ref):
                    text = ' '.join([str(ref.get(col, '')).lower() for col in ['title', 'snippet', 'url']])
                    gold = (not selected_brands and not selected_institutions) or any(b in text for b in selected_brands) or any(i in text for i in selected_institutions)
                    skill = any(s.lower() in text for s in selected_skills)
                    return gold and skill
                filtered_refs = [r for r in all_refs if is_gold_standard_and_skill_and_provider(r)]
        if filtered_refs:
            ref_df = pd.DataFrame(filtered_refs)
            ref_df['url'] = ref_df['url'].apply(lambda x: f"[link]({x})" if x else "")
            st.write(ref_df[['title', 'snippet', 'url']].rename(columns={'title': 'Title', 'snippet': 'Snippet', 'url': 'URL'}), unsafe_allow_html=True)
        else:
            st.info("No high-quality references found for the selected skill(s) or provider(s).")
    else:
        queries = sorted(set(r['query'] for r in refs if 'query' in r))
        selected_query = st.selectbox("Filter by topic/query:", ["All"] + queries)
        if selected_query == "All":
            filtered_refs = refs
        else:
            filtered_refs = [r for r in refs if r.get('query') == selected_query]
        search_term = st.text_input("Search across all columns:")
        if search_term:
            search_term_lower = search_term.lower()
            filtered_refs = [r for r in filtered_refs if any(search_term_lower in str(r.get(col, '')).lower() for col in ['title', 'snippet', 'url'])]
        if filtered_refs:
            ref_df = pd.DataFrame(filtered_refs)
            ref_df['url'] = ref_df['url'].apply(lambda x: f"[link]({x})" if x else "")
            st.write(ref_df[['title', 'snippet', 'url']].rename(columns={'title': 'Title', 'snippet': 'Snippet', 'url': 'URL'}), unsafe_allow_html=True)
        else:
            st.info("No references found for this topic or search.")

elif view_option == "Generate Report":
    st.header("Generate Skills Gap Report")
    st.write("Select the output format and generate a comprehensive report from the ingested and analyzed data.")
    report_format = st.radio("Select report format:", ["Markdown (.md)", "Quarto (.qmd)"])

    # --- Document selection for RAG Q&A and report generation ---
    all_filenames = [f["filename"] for f in categorized_skills]
    st.markdown("**Select which documents to include in RAG Q&A and report generation:**")
    selected_docs = st.multiselect("Documents:", all_filenames, default=all_filenames)
    st.info(f"{len(selected_docs)} document(s) selected for RAG Q&A and report generation.")

    # --- Skill-driven web references selection for report ---
    skill_web_refs = st.session_state.get("skill_web_refs", {})
    # Gather all cached references for selected files
    all_refs = []
    for cache_key, refs in skill_web_refs.items():
        # cache_key is a tuple of filenames
        if any(f in selected_docs for f in cache_key):
            all_refs.extend(refs)
    # Remove duplicates (by title+url+skill+file)
    seen = set()
    unique_refs = []
    for r in all_refs:
        key = (r.get('title',''), r.get('url',''), r.get('skill',''), r.get('file',''))
        if key not in seen:
            unique_refs.append(r)
            seen.add(key)
    # Let user select which references to include
    st.markdown("**Select which skill-driven web references to include in the report:**")
    if unique_refs:
        ref_options = [f"{r['file']} | {r['model']} | {r['skill']} | {r['title']}" for r in unique_refs]
        selected_ref_labels = st.multiselect("References:", ref_options, default=ref_options)
        selected_refs = [r for r, label in zip(unique_refs, ref_options) if label in selected_ref_labels]
    else:
        st.info("No cached skill-driven web references found for selected files.")
        selected_refs = []

    # If user doesn't select any, export all by default
    if not selected_refs:
        selected_refs = unique_refs

    # --- Sprint 3: Quarto Template Rendering & UI Integration ---
    st.subheader("Render Quarto Template (HTML/PDF)")
    from tools.render_quarto_template import render_quarto_template
    TEMPLATE_PATH = os.path.join(BASE_DIR, 'standards', 'skills_gap_template.qmd')
    TEMPLATE_HTML = os.path.join(DOC_DIR, 'skills_gap_template.html')
    TEMPLATE_PDF = os.path.join(DOC_DIR, 'skills_gap_template.pdf')
    TEMPLATE_QMD = os.path.join(DOC_DIR, 'skills_gap_template.qmd')
    # Copy template to doc dir if not present
    if not os.path.exists(TEMPLATE_QMD):
        with open(TEMPLATE_PATH, 'r', encoding='utf-8') as src, open(TEMPLATE_QMD, 'w', encoding='utf-8') as dst:
            dst.write(src.read())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Report Source Files**")
        if os.path.exists(MD_REPORT):
            st.info(f"Markdown Report (.md) saved at: {MD_REPORT}")
        if os.path.exists(QMD_REPORT):
            st.info(f"Quarto Report (.qmd) saved at: {QMD_REPORT}")
        if os.path.exists(TEMPLATE_QMD):
            st.info(f"Template Quarto File (.qmd) saved at: {TEMPLATE_QMD}")
    with col2:
        st.markdown("**Rendered Output Files**")
        if os.path.exists(HTML_REPORT):
            file_url_info(HTML_REPORT, "Open HTML Report in Browser")
        if os.path.exists(PDF_REPORT):
            file_url_info(PDF_REPORT, "Open PDF Report in Browser")
        if os.path.exists(TEMPLATE_HTML):
            file_url_info(TEMPLATE_HTML, "Open Template HTML in Browser")
        if os.path.exists(TEMPLATE_PDF):
            file_url_info(TEMPLATE_PDF, "Open Template PDF in Browser")

    show_links = st.radio("Show/Hide links to generated files:", ["Show links", "Hide links"], index=0)
    if show_links == "Hide links":
        st.info("Links to generated files are hidden. Use the radio button to show them.")
    else:
        st.success("Links to all generated and template files are shown above.")

    if st.button("Render Quarto Template (HTML/PDF)"):
        import threading
        def run_render():
            try:
                render_quarto_template(TEMPLATE_PATH, DOC_DIR)
            except Exception as e:
                st.error(f"Template rendering failed: {e}")
        threading.Thread(target=run_render, daemon=True).start()
        st.info("Template rendering started in the background. Refresh or wait for links to update.")

    # --- Existing report generation logic ---
    st.subheader("Generate Data Report (from project analysis)")
    if st.button("Generate Report"):
        with st.spinner("Generating report. This may take a minute..."):
            # Filter categorized_skills to only selected_docs
            selected_skills = [f for f in categorized_skills if f["filename"] in selected_docs]
            # Save filtered skills to a temp file for the report script
            import tempfile
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
                json.dump(selected_skills, tf)
                temp_skills_path = tf.name
            # Save selected web references to a temp file for the report script
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf_refs:
                json.dump(selected_refs, tf_refs)
                temp_refs_path = tf_refs.name
            # Pass the temp file paths to the report script via env var
            env = os.environ.copy()
            env["SKILLS_PATH"] = temp_skills_path
            env["WEBREFS_PATH"] = temp_refs_path
            if report_format == "Markdown (.md)":
                result = subprocess.run(["python3", os.path.join(TOOLS_DIR, "generate_document.py")], cwd=BASE_DIR, capture_output=True, text=True, env=env)
                st.success("Markdown report generated!")
                st.info(f"Markdown Report (.md) saved at: {MD_REPORT}")
            else:
                result = subprocess.run(["python3", os.path.join(TOOLS_DIR, "generate_document_qmd.py")], cwd=BASE_DIR, capture_output=True, text=True, env=env)
                st.success("Quarto report generated!")
                st.info(f"Quarto Report (.qmd) saved at: {QMD_REPORT}")
                file_url_info(HTML_REPORT, "Open HTML Report in Browser")
                file_url_info(PDF_REPORT, "Open PDF Report in Browser")
            st.text_area("Script Output", result.stdout + '\n' + result.stderr, height=200)
            os.remove(temp_skills_path)
            os.remove(temp_refs_path)
    else:
        if os.path.exists(MD_REPORT):
            st.info(f"Markdown Report (.md) saved at: {MD_REPORT}")
        if os.path.exists(QMD_REPORT):
            st.info(f"Quarto Report (.qmd) saved at: {QMD_REPORT}")
        if os.path.exists(HTML_REPORT):
            file_url_info(HTML_REPORT, "Open HTML Report in Browser")
        if os.path.exists(PDF_REPORT):
            file_url_info(PDF_REPORT, "Open PDF Report in Browser")

