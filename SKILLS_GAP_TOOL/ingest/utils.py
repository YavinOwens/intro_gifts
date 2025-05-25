# utils.py - Minimal stubs for SKILLS_GAP_TOOL/ingest/app.py

import os
import pdfplumber
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

EMBEDDING_MODEL = 'nomic-embed-text'
LLM_MODEL = 'phi3:mini'
DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'downstream', 'doc')
os.makedirs(DOC_DIR, exist_ok=True)


def load_and_split_documents(file_path=None, url=None, db_uri=None, chunk_size=1000):
    """
    Loads and splits documents from file, URL, or DB.
    Returns (documents, source_name).
    """
    documents = []
    source_name = None

    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        source_name = os.path.basename(file_path)
        try:
            if ext == '.py':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                import_lines = [line for line in lines if line.strip().startswith('import') or line.strip().startswith('from')]
                other_lines = [line for line in lines if line not in import_lines]
                if import_lines:
                    documents.append({'content': ''.join(import_lines), 'metadata': {'source': source_name, 'chunk_type': 'imports'}})
                code = ''.join(other_lines)
                code_chunks = [code[i:i+chunk_size] for i in range(0, len(code), chunk_size)]
                for chunk in code_chunks:
                    documents.append({'content': chunk, 'metadata': {'source': source_name, 'chunk_type': 'code'}})
            elif ext in ['.txt', '.sql', '.json', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                for chunk in chunks:
                    documents.append({'content': chunk, 'metadata': {'source': source_name}})
            elif ext in ['.csv']:
                df = pd.read_csv(file_path)
                content = df.to_csv(index=False)
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                for chunk in chunks:
                    documents.append({'content': chunk, 'metadata': {'source': source_name}})
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                content = df.to_csv(index=False)
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                for chunk in chunks:
                    documents.append({'content': chunk, 'metadata': {'source': source_name}})
            elif ext in ['.pdf']:
                try:
                    with pdfplumber.open(file_path) as pdf:
                        text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                    if not text.strip():
                        raise ValueError('No extractable text found in PDF.')
                    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                    for chunk in chunks:
                        documents.append({'content': chunk, 'metadata': {'source': source_name}})
                except Exception as e:
                    print(f"PDF extraction failed: {e}")
                    return [], None
            else:
                print(f"Unsupported file type: {ext}")
                return [], None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return [], None

    elif url:
        source_name = url
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text(separator='\n')
            if not text.strip():
                raise ValueError('No extractable text found at URL.')
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            for chunk in chunks:
                documents.append({'content': chunk, 'metadata': {'source': source_name}})
        except Exception as e:
            print(f"URL extraction failed: {e}")
            return [], None

    elif db_uri:
        source_name = db_uri
        try:
            if db_uri.startswith('sqlite') or db_uri.endswith('.db') or db_uri.endswith('.sqlite'):
                conn = sqlite3.connect(db_uri.split('///')[-1])
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                schema = ''
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    schema += f"Table: {table_name}\nColumns: {[col[1] for col in columns]}\n"
                conn.close()
                chunks = [schema[i:i+chunk_size] for i in range(0, len(schema), chunk_size)]
                for chunk in chunks:
                    documents.append({'content': chunk, 'metadata': {'source': source_name}})
            elif db_uri.startswith('postgresql'):
                import psycopg2
                import re
                m = re.match(r'postgresql://([^:]+):([^@]+)@([^:/]+):(\d+)/(\w+)', db_uri)
                if not m:
                    raise ValueError('Invalid PostgreSQL URI format.')
                user, password, host, port, dbname = m.groups()
                conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
                cursor = conn.cursor()
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                tables = cursor.fetchall()
                schema = ''
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';")
                    columns = cursor.fetchall()
                    schema += f"Table: {table_name}\nColumns: {[col[0] for col in columns]}\n"
                conn.close()
                chunks = [schema[i:i+chunk_size] for i in range(0, len(schema), chunk_size)]
                for chunk in chunks:
                    documents.append({'content': chunk, 'metadata': {'source': source_name}})
            else:
                print(f"Unsupported DB URI: {db_uri}")
                return [], None
        except Exception as e:
            print(f"DB extraction failed: {e}")
            return [], None

    else:
        print("No file_path, url, or db_uri provided.")
        return [], None

    return documents, source_name


def check_ollama_availability():
    """
    Stub for checking Ollama server and model availability.
    Returns (overall_ok, server_running, models_status, error_msg).
    """
    # TODO: Implement actual Ollama check
    return True, True, {EMBEDDING_MODEL: True, LLM_MODEL: True}, ''


def get_rag_chain():
    """
    Stub for getting a RAG chain object for document Q&A.
    Should return an object with an .invoke(prompt) method.
    """
    class DummyRAG:
        def invoke(self, prompt):
            return f"Dummy RAG response for: {prompt}"
    return DummyRAG()


def should_use_web_search(prompt, rag_response):
    """
    Stub for deciding if a web search should be performed.
    """
    # TODO: Implement actual logic
    return False


def run_web_search(prompt):
    """
    Stub for running a web search and returning results.
    """
    # TODO: Implement actual web search
    return "No web search results (stub)."


# Parallel embedding utility with timeout
def parallel_embed_chunks(chunks, embed_func, max_workers=4, timeout=60, progress_callback=None):
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(embed_func, chunk): idx for idx, chunk in enumerate(chunks)}
        for i, future in enumerate(as_completed(future_to_idx, timeout=timeout)):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=timeout)
                results[idx] = result
            except Exception as e:
                print(f"Embedding failed for chunk {idx}: {e}")
                results[idx] = None
            if progress_callback:
                progress_callback(i+1, len(chunks))
    return results 