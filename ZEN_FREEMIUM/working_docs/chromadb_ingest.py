import os
import glob
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sys

CSV_DIR = os.path.join(os.path.dirname(__file__), 'csv')
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../chromadb_store'))
COLLECTION_NAME = 'csv_docs'
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Ensure CHROMA_DIR exists before initializing ChromaDB client
try:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    print(f"Ensured CHROMA_DIR exists: {CHROMA_DIR}")
except Exception as e:
    print(f"Error creating CHROMA_DIR: {e}")

# Initialize ChromaDB persistent client (new API)
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL)

# Create or get collection
collection = client.get_or_create_collection(COLLECTION_NAME)

def ingest_csvs():
    csv_files = glob.glob(os.path.join(CSV_DIR, '*.csv'))
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, dtype=str)
        filename = os.path.basename(csv_path)
        for idx, row in df.iterrows():
            doc_id = f"{filename}-{idx}"
            text = row.to_json()
            try:
                embedding = embedder.encode([text])[0]
                collection.add(
                    documents=[text],
                    metadatas=[{"filename": filename, "row": idx}],
                    ids=[doc_id],
                    embeddings=[embedding]
                )
                print(f"[CSV] Added: id={doc_id}, content_len={len(text)}")
            except Exception as e:
                print(f"[CSV] Failed to add: id={doc_id}, content_len={len(text)}, error={e}")
    print(f"Ingested {len(csv_files)} CSV files into ChromaDB at {CHROMA_DIR}")

def ingest_workforce_files():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'workforce/Department'))
    total_files = 0
    for dept in os.listdir(base_dir):
        dept_path = os.path.join(base_dir, dept)
        if not os.path.isdir(dept_path):
            continue
        dept_file_count = 0
        # Ingest files in files/
        files_dir = os.path.join(dept_path, 'files')
        if os.path.isdir(files_dir):
            for fname in os.listdir(files_dir):
                fpath = os.path.join(files_dir, fname)
                file_type = os.path.splitext(fname)[1].lstrip('.')
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception:
                    try:
                        with open(fpath, 'rb') as f:
                            content = str(f.read())
                    except Exception:
                        content = ''
                doc_id = f"{dept}-files-{fname}"
                try:
                    embedding = embedder.encode([content])[0]
                    collection.add(
                        documents=[content],
                        metadatas=[{
                            "department": dept,
                            "file_type": file_type,
                            "filename": fname,
                            "relative_path": os.path.relpath(fpath, base_dir)
                        }],
                        ids=[doc_id],
                        embeddings=[embedding]
                    )
                    print(f"[WORKFORCE] Added: id={doc_id}, content_len={len(content)}")
                except Exception as e:
                    print(f"[WORKFORCE] Failed to add: id={doc_id}, content_len={len(content)}, error={e}")
                dept_file_count += 1
        # Ingest people.csv in people/
        people_dir = os.path.join(dept_path, 'people')
        people_csv = os.path.join(people_dir, 'people.csv')
        if os.path.isfile(people_csv):
            try:
                with open(people_csv, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                content = ''
            doc_id = f"{dept}-people-people.csv"
            try:
                embedding = embedder.encode([content])[0]
                collection.add(
                    documents=[content],
                    metadatas=[{
                        "department": dept,
                        "file_type": 'csv',
                        "filename": 'people.csv',
                        "relative_path": os.path.relpath(people_csv, base_dir)
                    }],
                    ids=[doc_id],
                    embeddings=[embedding]
                )
                print(f"[WORKFORCE] Added: id={doc_id}, content_len={len(content)}")
            except Exception as e:
                print(f"[WORKFORCE] Failed to add: id={doc_id}, content_len={len(content)}, error={e}")
            dept_file_count += 1
        print(f"Ingested {dept_file_count} files for department '{dept}'")
        total_files += dept_file_count
    print(f"Ingested a total of {total_files} department files into ChromaDB at {CHROMA_DIR}")

def main():
    ingest_csvs()
    ingest_workforce_files()

def print_chromadb_files():
    print('--- Files in ChromaDB ---')
    metas = collection.get(include=['metadatas'])['metadatas']
    count = 0
    for meta in metas:
        if meta and 'filename' in meta:
            print(f"{meta.get('filename')} (type: {meta.get('file_type', 'unknown')}, dept: {meta.get('department', '-')}, path: {meta.get('relative_path', '-')})")
            count += 1
    print(f"Total files in ChromaDB: {count}")

def test_add_and_list():
    print('--- Testing ChromaDB Add/List ---')
    test_id = 'test-doc'
    test_content = 'This is a test document.'
    test_embedding = embedder.encode([test_content])[0]
    collection.add(
        documents=[test_content],
        metadatas=[{'filename': 'test.txt', 'file_type': 'txt', 'department': 'test', 'relative_path': 'test.txt'}],
        ids=[test_id],
        embeddings=[test_embedding]
    )
    print_chromadb_files()

def print_debug_info():
    print(f"CHROMA_DIR absolute path: {CHROMA_DIR}")
    print(f"ChromaDB version: {getattr(chromadb, '__version__', 'unknown')}")
    try:
        backend = client._settings.get('chroma_db_impl', 'unknown')
        print(f"ChromaDB backend: {backend}")
    except Exception as e:
        print(f"Could not get backend: {e}")
    print("Contents of CHROMA_DIR:")
    try:
        for root, dirs, files in os.walk(CHROMA_DIR):
            print(f"{root}/")
            for d in dirs:
                print(f"  [dir] {d}")
            for f in files:
                print(f"  [file] {f}")
    except Exception as e:
        print(f"Error listing CHROMA_DIR: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_add_and_list()
    elif len(sys.argv) > 1 and sys.argv[1] == '--list':
        print_chromadb_files()
    elif len(sys.argv) > 1 and sys.argv[1] == '--debug':
        print_debug_info()
    else:
        main()
        print_debug_info() 