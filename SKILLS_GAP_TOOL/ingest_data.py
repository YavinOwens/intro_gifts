import os
from typing import List

# Define the local folder path for data ingestion
INGEST_FOLDER = os.path.join(os.path.dirname(__file__), 'ingest')

# Define the downstream/doc directory
DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'downstream', 'doc')
os.makedirs(DOC_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = ['.py', '.csv', '.txt']

def list_supported_files(folder_path: str, extensions: List[str]) -> List[str]:
    """
    List all files in the folder_path with the given extensions.
    """
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(file)
    return files

if __name__ == "__main__":
    print(f"Listing supported files in: {INGEST_FOLDER}")
    if not os.path.exists(INGEST_FOLDER):
        print(f"Folder does not exist: {INGEST_FOLDER}")
    else:
        files = list_supported_files(INGEST_FOLDER, SUPPORTED_EXTENSIONS)
        if files:
            print("Supported files found:")
            for f in files:
                print(f"- {f}")
        else:
            print("No supported files found.") 