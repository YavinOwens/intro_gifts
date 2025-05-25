import requests
import os

def query_ollama(prompt: str, model: str = 'phi4:latest', base_url: str = 'http://localhost:11434') -> str:
    """
    Send a prompt to the local Ollama instance and return the response.
    """
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get('response', '').strip()
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def is_ollama_running(base_url: str = 'http://localhost:11434') -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

if __name__ == "__main__":
    if not is_ollama_running():
        print("WARNING: Ollama is not running at http://localhost:11434. Embedding and LLM queries will fail.")
    # Example usage
    prompt = "Summarize the importance of data engineering in one paragraph."
    print("Prompt:", prompt)
    print("Response:")
    print(query_ollama(prompt)) 