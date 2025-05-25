import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ollama_rag import query_ollama

def test_query_ollama():
    prompt = "Explain what a data pipeline is in simple terms."
    print("Prompt:", prompt)
    response = query_ollama(prompt, model='phi4-mini:latest')
    print("Response:", response)

if __name__ == "__main__":
    test_query_ollama() 