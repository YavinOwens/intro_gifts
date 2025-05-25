import subprocess
import json

def query_codellama_ollama(code: str, model: str = 'codellama:latest') -> str:
    prompt = (
        """
Given the following code, list the main programming skills, libraries, and topics it demonstrates. 
Return the answer as a comma-separated list of skills/topics only (no explanation):

CODE:
""" + code.strip()
    )
    result = subprocess.run([
        "ollama", "run", model, prompt
    ], capture_output=True, text=True)
    return result.stdout.strip()

def extract_skills_from_llama_response(response: str):
    # Split by comma and clean up
    return [s.strip() for s in response.split(',') if s.strip()]

if __name__ == "__main__":
    code = '''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
X = pd.read_csv('data.csv')
X = X.dropna()
plt.scatter(X['a'], X['b'])
model = RandomForestClassifier().fit(X, [0,1,0,1])
'''
    response = query_codellama_ollama(code)
    print("Raw Llama response:", response)
    skills = extract_skills_from_llama_response(response)
    print("Extracted skills:", skills) 