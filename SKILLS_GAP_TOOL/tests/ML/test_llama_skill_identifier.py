from skills_gap.tools.ML.llama_skill_identifier import query_codellama_ollama, extract_skills_from_llama_response
from skills_gap.tools.ml_nlp_skill_identifier import identify_skills_nlp

def test_llama_vs_hybrid():
    code = '''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
X = pd.read_csv('data.csv')
X = X.dropna()
plt.scatter(X['a'], X['b'])
model = RandomForestClassifier().fit(X, [0,1,0,1])
'''
    llama_response = query_codellama_ollama(code)
    llama_skills = extract_skills_from_llama_response(llama_response)
    print(f"Code Llama skills: {llama_skills}")
    hybrid_skills = identify_skills_nlp(code)
    print(f"Hybrid approach skills: {hybrid_skills}")
    # No assert: this is for comparison and lessons learned 