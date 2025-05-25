import os
import re
import spacy
from typing import List, Set

# Ensure spaCy model is downloaded to a local directory
LOCAL_SPACY_DIR = os.path.join(os.path.dirname(__file__), '..', '.spacy')
os.environ['SPACY_DATA'] = os.path.abspath(LOCAL_SPACY_DIR)

def ensure_spacy_model(model_name='en_core_web_sm'):
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name, direct=True)
        return spacy.load(model_name)

# Example skills/topics list (can be expanded)
SKILL_TOPICS = [
    'data cleaning', 'visualization', 'ml modeling', 'data loading',
    'feature engineering', 'statistics', 'nlp', 'deep learning',
    'regression', 'classification', 'clustering', 'tokenization', 'embedding',
    'normalization', 'scaling', 'encoding', 'transformer', 'neural network',
    'data engineering', 'etl', 'api', 'web development', 'front end', 'back end', 'fullstack',
    'database', 'cloud', 'devops', 'testing', 'deployment', 'authentication', 'authorization',
]

# Expanded mapping from code tokens to skill categories
CODE_TOKEN_TO_SKILL = {
    # --- Visualization ---
    'plt': 'visualization', 'matplotlib': 'visualization', 'seaborn': 'visualization',
    'plot': 'visualization', 'scatter': 'visualization', 'bar': 'visualization',
    'hist': 'visualization', 'show': 'visualization', 'figure': 'visualization',
    'chart': 'visualization', 'bokeh': 'visualization', 'altair': 'visualization',
    'plotly': 'visualization', 'heatmap': 'visualization',
    # --- Data Loading & Cleaning ---
    'read_csv': 'data loading', 'read_excel': 'data loading', 'read_json': 'data loading',
    'to_csv': 'data loading', 'to_excel': 'data loading', 'loadtxt': 'data loading',
    'dropna': 'data cleaning', 'fillna': 'data cleaning', 'replace': 'data cleaning',
    'isnull': 'data cleaning', 'notnull': 'data cleaning', 'drop_duplicates': 'data cleaning',
    'merge': 'data engineering', 'concat': 'data engineering', 'join': 'data engineering',
    # --- Feature Engineering ---
    'feature': 'feature engineering', 'encode': 'feature engineering', 'scale': 'feature engineering',
    'normalize': 'feature engineering', 'onehot': 'feature engineering', 'LabelEncoder': 'feature engineering',
    'StandardScaler': 'feature engineering', 'MinMaxScaler': 'feature engineering',
    # --- Statistics ---
    'mean': 'statistics', 'median': 'statistics', 'std': 'statistics', 'var': 'statistics',
    'describe': 'statistics', 'correlation': 'statistics', 'covariance': 'statistics', 'stats': 'statistics',
    # --- ML Modeling ---
    'fit': 'ml modeling', 'predict': 'ml modeling', 'transform': 'ml modeling',
    'train_test_split': 'ml modeling', 'model': 'ml modeling', 'regression': 'regression',
    'classification': 'classification', 'RandomForestClassifier': 'ml modeling',
    'SVC': 'ml modeling', 'KMeans': 'clustering', 'LogisticRegression': 'ml modeling',
    'DecisionTreeClassifier': 'ml modeling', 'GradientBoostingClassifier': 'ml modeling',
    'XGBClassifier': 'ml modeling', 'LGBMClassifier': 'ml modeling',
    # --- Deep Learning ---
    'keras': 'deep learning', 'tensorflow': 'deep learning', 'torch': 'deep learning',
    'nn': 'deep learning', 'Sequential': 'deep learning', 'Dense': 'deep learning',
    'Conv2D': 'deep learning', 'LSTM': 'deep learning', 'GRU': 'deep learning',
    'embedding': 'embedding', 'transformer': 'transformer', 'bert': 'transformer',
    # --- NLP ---
    'tokenize': 'nlp', 'stem': 'nlp', 'lemmatize': 'nlp', 'nltk': 'nlp', 'spacy': 'nlp',
    'textblob': 'nlp', 'word2vec': 'nlp', 'gensim': 'nlp',
    # --- Data Engineering / ETL ---
    'airflow': 'data engineering', 'luigi': 'data engineering', 'prefect': 'data engineering',
    'extract': 'etl', 'transform': 'etl', 'load': 'etl', 'pipeline': 'data engineering',
    'dag': 'data engineering', 'spark': 'data engineering', 'pyspark': 'data engineering',
    'dbt': 'data engineering', 'bigquery': 'data engineering', 'redshift': 'data engineering',
    'snowflake': 'data engineering', 'hadoop': 'data engineering',
    # --- API / Web Development ---
    'flask': 'api', 'fastapi': 'api', 'django': 'web development', 'bottle': 'api',
    'request': 'api', 'response': 'api', 'endpoint': 'api', 'route': 'api',
    'rest': 'api', 'graphql': 'api',
    # --- Front End ---
    'html': 'front end', 'css': 'front end', 'javascript': 'front end', 'react': 'front end',
    'vue': 'front end', 'angular': 'front end', 'svelte': 'front end', 'bootstrap': 'front end',
    'materialui': 'front end', 'tailwind': 'front end',
    # --- Back End ---
    'node': 'back end', 'express': 'back end', 'spring': 'back end', 'java': 'back end',
    'csharp': 'back end', 'dotnet': 'back end', 'php': 'back end', 'ruby': 'back end',
    'rails': 'back end', 'go': 'back end', 'gin': 'back end', 'mysql': 'database',
    'postgres': 'database', 'mongodb': 'database', 'sqlite': 'database', 'redis': 'database',
    # --- Fullstack ---
    'fullstack': 'fullstack', 'nextjs': 'fullstack', 'nuxt': 'fullstack',
    # --- Cloud / DevOps ---
    'aws': 'cloud', 'azure': 'cloud', 'gcp': 'cloud', 'docker': 'devops', 'kubernetes': 'devops',
    'terraform': 'devops', 'ansible': 'devops', 'jenkins': 'devops', 'ci': 'devops', 'cd': 'devops',
    # --- Testing / Deployment / Auth ---
    'pytest': 'testing', 'unittest': 'testing', 'test': 'testing', 'deploy': 'deployment',
    'authentication': 'authentication', 'authorization': 'authorization', 'jwt': 'authentication',
    'oauth': 'authentication', 'login': 'authentication', 'signup': 'authentication',
}

nlp = ensure_spacy_model()

def extract_code_tokens(code: str) -> Set[str]:
    # Extract identifiers, function names, and comments
    tokens = set()
    # Extract words, function calls, and variable names
    tokens.update(re.findall(r'\b\w+\b', code))
    # Extract function calls (e.g., foo(), bar.fit())
    tokens.update(re.findall(r'(\w+)\s*\(', code))
    # Extract comments
    comments = re.findall(r'#.*', code)
    return tokens, comments

def identify_skills_nlp(code: str) -> Set[str]:
    """
    Identify skills/topics in code using regex for code tokens, code-token-to-skill mapping, and spaCy for comments/docstrings.
    Returns a set of matched skills/topics.
    """
    tokens, comments = extract_code_tokens(code)
    tokens = set(t.lower() for t in tokens)
    found = set()
    # Check for direct matches in code tokens
    for skill in SKILL_TOPICS:
        skill_lower = skill.lower()
        if any(kw in tokens for kw in skill_lower.split()):
            found.add(skill)
    # Check code tokens against CODE_TOKEN_TO_SKILL mapping
    for token in tokens:
        mapped_skill = CODE_TOKEN_TO_SKILL.get(token)
        if mapped_skill:
            found.add(mapped_skill)
    # Use spaCy for comments/docstrings
    for comment in comments:
        doc = nlp(comment)
        comment_tokens = set([t.text.lower() for t in doc if not t.is_stop and not t.is_punct])
        for skill in SKILL_TOPICS:
            skill_lower = skill.lower()
            if skill_lower in comment.lower() or any(kw in comment_tokens for kw in skill_lower.split()):
                found.add(skill)
    return found

if __name__ == "__main__":
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# Data loading and cleaning
X = pd.read_csv('data.csv')
X = X.dropna()
plt.scatter(X['a'], X['b'])
# This model does classification and regression
"""
    print("Sample code for skill extraction test:")
    print(code)
    print("Extracted skills:")
    print(identify_skills_nlp(code)) 