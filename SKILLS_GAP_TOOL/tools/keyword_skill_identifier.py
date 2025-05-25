KEYWORD_SKILLS = {
    'data cleaning': ['dropna', 'fillna', 'replace', 'isnull', 'notnull', 'clean', 'remove outlier', 'deduplicate'],
    'visualization': ['plot', 'scatter', 'bar', 'hist', 'seaborn', 'matplotlib', 'show', 'figure', 'chart'],
    'ml modeling': ['fit', 'predict', 'transform', 'train_test_split', 'model', 'regression', 'classification', 'RandomForest', 'SVC', 'KMeans'],
    'data loading': ['read_csv', 'read_excel', 'read_json', 'to_csv', 'to_excel', 'loadtxt', 'open', 'pandas.read'],
    'feature engineering': ['feature', 'encode', 'scale', 'normalize', 'onehot', 'LabelEncoder', 'StandardScaler'],
    'statistics': ['mean', 'median', 'std', 'var', 'describe', 'correlation', 'covariance', 'stats'],
    'nlp': ['tokenize', 'stem', 'lemmatize', 'nltk', 'spacy', 'textblob', 'word2vec', 'bert', 'transformer'],
    'deep learning': ['keras', 'tensorflow', 'torch', 'nn.', 'Sequential', 'Dense', 'Conv2D', 'LSTM', 'GRU'],
}

def identify_skills_from_code(code: str):
    """
    Identify skills/topics in code using a simple keyword-based approach.
    Returns a set of matched skill/topic categories.
    """
    code_lower = code.lower()
    matched = set()
    for skill, keywords in KEYWORD_SKILLS.items():
        for kw in keywords:
            if kw.lower() in code_lower:
                matched.add(skill)
                break
    return list(matched)

if __name__ == "__main__":
    # Example usage
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Data loading
X = pd.read_csv('data.csv')
X = X.dropna()
plt.scatter(X['a'], X['b'])
"""
    print(identify_skills_from_code(code)) 