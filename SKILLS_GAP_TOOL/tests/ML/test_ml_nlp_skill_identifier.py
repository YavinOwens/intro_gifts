from skills_gap.tools.ml_nlp_skill_identifier import identify_skills_nlp

def test_nlp_identify_skills_basic():
    code = """
import pandas as pd
X = pd.read_csv('data.csv')
X = X.dropna()
"""
    skills = identify_skills_nlp(code)
    print(f"test_nlp_identify_skills_basic: {skills}")
    assert 'data loading' in skills or 'data cleaning' in skills

def test_nlp_identify_skills_multiple():
    code = """
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.scatter([1,2,3], [4,5,6])
model = RandomForestClassifier().fit([[1,2],[3,4]], [0,1])
# This model does classification and regression
"""
    skills = identify_skills_nlp(code)
    print(f"test_nlp_identify_skills_multiple: {skills}")
    assert 'ml modeling' in skills or 'classification' in skills or 'regression' in skills
    assert 'visualization' in skills

def test_nlp_identify_skills_nlp():
    code = """
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test sentence.')
# This is an nlp task
"""
    skills = identify_skills_nlp(code)
    print(f"test_nlp_identify_skills_nlp: {skills}")
    assert 'nlp' in skills 