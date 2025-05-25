import pytest
from skills_gap.tools.keyword_skill_identifier import identify_skills_from_code

def test_identify_skills_basic():
    code = """
import pandas as pd
X = pd.read_csv('data.csv')
X = X.dropna()
"""
    skills = identify_skills_from_code(code)
    print(f"test_identify_skills_basic: {skills}")
    assert 'data loading' in skills
    assert 'data cleaning' in skills
    assert 'visualization' not in skills

def test_identify_skills_multiple():
    code = """
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.scatter([1,2,3], [4,5,6])
model = RandomForestClassifier().fit([[1,2],[3,4]], [0,1])
"""
    skills = identify_skills_from_code(code)
    print(f"test_identify_skills_multiple: {skills}")
    assert 'ml modeling' in skills
    assert 'visualization' in skills
    assert 'data loading' not in skills

def test_identify_skills_nlp():
    code = """
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test sentence.')
"""
    skills = identify_skills_from_code(code)
    print(f"test_identify_skills_nlp: {skills}")
    assert 'nlp' in skills
    assert 'deep learning' not in skills 