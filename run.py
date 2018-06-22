#!/usr/bin/env python
from localtype import app

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from lime.lime_text import LimeTextExplainer
import dill

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = SnowballStemmer('english')

try:
    base_dir = '/home/ubuntu/localtype_site/localtype/data/'
    tfidf_vectorizer = dill.load(open(base_dir+'tfidf_vectorizer.m', 'rb'))
    c = dill.load(open(base_dir+"trained_pipeline.m", 'rb'))
except:
    print('first import failed, trying secondary import')
    base_dir = '/Users/nknezek/Documents/Insight_local/localtype_site/localtype/data/'
    tfidf_vectorizer = dill.load(open(base_dir + 'tfidf_vectorizer.m', 'rb'))
    c = dill.load(open(base_dir + "trained_pipeline.m", 'rb'))


app.run(debug=True)
