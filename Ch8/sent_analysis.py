import pyprind  # PyPrind is the Python Process Indicator
import pandas as pd
import os

# Change the 'basepath' to the directory of the
# unzipped movie dataset
basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']


# Shuffling the DataFrame
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
input_file = 'movie_data.csv'
df.to_csv(input_file, index=False, encoding='utf=8')


# Confirming the data has been saved
df = pd.read_csv(input_file, encoding='utf-8')
df.head()

print("Data in DataFrame (should be (50000, 2)):")
print(df.shape)
print()
print()


# Show the last 50 characters from the first doc
# to illustrate why the text data needs to be cleaned
print("Last 50 characters:")
print(df.loc[0, 'review'][-50:])


# Use regex to strip html and emoticons and place emoticons at end
# It also decapitalizes everything
import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


# Testing the preprocessor
print()
print("Last 50 characters trimmed:")
print(preprocessor(df.loc[0, 'review'][-50:]))
print()
print("Testing regex:")
print(preprocessor("</a>This :) is :( a test :-)!"))
print()


# Apply preprocessor to DataFrame
df['review'] = df['review'].apply(preprocessor)


# Divide the DataFrame into test and training sets; 25000 each
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# Stopwords and PorterStemmer
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Use GridSearchCV to set optimal parameters
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words':  [stop, None],
               'vect__tokenizer':   [tokenizer, tokenizer_porter],
               'clf__penalty':      ['l1', 'l2'],
               'clf__C':            [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words':  [stop, None],
               'vect__tokenizer':   [tokenizer, tokenizer_porter],
               'vect__use_idf':     [False],
               'vect__norm':        [None],
               'clf__penalty':      ['l1', 'l2'],
               'clf__C':            [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])

# Set n_jobs to -1 to utilize all available cores
###### -1 breaks pickling; set to 1--pickling for process is not available for multiprocessing in current lib
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=1)

gs_lr_tfidf.fit(X_train, y_train)


print("Best parameter set from GridSearchCV: %s " % gs_lr_tfidf.best_params_)
print("CV Accuracy: %.3f" % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_

print("Test Accuracy: %.3f" % clf.score(X_test, y_test))

