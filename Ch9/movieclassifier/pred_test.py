# Test predictions and pickling
import pickle
import re
import os
from vectorizer import vect

# Load model from serialization
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

# Make prediction based on example/sample
import numpy as np

label = {0:'negative', 1:'positive'}

# Should be positive
example = ['I love this movie']
X = vect.transform(example)

print(f"Text: {example}")
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
print()

# Should be negative
example = ['I thought this movie was horrendously terrible']
X = vect.transform(example)

print(f"Text: {example}")
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
