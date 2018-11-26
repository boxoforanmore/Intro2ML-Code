# Out-Of-Core Learning with Stochastic Gradient Descent
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

input_file = 'movie_data.csv'

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Generator that reads and returns one document at a time
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# Verify stream_docs works correctly (should be a tuply with class label)
print(next(stream_docs(path=input_file)))
print()

# Fetchs a specific number of documents from the document stream
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# CountVectorizer cannot be used for out-of-core (has to hold whole library in memory)
# Use HashingVectorizer instead; is also data-independent
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# High number of features reduces chances of hashing collision
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path=input_file)

# Learn and estimate progress with PyPrind
# Estimating 45 iterations
import pyprind

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

print()

# Evaluate the performance
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)

print('Accuracy: %.3f' % clf.score(X_test, y_test))
print()


# Use the last 5000 documents to update the model
clf = clf.partial_fit(X_test, y_test)
