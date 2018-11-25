import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Change CountVectorizer's ngram_range to (2,2) or something else
# to change the n-gram if multilevel sentiment analysis is desired
count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)


print("The contents of the vocabulary")
print(count.vocabulary_)
print()
print("The feature vectors/columns:")
print(bag.toarray())


# Use the tfid transformer to rebalance the weights to favor
# the more infrequent words
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)

print()
print("The new feature vectors after tfid balancing:")
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
print()
