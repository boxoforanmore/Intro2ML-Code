# Decomposing text documents with LDA

import pandas as pd

input_file = 'movie_data.csv'
df = pd.read_csv(input_file, encoding='utf-8')

from sklearn.feature_extraction.text import CountVectorizer

# Uses sklearn's built in stop_word library
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)

# Fits LDA to the bag-of-words matrix
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)


# Shape should store a matrix containing the word importance of 10 topics in increasing order
print()
print(lda.components_.shape)
print()
print()

# Prints 5 most important words for each of the 10 topics
n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))


# Confirm assumed categories from keywords
print()
print()
horror = X_topics[:, 5].argsort()[::1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')


