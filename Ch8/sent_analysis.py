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
            df = df.append([[txt, labels[l]]], ignore_index=true)
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
print(df.loc[0, 'review'][-50:]


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
print(preprocessor(df.loc[0, 'review'][-50:])
print()
print("Testing regex:")
print(prerpocessor("</a>This :) is :( a test :-)!")
print()


# Apply preprocessor to DataFrame
df['review'] = df['review'].apply(preprocessor)


# Divide the DataFrame into test and training sets
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


