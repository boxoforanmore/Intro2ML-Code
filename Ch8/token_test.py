from copy import deepcopy

# Very standard tokenizing
def tokenizer(text):
    return text.split()

test_str = 'runners like running and thus they run a lot oh yes they do'
print("Standard Tokenizer:")
print(tokenizer(deepcopy(test_str)))
print()


# PorterStemmer stemming algorithm
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print("Porter Stemmer:")
print(tokenizer_porter(test_str))
print()


# Remove stop-words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')

print("PS with stop-words removed:")
print([w for w in tokenizer_porter(test_str)[-10:] if w not in stop])
print()


