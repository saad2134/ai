from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "I love natural language processing"
tokens = word_tokenize(text)

print("N-Grams:")
print("Text:", text)
print(f"\nUnigrams: {list(ngrams(tokens, 1))}")
print(f"Bigrams:  {list(ngrams(tokens, 2))}")
print(f"Trigrams: {list(ngrams(tokens, 3))}")