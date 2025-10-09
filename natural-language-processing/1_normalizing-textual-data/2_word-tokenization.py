import nltk
from nltk.tokenize import word_tokenize

text = "I don't like pizza! It costs $19.99, but it's delicious."
tokens = word_tokenize(text)

print("Original text:")
print(text)
print("\nTokenized words:")
print(tokens)
print(f"\nTotal tokens: {len(tokens)}")