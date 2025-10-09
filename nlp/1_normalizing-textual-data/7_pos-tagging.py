import nltk
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog happily."
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

print("Parts-of-Speech Tagging:")
print("-" * 40)
for word, pos in pos_tags:
    print(f"{word:12} → {pos}")