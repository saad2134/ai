# POS (Part-of-Speech) tagging is the process in Natural Language Processing (NLP) of assigning a grammatical category, such as noun, verb, or adjective, to each word in a text.

# Common POS tags explained:
# NNP: Proper noun, singular
# VBZ: Verb, 3rd person singular present
# JJ: Adjective
# IN: Preposition
# NN: Noun, singular or mass
# .: Punctuation mark

#Imports
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

text = "NLTK is great for text processing."
words = word_tokenize(text)
pos_tags = nltk.pos_tag(words)

print("Text:", text)
print("Tokens:", words)
print("POS Tags:", pos_tags)

# More readable output
print("\nDetailed breakdown:")
for word, tag in pos_tags:
    print(f"{word:12} -> {tag}")