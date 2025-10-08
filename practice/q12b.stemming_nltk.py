# Stemming in NLP reduces words to their base form by removing prefixes and suffixes, often creating non-real words. It groups variations like “running” and “runner” into “run,” improving efficiency and accuracy in tasks like NLP, text mining and information retrieval.

#Imports
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)

ps = PorterStemmer()
text = "Stemming words reduces them to roots running runner runs."
words = word_tokenize(text)

# Stem each word
stemmed_words = [ps.stem(w) for w in words]

print("Original:", words)
print("Stemmed: ", stemmed_words)