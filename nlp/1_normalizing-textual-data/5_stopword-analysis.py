from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is a sample sentence demonstrating the removal of stop words from text."
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))

filtered_words = [word for word in tokens if word not in stop_words and word.isalpha()]

print("Original text:")
print(text)
print(f"\nAll tokens: {tokens}")
print(f"\nStop words removed: {[w for w in tokens if w in stop_words]}")
print(f"\nFiltered words: {filtered_words}")