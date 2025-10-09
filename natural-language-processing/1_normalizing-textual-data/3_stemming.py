from nltk.stem import PorterStemmer

words = ["running", "runner", "ran", "runs", "happily", "happiness", "unhappy"]
stemmer = PorterStemmer()

print("Word Stemming Examples:")
print("-" * 30)
for word in words:
    stemmed = stemmer.stem(word)
    print(f"{word:12} → {stemmed}")