import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
words = ["running", "runner", "ran", "runs", "happily", "happiness", "unhappy", "better"]
lemmatizer = WordNetLemmatizer()

print("Word Lemmatization Examples:")
print("-" * 35)
for word in words:
    lemma = lemmatizer.lemmatize(word, pos='v')  # 'v' for verb
    print(f"{word:12} → {lemma}")