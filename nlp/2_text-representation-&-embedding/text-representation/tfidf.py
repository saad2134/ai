from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("TF-IDF Representation:")
print("Features:", vectorizer.get_feature_names_out())
print("\nTF-IDF matrix:")
print(X.toarray())