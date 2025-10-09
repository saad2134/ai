from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is awesome",
    "I love coding in Python"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Bag of Words:")
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Matrix shape:", X.shape)
print("\nDocument-term matrix:")
print(X.toarray())