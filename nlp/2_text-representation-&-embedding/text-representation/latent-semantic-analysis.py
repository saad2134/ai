from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning deep learning",
    "artificial intelligence machine learning", 
    "deep neural networks",
    "natural language processing"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

lsa = TruncatedSVD(n_components=2)
X_lsa = lsa.fit_transform(X)

print("Latent Semantic Analysis:")
print("Original shape:", X.shape)
print("LSA shape:", X_lsa.shape)
print("Reduced features:\n", X_lsa)