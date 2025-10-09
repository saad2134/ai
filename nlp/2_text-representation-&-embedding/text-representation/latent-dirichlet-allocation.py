from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning artificial intelligence",
    "deep learning neural networks",
    "natural language processing text mining",
    "computer vision image processing"
]

vectorizer = CountVectorizer(max_features=10)
X = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

print("Latent Dirichlet Allocation:")
print("Topics found:")
for i, topic in enumerate(lda.components_):
    print(f"Topic {i+1}: {[vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-3:]]}")