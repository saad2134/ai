from gensim.models import Word2Vec

sentences = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "neural", "networks"],
    ["natural", "language", "processing"]
]

model = Word2Vec(sentences, vector_size=10, window=2, sg=0, min_count=1)  # sg=0 for CBOW

print("Word2Vec CBOW:")
print("Vocabulary size:", len(model.wv.key_to_index))
print("Vector for 'learning':", model.wv['learning'])