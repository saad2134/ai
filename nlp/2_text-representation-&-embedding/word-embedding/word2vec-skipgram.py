from gensim.models import Word2Vec

sentences = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "neural", "networks"],
    ["natural", "language", "processing"]
]

model = Word2Vec(sentences, vector_size=10, window=2, sg=1, min_count=1)  # sg=1 for SkipGram

print("Word2Vec SkipGram:")
print("Vocabulary size:", len(model.wv.key_to_index))
print("Vector for 'learning':", model.wv['learning'])
print("\nSimilar to 'learning':", model.wv.most_similar('learning'))