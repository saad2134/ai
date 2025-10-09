from gensim.models import FastText

sentences = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "neural", "networks"],
    ["natural", "language", "processing"]
]

model = FastText(sentences, vector_size=10, window=3, min_count=1)

print("FastText Embeddings:")
print("Vocabulary size:", len(model.wv.key_to_index))
print("Vector for 'learning':", model.wv['learning'])

# FastText can handle OOV words using subword information
print("\nVector for OOV word 'learnings':", model.wv['learnings'])