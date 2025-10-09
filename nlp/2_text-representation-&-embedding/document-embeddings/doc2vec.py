from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

documents = [
    TaggedDocument(words=["machine", "learning", "is", "fun"], tags=["doc1"]),
    TaggedDocument(words=["deep", "learning", "neural", "networks"], tags=["doc2"]),
    TaggedDocument(words=["natural", "language", "processing"], tags=["doc3"])
]

model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, epochs=10)

print("Doc2Vec Document Embeddings:")
print("Document vectors:")
for doc_id in ["doc1", "doc2", "doc3"]:
    print(f"{doc_id}: {model.dv[doc_id]}")