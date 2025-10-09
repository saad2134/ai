import numpy as np

# Simulated GloVe embeddings (in practice, load pre-trained)
vocab = {"king": 0, "queen": 1, "man": 2, "woman": 3}
embeddings = np.array([
    [0.5, 0.8, 0.2],  # king
    [0.5, 0.9, 0.1],  # queen  
    [0.6, 0.1, 0.8],  # man
    [0.6, 0.2, 0.7]   # woman
])

print("GloVe Embeddings (simulated):")
for word, idx in vocab.items():
    print(f"{word}: {embeddings[idx]}")
    
# Vector arithmetic example
king = embeddings[0]
man = embeddings[2] 
woman = embeddings[3]
result = king - man + woman

print(f"\nking - man + woman ≈ queen: {result}")
print(f"Similarity with actual queen: {np.dot(result, embeddings[1]):.3f}")