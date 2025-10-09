from sklearn.preprocessing import LabelEncoder, OneHotEncoder

words = ["cat", "dog", "cat", "bird", "dog", "fish"]
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(words)

onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

print("One-Hot Encoding:")
for word, onehot in zip(words, onehot_encoded):
    print(f"{word:5} → {onehot}")