import nltk
from nltk.tokenize import sent_tokenize

text = "Hello world! How are you? I'm learning NLP. It's fascinating."
sentences = sent_tokenize(text)

print("Original text:")
print(text)
print("\nSegmented sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")