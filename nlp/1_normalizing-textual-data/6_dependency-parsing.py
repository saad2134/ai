import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

print("Dependency Parsing:")
print("-" * 50)
for token in doc:
    print(f"{token.text:8} → {token.dep_:12} → {token.head.text:8} (Head)")