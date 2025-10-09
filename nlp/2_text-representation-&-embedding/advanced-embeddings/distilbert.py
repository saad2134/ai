from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

text = "I love natural language processing"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

print("DistilBERT Embeddings:")
print("Input text:", text)
print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("→ DistilBERT is 40% smaller and 60% faster than BERT")