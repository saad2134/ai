from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

text = "I love natural language processing"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

print("RoBERTa Embeddings:")
print("Input text:", text)
print("Last hidden state shape:", outputs.last_hidden_state.shape)