from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "I love natural language processing"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

print("BERT Embeddings:")
print("Input text:", text)
print("Tokenized:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("Pooler output shape:", outputs.pooler_output.shape)