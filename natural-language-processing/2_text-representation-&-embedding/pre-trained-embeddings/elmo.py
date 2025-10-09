# ELMo requires tensorflow_hub and is computationally intensive
# This is a simplified demonstration

print("ELMo (Embeddings from Language Models)")
print("Key features:")
print("- Contextual word embeddings")
print("- Uses bidirectional LSTM")
print("- Captures polysemy (multiple meanings)")
print("- Example: 'bank' in 'river bank' vs 'money bank'")

# In practice:
# import tensorflow_hub as hub
# elmo = hub.load("https://tfhub.dev/google/elmo/3")
# embeddings = elmo.signatures["default"](tf.constant(["I love NLP"]))["elmo"]