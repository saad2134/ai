# A stop word is a common word (e.g., "the," "a," "is") often removed in NLP and search tasks because it adds little meaning and appears too frequently.

#Imports
import nltk #Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

text = "This is an example showing stopword removal."
words = word_tokenize(text)

# Pre-compile stopwords for better performance
stop_words = set(stopwords.words('english'))
filtered = [w for w in words if w.lower() not in stop_words]

print("Original:", words)
print("Filtered:", filtered)