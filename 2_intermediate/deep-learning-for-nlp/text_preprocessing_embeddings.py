import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
except:
    print("NLTK downloads may require internet connection")

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_pipeline(self, texts, method='lemmatize', remove_stopwords=True):
        """Complete text preprocessing pipeline"""
        processed_texts = []
        
        for text in texts:
            # Clean text
            cleaned = self.clean_text(str(text))
            
            # Tokenize
            tokens = self.tokenize_text(cleaned)
            
            # Remove stopwords
            if remove_stopwords:
                tokens = self.remove_stopwords(tokens)
            
            # Apply stemming or lemmatization
            if method == 'stem':
                tokens = self.stem_tokens(tokens)
            elif method == 'lemmatize':
                tokens = self.lemmatize_tokens(tokens)
            
            processed_texts.append(tokens)
        
        return processed_texts
    
    def build_vocabulary(self, processed_texts, min_freq=2):
        """Build vocabulary from processed texts"""
        # Count word frequencies
        word_freq = Counter()
        for tokens in processed_texts:
            word_freq.update(tokens)
        
        # Filter by minimum frequency
        self.vocab = {word for word, count in word_freq.items() if count >= min_freq}
        self.vocab = sorted(list(self.vocab))
        
        # Create word to index mapping
        self.word2idx = {word: idx + 1 for idx, word in enumerate(self.vocab)}  # 0 for padding
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = len(self.vocab) + 1
        
        # Create index to word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total vocabulary with special tokens: {len(self.word2idx)}")
        
        return self.vocab, self.word2idx, self.idx2word
    
    def texts_to_sequences(self, processed_texts, max_length=100):
        """Convert processed texts to sequences of indices"""
        sequences = []
        
        for tokens in processed_texts:
            sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
            sequences.append(sequence)
        
        # Pad sequences
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', 
                                       truncating='post', value=self.word2idx['<PAD>'])
        
        return padded_sequences

class EmbeddingAnalyzer:
    """Analyze and create word embeddings"""
    
    def __init__(self):
        self.w2v_model = None
        self.glove_embeddings = None
    
    def train_word2vec(self, processed_texts, vector_size=100, window=5, min_count=2):
        """Train Word2Vec model on processed texts"""
        print("Training Word2Vec model...")
        
        self.w2v_model = Word2Vec(
            sentences=processed_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1  # 1 for skip-gram, 0 for CBOW
        )
        
        print(f"Word2Vec vocabulary size: {len(self.w2v_model.wv.key_to_index)}")
        return self.w2v_model
    
    def load_glove_embeddings(self, file_path, vocab):
        """Load pre-trained GloVe embeddings"""
        print("Loading GloVe embeddings...")
        
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        # Create embedding matrix for our vocabulary
        embedding_dim = len(next(iter(embeddings_index.values())))
        embedding_matrix = np.zeros((len(vocab) + 2, embedding_dim))  # +2 for PAD and UNK
        
        for word, idx in vocab.items():
            if word in embeddings_index:
                embedding_matrix[idx] = embeddings_index[word]
            else:
                # Initialize with random values for unknown words
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
        self.glove_embeddings = embedding_matrix
        print(f"Loaded GloVe embeddings. Dimension: {embedding_dim}")
        print(f"Coverage: {np.sum(np.any(embedding_matrix[1:], axis=1)) / len(vocab):.2%}")
        
        return embedding_matrix
    
    def visualize_embeddings(self, words, model_type='word2vec'):
        """Visualize word embeddings using PCA"""
        from sklearn.decomposition import PCA
        
        if model_type == 'word2vec' and self.w2v_model:
            model = self.w2v_model
        else:
            print("Model not available")
            return
        
        # Get vectors for specified words
        vectors = []
        valid_words = []
        for word in words:
            if word in model.wv.key_to_index:
                vectors.append(model.wv[word])
                valid_words.append(word)
        
        if not vectors:
            print("No valid words found in vocabulary")
            return
        
        vectors = np.array(vectors)
        
        # Apply PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
        
        for i, word in enumerate(valid_words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        fontsize=12, alpha=0.8)
        
        plt.title(f'Word Embeddings Visualization (PCA) - {model_type.upper()}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return vectors_2d
    
    def find_similar_words(self, word, topn=10):
        """Find similar words using embeddings"""
        if self.w2v_model and word in self.w2v_model.wv.key_to_index:
            similar = self.w2v_model.wv.most_similar(word, topn=topn)
            print(f"Words similar to '{word}':")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")
            return similar
        else:
            print(f"Word '{word}' not in vocabulary")
            return None

def demonstrate_text_preprocessing():
    """Demonstrate text preprocessing techniques"""
    # Sample texts for demonstration
    sample_texts = [
        "This is a GREAT movie! I loved it so much. 😊",
        "The acting was terrible and the plot was boring...",
        "I've never seen such an amazing film in my life!!!",
        "It was okay, nothing special. 2.5/5 stars.",
        "<html>This movie is <b>fantastic</b>! Visit http://example.com for more.</html>"
    ]
    
    preprocessor = TextPreprocessor()
    
    print("Original Texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")
    
    print("\n" + "="*50)
    print("Cleaned Texts:")
    cleaned_texts = [preprocessor.clean_text(text) for text in sample_texts]
    for i, text in enumerate(cleaned_texts):
        print(f"{i+1}. {text}")
    
    print("\n" + "="*50)
    print("Tokenized Texts:")
    tokenized_texts = preprocessor.preprocess_pipeline(sample_texts, method='lemmatize')
    for i, tokens in enumerate(tokenized_texts):
        print(f"{i+1}. {tokens}")
    
    return preprocessor, tokenized_texts

def demonstrate_embeddings(processed_texts):
    """Demonstrate word embedding techniques"""
    analyzer = EmbeddingAnalyzer()
    
    # Train Word2Vec
    print("="*60)
    w2v_model = analyzer.train_word2vec(processed_texts, vector_size=50)
    
    # Demonstrate similar words
    print("\n" + "="*50)
    test_words = ['great', 'amazing', 'terrible', 'boring', 'love']
    for word in test_words:
        analyzer.find_similar_words(word)
    
    # Visualize embeddings
    print("\n" + "="*50)
    vectors_2d = analyzer.visualize_embeddings(test_words)
    
    return analyzer, w2v_model

def compare_vectorization_methods(texts, labels):
    """Compare different text vectorization methods"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    vectorizers = {
        'Count Vectorizer': CountVectorizer(max_features=1000),
        'TF-IDF Vectorizer': TfidfVectorizer(max_features=1000)
    }
    
    results = {}
    
    for name, vectorizer in vectorizers.items():
        print(f"\n{name}:")
        
        # Fit and transform
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train classifier
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'vectorizer': vectorizer,
            'classifier': clf,
            'accuracy': accuracy
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return results

def create_sample_dataset():
    """Create a sample sentiment analysis dataset"""
    # Positive samples
    positive_texts = [
        "This movie is absolutely fantastic and wonderful",
        "I loved every moment of this amazing film",
        "Great acting and excellent storyline",
        "One of the best movies I have ever seen",
        "Incredible performance by all actors",
        "The cinematography was breathtaking and beautiful",
        "This film touched my heart deeply",
        "Outstanding direction and brilliant screenplay",
        "I was completely captivated from start to finish",
        "A masterpiece that deserves all the awards"
    ]
    
    # Negative samples
    negative_texts = [
        "This movie was terrible and boring",
        "I hated every minute of this awful film",
        "Poor acting and terrible storyline",
        "One of the worst movies I have ever seen",
        "Horrible performance by all actors",
        "The cinematography was dreadful and ugly",
        "This film broke my heart in a bad way",
        "Terrible direction and awful screenplay",
        "I was completely bored from start to finish",
        "A disaster that deserves no awards"
    ]
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, labels

def main():
    """Main function to demonstrate NLP preprocessing and embeddings"""
    print("=== Text Preprocessing and Word Embeddings ===")
    
    # Create sample dataset
    print("\n1. Creating Sample Dataset...")
    texts, labels = create_sample_dataset()
    
    # Demonstrate preprocessing
    print("\n2. Text Preprocessing Demonstration...")
    preprocessor, processed_texts = demonstrate_text_preprocessing()
    
    # Build vocabulary
    print("\n3. Building Vocabulary...")
    vocab, word2idx, idx2word = preprocessor.build_vocabulary(processed_texts)
    
    # Convert to sequences
    print("\n4. Converting to Sequences...")
    sequences = preprocessor.texts_to_sequences(processed_texts, max_length=20)
    print(f"Sequences shape: {sequences.shape}")
    print(f"Sample sequence: {sequences[0]}")
    
    # Demonstrate embeddings
    print("\n5. Word Embeddings Demonstration...")
    analyzer, w2v_model = demonstrate_embeddings(processed_texts)
    
    # Compare vectorization methods
    print("\n6. Vectorization Methods Comparison...")
    results = compare_vectorization_methods(texts, labels)
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Word2Vec model trained: {w2v_model is not None}")
    if w2v_model:
        print(f"Word2Vec vocabulary: {len(w2v_model.wv.key_to_index)}")
    
    for method, result in results.items():
        print(f"{method} Accuracy: {result['accuracy']:.4f}")
    
    return preprocessor, analyzer, results

if __name__ == "__main__":
    preprocessor, analyzer, results = main()