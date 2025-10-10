from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def create_sample_data(self):
        """Create sample movie review dataset"""
        data = {
            'text': [
                "This movie is absolutely fantastic and wonderful",
                "I loved every moment of this amazing film",
                "Great acting and excellent storyline",
                "One of the best movies I have ever seen",
                "This movie was terrible and boring",
                "I hated every minute of this awful film",
                "Poor acting and terrible storyline",
                "One of the worst movies I have ever seen",
                "The plot was predictable and characters were flat",
                "Brilliant cinematography and outstanding performances",
                "Waste of time and money, completely disappointing",
                "Masterpiece that deserves all the awards",
                "Boring dialogue and weak character development",
                "Heartwarming story with great emotional depth",
                "Terrible editing and confusing plot",
                "Funny, engaging, and thoroughly entertaining",
                "Slow pacing made it difficult to stay interested",
                "Visual effects were stunning and impressive"
            ],
            'label': [
                'positive', 'positive', 'positive', 'positive',
                'negative', 'negative', 'negative', 'negative',
                'negative', 'positive', 'negative', 'positive',
                'negative', 'positive', 'negative', 'positive',
                'negative', 'positive'
            ]
        }
        return pd.DataFrame(data)
    
    def train(self, texts, labels):
        """Train the text classifier"""
        # Convert text to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        print("✅ Text classifier trained successfully!")
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
    
    def predict(self, text):
        """Predict sentiment of new text"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        # Transform text to features
        X = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X).max()
        
        return prediction, probability
    
    def evaluate(self, texts, labels):
        """Evaluate classifier performance"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        
        accuracy = accuracy_score(labels, predictions)
        print(f"📊 Model Accuracy: {accuracy:.2%}")
        print("\n📋 Classification Report:")
        print(classification_report(labels, predictions))

def demo_text_classification():
    classifier = TextClassifier()
    
    # Create and prepare data
    df = classifier.create_sample_data()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    print("=== Text Classification Demo ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train classifier
    classifier.train(X_train, y_train)
    
    # Evaluate performance
    print("\n=== Model Evaluation ===")
    classifier.evaluate(X_test, y_test)
    
    # Interactive predictions
    print("\n=== Interactive Predictions ===")
    test_reviews = [
        "This movie is amazing and I love it!",
        "Terrible film, waste of time",
        "The acting was okay but the plot was weak",
        "Brilliant performance by all actors"
    ]
    
    for review in test_reviews:
        prediction, confidence = classifier.predict(review)
        print(f"\"{review}\"")
        print(f"→ Sentiment: {prediction} (confidence: {confidence:.2%})")
        print()

if __name__ == "__main__":
    demo_text_classification()