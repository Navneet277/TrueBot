import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re
import string

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common clickbait words and excessive punctuation
    clickbait_patterns = [
        r'breaking[!]*',
        r'urgent[!]*',
        r'shocking[!]*',
        r'must share[!]*',
        r'warning[!]*',
        r'exposed[!]*',
        r'secret[!]*',
        r'miracle[!]*'
    ]
    
    for pattern in clickbait_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def train_model(save_dir='models'):
    # Load the dataset
    df = pd.read_csv('news_dataset.csv')
    
    # Basic preprocessing
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']  # Ensure balanced split
    )
    
    # Convert text to numbers using TF-IDF with enhanced features
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Include bigrams
        stop_words='english',
        min_df=2  # Minimum document frequency
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train a Logistic Regression model with balanced class weights
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy*100:.2f}%')
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model and vectorizer
    model_path = os.path.join(save_dir, 'fake_news_model.pkl')
    vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"\nModel and vectorizer saved successfully in {save_dir}!")
    
    return accuracy, model_path, vectorizer_path

if __name__ == "__main__":
    print("Training the fake news detection model...")
    accuracy, model_path, vectorizer_path = train_model() 