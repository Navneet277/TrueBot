import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import re
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_text(text):
    """Enhanced text preprocessing with advanced NLP techniques"""
    try:
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers and special characters
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove common clickbait and sensational words
        clickbait_patterns = [
            r'breaking[!]*',
            r'urgent[!]*',
            r'shocking[!]*',
            r'must share[!]*',
            r'warning[!]*',
            r'exposed[!]*',
            r'secret[!]*',
            r'miracle[!]*',
            r'incredible[!]*',
            r'leaked[!]*',
            r'you wont believe[!]*',
            r'mind blowing[!]*',
            r'unbelievable[!]*',
            r'amazing[!]*',
            r'revolutionary[!]*'
        ]
        
        text = ' '.join(tokens)
        for pattern in clickbait_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return text

def extract_text_features(text):
    """Extract additional features from text"""
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }
    return features

def evaluate_model(model, X, y, vectorizer):
    """Evaluate model performance using cross-validation"""
    cv_scores = cross_val_score(model, vectorizer.transform(X), y, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    return cv_scores.mean()

def train_model(save_dir='models'):
    """Train an improved fake news detection model."""
    try:
        # Get project root and create absolute paths
        project_root = get_project_root()
        data_path = os.path.join(project_root, 'news_dataset.csv')
        save_dir = os.path.join(project_root, save_dir)
        
        print(f"Looking for dataset at: {data_path}")
        print(f"Models will be saved to: {save_dir}")
        
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and validate dataset
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"Dataset loaded successfully with {len(df)} rows")
        
        # Check class balance
        class_distribution = df['label'].value_counts()
        print("\nClass distribution:")
        print(class_distribution)
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")
        
        # Advanced text preprocessing
        print("\nPreprocessing text data...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Extract additional features
        print("\nExtracting additional features...")
        additional_features = df['processed_text'].apply(extract_text_features).apply(pd.Series)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create TF-IDF vectorizer with optimized parameters
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear scaling to term frequencies
        )
        
        # Create and train the model pipeline
        print("\nTraining the model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Fit TF-IDF vectorizer
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f'\nModel Accuracy: {accuracy*100:.2f}%')
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Evaluate model with cross-validation
        cv_score = evaluate_model(model, X_train, y_train, tfidf_vectorizer)
        
        # Save the model and vectorizer
        model_path = os.path.join(save_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
        
        print(f"\nSaving model to: {model_path}")
        print(f"Saving vectorizer to: {vectorizer_path}")
        
        joblib.dump(model, model_path)
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        print(f"\nModel and vectorizer saved successfully in {save_dir}!")
        
        # Test some example predictions
        print("\nTesting example predictions:")
        test_texts = [
            "Local weather report predicts rain tomorrow",
            "SHOCKING: Miracle cure found in ordinary tap water!!!",
            "New study shows regular exercise improves heart health",
            "Scientists discover evidence of ancient civilization on Mars!"
        ]
        
        test_vectors = tfidf_vectorizer.transform([preprocess_text(text) for text in test_texts])
        predictions = model.predict_proba(test_vectors)
        
        for text, pred in zip(test_texts, predictions):
            print(f"\nText: {text}")
            print(f"Probability - Real: {pred[1]:.2f}, Fake: {pred[0]:.2f}")
        
        return accuracy, model_path, vectorizer_path
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("Training the fake news detection model...")
    try:
        accuracy, model_path, vectorizer_path = train_model()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
