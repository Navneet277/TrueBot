import streamlit as st
import joblib
import os
from train_model import preprocess_text
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="TrueBot - Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title-text {
        text-align: center;
        color: #1E88E5;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #E53935;
        font-size: 24px;
        font-style: italic;
        margin-top: 0;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #1E88E5;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Subtitle
st.markdown('<p class="title-text">TrueBot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Fake news spreads faster than the truth</p>', unsafe_allow_html=True)

# Add a relevant GIF
st.markdown("""
    <div style="display: flex; justify-content: center;">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd6Y2E3NWF1dXF3OWF4OWF4OWF1d3E5YXg5YXg5YXg5YXg5YXg5YXg5/3o7TKwxYkeW0ZvTqsU/giphy.gif" 
        alt="Fake News GIF" width="300px">
    </div>
    """, unsafe_allow_html=True)

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        model_path = os.path.join(model_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.error("Model files not found. Please train the model first.")
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def analyze_text_features(text, vectorizer):
    """Analyze important features in the text."""
    # Get feature names and their TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vector = vectorizer.transform([text])
    
    # Get non-zero features and their scores
    non_zero_elements = tfidf_vector.nonzero()[1]
    scores = tfidf_vector.data
    
    # Create a dataframe of features and their scores
    features_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in non_zero_elements],
        'Score': scores
    })
    features_df = features_df.sort_values('Score', ascending=False)
    
    return features_df.head(10)  # Return top 10 features

def get_confidence_color(confidence):
    """Return color based on confidence level."""
    if confidence >= 0.8:
        return "red" if confidence >= 0.9 else "orange"
    return "green"

def main():
    st.title("🔍 TrueBot - Fake News Detector")
    st.markdown("""
    ### Analyze text to detect potential fake news
    Enter your text below to check if it shows characteristics of fake news.
    For best results, enter complete sentences or paragraphs.
    """)
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Text input
    text_input = st.text_area(
        "Enter the text to analyze:",
        height=150,
        placeholder="Paste your text here..."
    )
    
    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            st.stop()
            
        try:
            # Preprocess the text
            processed_text = preprocess_text(text_input)
            
            # Transform the text
            text_vector = vectorizer.transform([processed_text])
            
            # Get prediction probabilities
            probabilities = model.predict_proba(text_vector)[0]
            fake_prob, real_prob = probabilities
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Analysis Results")
                
                # Display the main prediction
                prediction = "FAKE" if fake_prob > 0.5 else "REAL"
                confidence = max(fake_prob, real_prob)
                
                st.markdown(f"""
                ### Prediction: <span style='color: {get_confidence_color(confidence)}'>{prediction}</span>
                """, unsafe_allow_html=True)
                
                # Display probability bars
                st.markdown("### Confidence Scores")
                st.progress(real_prob)
                st.write(f"Probability of being REAL: {real_prob:.1%}")
                
                st.progress(fake_prob)
                st.write(f"Probability of being FAKE: {fake_prob:.1%}")
            
            with col2:
                st.subheader("Feature Analysis")
                # Analyze and display important text features
                features_df = analyze_text_features(processed_text, vectorizer)
                
                st.markdown("### Top Contributing Words/Phrases")
                st.dataframe(features_df.style.format({'Score': '{:.3f}'}))
                
                # Display interpretation
                st.markdown("### Interpretation Guide")
                if confidence >= 0.9:
                    st.warning("⚠️ High confidence prediction - The model is very certain about this classification.")
                elif confidence >= 0.7:
                    st.info("ℹ️ Moderate confidence prediction - The model shows reasonable certainty.")
                else:
                    st.warning("⚠️ Low confidence prediction - The model is uncertain about this classification.")
            
            # Additional analysis and tips
            st.subheader("Analysis Details")
            with st.expander("See detailed analysis"):
                st.markdown("""
                #### What was analyzed:
                - Text structure and patterns
                - Common fake news indicators
                - Language complexity and tone
                
                #### Tips for interpretation:
                - High confidence scores (>90%) indicate stronger predictions
                - Consider the context and source of the text
                - Look for emotional language and exaggerated claims
                - Check multiple sources for verification
                """)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Please try again with different text or contact support if the problem persists.")

if __name__ == "__main__":
    main()

# Footer
st.markdown(
    """
    <div class="footer">
        Created by Navneet Sharma, Neha Tamboli and Riddhima Taose
    </div>
    """,
    unsafe_allow_html=True)
