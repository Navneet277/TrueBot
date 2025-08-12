import streamlit as st
import joblib
from train_model import preprocess_text
import os

# Page configuration
st.set_page_config(
    page_title="TrueBot - Fake News Detector",
    page_icon="🤖",
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

@st.cache_resource
def load_model(model_dir='models'):
    """Load the trained model and vectorizer from the specified directory."""
    try:
        model_path = os.path.join(model_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        # Try to load the model and vectorizer
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            return model, vectorizer
        else:
            st.error(f"Model files not found in {model_dir}. Please ensure the model is trained first.")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Main content
try:
    # Try to load from models directory
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("Could not load the model. Please make sure the model is trained and model files are present in the 'models' directory.")
    else:
        # Text input
        news_text = st.text_area("Enter the news text here:", height=150)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Check News", use_container_width=True):
                if news_text.strip() == "":
                    st.warning("Please enter some text to analyze.")
                else:
                    with st.spinner("Analyzing..."):
                        # Preprocess the text
                        processed_text = preprocess_text(news_text)
                        
                        # Convert text to numbers
                        text_vector = vectorizer.transform([processed_text])
                        
                        # Make prediction
                        prediction = model.predict(text_vector)[0]
                        
                        # Show result
                        if prediction == 0:
                            st.error("⚠️ This news appears to be FAKE")
                        else:
                            st.success("✅ This news appears to be REAL")

        # Collapsible section about fake news
        with st.expander("💡 Impact of Fake News on Daily Life"):
            st.markdown("""
            ### Harmful Effects of Fake News

            1. **Social Impact**
               - Creates division in society
               - Damages personal relationships
               - Leads to mistrust among communities

            2. **Mental Health**
               - Causes anxiety and stress
               - Creates fear and panic
               - Leads to confusion and uncertainty

            3. **Democratic Process**
               - Influences voting decisions
               - Undermines electoral integrity
               - Reduces trust in institutions

            4. **Economic Impact**
               - Affects stock markets
               - Damages business reputations
               - Leads to financial losses

            5. **Public Health**
               - Spreads medical misinformation
               - Affects health-related decisions
               - Undermines public health initiatives

            > "In today's digital age, fake news can reach millions within seconds. Being able to identify it is crucial for maintaining a well-informed society."
            """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown(
    """
    <div class="footer">
        Created by Navneet Sharma, Neha Tamboli and Riddhima Taose
    </div>
    """,
    unsafe_allow_html=True) 