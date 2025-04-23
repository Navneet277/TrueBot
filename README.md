# TrueBot - Advanced Fake News Detection System

TrueBot is an AI-powered system that uses advanced Natural Language Processing (NLP) techniques to detect fake news and misleading content with high accuracy (>85%).

## 🌟 Features

- Advanced text preprocessing using NLTK
- Machine learning-based classification using Random Forest
- Real-time analysis of news content
- Detailed feature analysis and confidence scores
- User-friendly web interface built with Streamlit
- Cross-validation and model performance metrics
- Cloud deployment ready

## 🚀 Quick Start Guide

### Local Setup (5 minutes)

1. **Get the Code**:
```bash
git clone https://github.com/Navneet277/TrueBot.git
cd TrueBot
```

2. **Set Up Environment**:
```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Install setuptools first
pip install --upgrade pip setuptools wheel
```

3. **Install Requirements**:
```bash
pip install -r requirements.txt
```

4. **Download Required Data**:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

5. **Train & Run**:
```bash
# Train the model
python train_model.py

# Start the app
streamlit run streamlit_app.py
```

### Cloud Deployment (2 minutes)

1. **Prepare Repository**:
   - Fork this repository to your GitHub account
   - Ensure all files are present:
     - train_model.py
     - streamlit_app.py
     - requirements.txt
     - news_dataset.csv
     - .streamlit/config.toml

2. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select TrueBot repository
   - Click "Deploy"

That's it! Your app is now live 🎉

### Verify Setup

Run the test script to check if everything is properly installed:
```bash
python test_setup.py
```

This will verify:
- Python version
- Required packages
- NLTK data
- Required files
- Model training capability

### Troubleshooting

1. **Package Installation Issues**:
```bash
# Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# Then install requirements
pip install --no-cache-dir -r requirements.txt
```

2. **NLTK Data Issues**:
```bash
# Download all NLTK data
python -m nltk.downloader all
```

3. **Common Problems**:
   - **Missing Files**: Check if all required files are in your project folder
   - **Memory Error**: Close other applications or increase available memory
   - **Import Errors**: Make sure you're in your virtual environment
   - **Streamlit Issues**: Check if port 8501 is free

Need help? Open an issue on GitHub!

## 🔧 Project Structure

```
TrueBot/
├── .streamlit/              # Streamlit configuration
│   └── config.toml         # Theme and server settings
├── models/                 # Directory for saved models (auto-created)
│   ├── fake_news_model.pkl    # Trained classifier
│   └── tfidf_vectorizer.pkl   # Text vectorizer
├── train_model.py         # Model training script
├── streamlit_app.py       # Streamlit web application
├── news_dataset.csv       # Training dataset
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 📋 Requirements

- Python 3.8+
- Dependencies (specified in requirements.txt):
  ```
  numpy>=1.24.0
  pandas>=2.0.0
  scikit-learn>=1.2.0
  streamlit>=1.30.0
  joblib>=1.3.0
  nltk>=3.8.1
  regex>=2023.0.0
  python-dotenv==1.0.0
  ```

## 🚀 Installation & Deployment

### Local Development Setup

1. **System Requirements**:
   - Python 3.8 or higher
   - Git
   - 4GB RAM minimum (8GB recommended)
   - 500MB free disk space
   - Internet connection (for initial setup)

2. **Clone the Repository**:
```bash
git clone https://github.com/Navneet277/TrueBot.git
cd TrueBot
```

3. **Set Up Python Environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.8 or higher
```

4. **Install Dependencies**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install development tools
pip install black flake8 pytest

# Verify installations
python -c "import numpy; import pandas; import sklearn; import nltk; import streamlit; print('All packages installed successfully!')"
```

5. **Download NLTK Data**:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

6. **Prepare the Dataset**:
   - Ensure `news_dataset.csv` is in the project root
   - Verify file permissions
   - Check file encoding (should be UTF-8)

7. **Configure Development Environment**:
```bash
# Create .streamlit directory
mkdir .streamlit

# Create config.toml with required settings
echo "[theme]
primaryColor='#1E88E5'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#F0F2F6'
textColor='#262730'
font='sans serif'

[server]
enableCORS=false
enableXsrfProtection=true" > .streamlit/config.toml

# Format code
black .

# Run linting
flake8 .

# Run tests
pytest tests/
```

8. **Train the Model**:
```bash
# Train the model
python train_model.py

# Verify model files were created
dir models\  # On Windows
ls models/   # On Unix/MacOS
```

9. **Start the Application**:
```bash
streamlit run streamlit_app.py
```

10. **Verify Installation**:
    - Check model files exist in `models/` directory
    - Verify NLTK data is downloaded
    - Test with sample news articles

### Troubleshooting Local Setup

1. **Package Installation Issues**:
```bash
# If pip install fails, try:
pip install --no-cache-dir -r requirements.txt

# For NLTK errors:
python -m nltk.downloader all
```

2. **Model Training Issues**:
```bash
# Clear existing models
rm -rf models/*  # Unix/MacOS
del /F /Q models\*  # Windows

# Retrain with verbose output
python -m train_model --verbose
```

3. **Common Problems and Solutions**:
   - **Missing NLTK Data**: Run NLTK downloads individually
   - **Memory Issues**: Close other applications
   - **Permission Errors**: Run terminal as administrator
   - **Path Issues**: Use absolute paths in configurations

4. **Performance Optimization**:
   - Adjust model parameters in `train_model.py`
   - Configure Streamlit memory settings
   - Use production WSGI server for deployment

### Development Tools (Optional)

1. **Code Formatting**:
```bash
pip install black
black .
```

2. **Linting**:
```bash
pip install flake8
flake8 .
```

3. **Testing**:
```bash
pip install pytest
pytest tests/
```

### ☁️ Streamlit Cloud Deployment

Direct Link: https://truebot.streamlit.app/

1. **Prepare Your Repository**:
   - Create a GitHub repository for your project
   - Ensure your repository has:
     - All project files
     - requirements.txt
     - .streamlit/config.toml (optional, for custom themes)

2. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Select your TrueBot repository
   - Choose main branch and streamlit_app.py
   - Click "Deploy!"

3. **Environment Setup**:
   - Streamlit Cloud will automatically:
     - Install requirements from requirements.txt
     - Set up Python environment
     - Configure NLTK data
     - Handle model training on first run

4. **Post-Deployment**:
   - Monitor app performance
   - Check resource usage
   - Set up usage alerts
   - Configure automatic redeployment

## 💻 Usage

1. Train the model:
```bash
python train_model.py
```

2. Start the application:
```bash
streamlit run streamlit_app.py
```

## 🎯 Model Details

### Text Preprocessing
- Tokenization and lemmatization using NLTK
- Stopword removal
- URL and email address removal
- Special character handling
- Advanced clickbait pattern detection

### Feature Engineering
- TF-IDF vectorization with n-grams
- Text length analysis
- Word count statistics
- Punctuation pattern analysis
- Uppercase ratio calculation
- Custom feature extraction

### Model Parameters (Random Forest)
- 200 decision trees
- Maximum depth of 20
- Balanced class weights
- Parallel processing support
- Cross-validation evaluation

## 📊 Performance

- Accuracy: >85%
- Balanced precision and recall
- Cross-validated results
- Detailed classification reports
- Confusion matrix analysis

## 💡 Example Usage

```python
from train_model import preprocess_text
import joblib

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Analyze text
text = "Your news article here..."
processed_text = preprocess_text(text)
vector = vectorizer.transform([processed_text])
prediction = model.predict_proba(vector)
```

## ⚠️ Limitations and Best Practices

1. **Model Limitations**:
   - Best suited for obvious cases of misinformation
   - Should be used alongside other fact-checking methods
   - Not a replacement for critical thinking
   - Limited to text-based analysis

2. **Usage Tips**:
   - Provide sufficient context in input text
   - Check confidence scores carefully
   - Cross-reference with reliable sources
   - Consider the source and context of news
   - Look for multiple perspectives

## 👥 Authors

- Navneet Sharma
- Neha Tamboli
- Riddhima Taose

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Thanks to all contributors
- Special thanks to the Streamlit team
- Gratitude to the scikit-learn and NLTK communities

For issues, suggestions, or contributions, please use the GitHub issue tracker. 
