# TrueBot - Fake News Detection System

TrueBot is a machine learning-based fake news detection system that helps users identify potentially misleading news articles. It uses a Logistic Regression model trained on text patterns commonly found in fake news, such as sensationalist language and clickbait. The system analyzes news text using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features, which are then used to classify the news as potentially real or fake.

## 🌟 Features

- Text-based fake news detection using TF-IDF and logistic regression
- Simple and intuitive web interface built with Streamlit
- Real-time text analysis and classification
- Clickbait pattern detection
- Educational resources about fake news impact
- Transparent prediction process

## 🔧 Project Structure

```
TrueBot/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── train_model.py         # Model training script
├── streamlit_app.py       # Streamlit web application
├── news_dataset.csv       # Training dataset
├── .gitignore             # Git ignore file
└── models/                # Directory for saved models
    ├── fake_news_model.pkl    # Trained model
    └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
```

## 🚀 Local Installation

1. Clone the repository:
```bash
git clone https://github.com/Navneet277/TrueBot.git
cd TrueBot
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Dependencies

- Python 3.8+
- numpy>=1.21.0,<2.0.0
- pandas>=1.3.0,<3.0.0
- scikit-learn>=1.0.0,<2.0.0
- streamlit>=1.25.0
- joblib>=1.1.0

## 🎯 Local Usage

1. Train the model (if not already trained):
```bash
python train_model.py
```

2. Run the Streamlit app locally:
```bash
streamlit run streamlit_app.py
```

3. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

## ☁️ Streamlit Cloud Deployment

1. **GitHub Setup**:
   - Your code is already on GitHub at [https://github.com/Navneet277/TrueBot.git](https://github.com/Navneet277/TrueBot.git)
   - All necessary files are included

2. **Streamlit Cloud Setup**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click on "New app"
   - Select your repository: `Navneet277/TrueBot`
   - Set main file to: `streamlit_app.py`
   - Click "Deploy"

3. **Important Notes**:
   - The requirements.txt uses flexible version constraints for better compatibility
   - Model files are included in the repository
   - App will automatically install dependencies

## 🎯 How It Works

1. **Text Analysis**:
   - Converts news text to lowercase for consistency
   - Removes punctuation and extra whitespace
   - Identifies common clickbait patterns and sensationalist language
   - Uses basic text preprocessing techniques

2. **Classification Process**:
   - Converts text into numerical features using TF-IDF vectorization
   - Analyzes word patterns and their frequency
   - Uses a trained logistic regression model to classify the text
   - Provides binary classification: Real (1) or Fake (0)

3. **Current Model Capabilities**:
   - Detects obvious clickbait headlines
   - Identifies sensationalist language
   - Recognizes common patterns in fake news
   - Works best with health, science, and general news claims

## ⚠️ Limitations

- Model is trained on a limited dataset
- Best suited for obvious cases of misinformation
- Should be used alongside other fact-checking methods
- Not a replacement for critical thinking and verification

## 🔍 Future Improvements

- Expand training dataset
- Implement advanced NLP techniques
- Add support for multiple languages
- Integrate fact-checking APIs
- Enhance user interface
- Add real-time news source verification

## 👥 Authors

- Navneet Sharma
- Neha Tamboli
- Riddhima Taose
- Nishkarsh Sharma

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the Streamlit team for their excellent framework
- Gratitude to the scikit-learn community for their machine learning tools
