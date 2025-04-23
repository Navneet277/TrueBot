import sys
import pkg_resources
import importlib

def check_python_version():
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"

def check_required_packages():
    required = {
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'scikit-learn': '1.2.0',
        'streamlit': '1.30.0',
        'joblib': '1.3.0',
        'nltk': '3.8.1',
        'regex': '2023.0.0'
    }
    
    for package, min_version in required.items():
        try:
            importlib.import_module(package)
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version} (required: >={min_version})")
        except ImportError:
            print(f"ERROR: {package} is not installed!")
            return False
    return True

def check_nltk_data():
    import nltk
    required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if data == 'punkt' 
                          else f'corpora/{data}' if data in ['stopwords', 'wordnet']
                          else f'taggers/{data}')
            print(f"NLTK {data}: ✓")
        except LookupError:
            print(f"ERROR: NLTK {data} is not downloaded!")
            return False
    return True

def check_files():
    import os
    required_files = [
        'train_model.py',
        'streamlit_app.py',
        'news_dataset.csv',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"{file}: ✓")
        else:
            print(f"ERROR: {file} is missing!")
            return False
    return True

if __name__ == "__main__":
    print("🔍 Testing TrueBot Setup...")
    print("\n1. Checking Python version:")
    check_python_version()
    
    print("\n2. Checking required packages:")
    check_required_packages()
    
    print("\n3. Checking NLTK data:")
    check_nltk_data()
    
    print("\n4. Checking required files:")
    check_files()
    
    print("\n✨ Setup verification complete!") 