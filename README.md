# Income Tax Fraud Detection System

This is a Streamlit application for detecting potential income tax fraud using machine learning.

## Deployment Instructions

### Local Deployment
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

### Deploying to Streamlit Cloud
1. Create a GitHub repository and push your code
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Click "Deploy"

## Required Files
- app.py (main application file)
- best_model.joblib (trained model)
- label_encoder_*.joblib (label encoders)
- requirements.txt (dependencies)
- images/ (directory containing images)

## Note
Make sure all the required files (model files and images) are included in your repository before deploying. 