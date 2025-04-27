import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image

# Must be the first Streamlit command
st.set_page_config(
    page_title="Income Tax Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load the label encoders
label_encoder_occupation = joblib.load("label_encoder_occupation.joblib")
label_encoder_marital = joblib.load("label_encoder_marital_status.joblib")
label_encoder_children = joblib.load("label_encoder_children.joblib")

def load_best_model():
    return joblib.load("best_model.joblib")

def predict_income(model, input_data):
    predicted = model.predict(input_data.reshape(1, -1))[0]
    
    # Ensure prediction is within reasonable bounds
    min_income = 200000  # 2L minimum
    max_income = 5000000  # 50L maximum
    
    if predicted < min_income:
        predicted = min_income
    elif predicted > max_income:
        predicted = max_income
        
    return predicted

def classify_fraud(reported_income, predicted_income, age, occupation):
    # Direct high risk check for large discrepancies
    if reported_income > predicted_income * 2:
        return "High Risk of Fraud", "red"
    # 1. Calculate percentage difference between reported and predicted income
    income_difference = abs(reported_income - predicted_income)
    percentage_difference = (income_difference / predicted_income) * 100
    
    # 2. Check tax slab differences
    def get_tax_slab(income):
        if income <= 300000: return "0%"
        elif income <= 600000: return "5%"
        elif income <= 900000: return "10%"
        elif income <= 1200000: return "15%"
        elif income <= 1500000: return "20%"
        else: return "30%"
    
    # 3. Set occupation-specific thresholds (made more stringent)
    base_thresholds = {
        "Salaried": {"variance": 3},  # Reduced from 5
        "Self-employed": {"variance": 7},  # Reduced from 10
        "Business": {"variance": 10}  # Reduced from 15
    }

    # 4. Collect fraud indicators
    fraud_indicators = []
    
    # Check 1: Tax slab mismatch
    if get_tax_slab(reported_income) != get_tax_slab(predicted_income):
        fraud_indicators.append("tax_slab_mismatch")
    
    # Check 2: Income difference exceeds threshold
    if percentage_difference > base_thresholds[occupation]["variance"]:
        fraud_indicators.append("income_variance")
    
    # Check 3: Severe underreporting (made more stringent)
    if reported_income < (predicted_income * 0.95):  # Changed from 0.9 to 0.95
        fraud_indicators.append("severe_underreporting")
    
    # Check 4: Age-based income consistency
    if age < 30 and reported_income > 1000000:  # High income for young age
        fraud_indicators.append("age_income_mismatch")
    
    # Check 5: Occupation-based income consistency
    if occupation == "Salaried" and reported_income > 2000000:  # Very high salary
        fraud_indicators.append("occupation_income_mismatch")
    
    # 5. Final Risk Classification
    if len(fraud_indicators) == 0:
        return "No Fraud Detected", "green"     # No indicators triggered
    elif len(fraud_indicators) == 1:
        return "Low Risk of Fraud", "yellow"    # One indicator triggered
    elif len(fraud_indicators) == 2:
        return "Medium Risk of Fraud", "#ff9800"  # Two indicators triggered
    else:
        return "High Risk of Fraud", "red"      # Three or more indicators triggered

def calculate_tax(income):
    tax_slabs = [(250000, 0.05), (500000, 0.2), (1000000, 0.3)]
    tax = 0
    prev_limit = 250000  # Start from exemption limit
    
    if income <= 250000:
        return 0  # No tax
    
    for limit, rate in tax_slabs:
        if income > limit:
            tax += (limit - prev_limit) * rate
            prev_limit = limit
        else:
            tax += (income - prev_limit) * rate
            break
    
    tax += tax * 0.04  # Adding 4% Cess
    return round(tax, 2)

# Update the Custom CSS section with better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Card styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #f8f9fa;
        border-radius: 5px;
        color: #1565c0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1565c0;
        color: white;
    }
    
    /* Form styling */
    .stTextInput > div > div {
        background-color: white;
        padding: 0.5rem;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    
    .stNumberInput > div > div {
        background-color: white;
        padding: 0.5rem;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background-color: #1565c0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1565c0;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 500;
        width: 200px;
        height: 45px;
    }
    
    .stButton > button:hover {
        background-color: #1976d2;
    }
    
    /* Text styling */
    h1 {
        color: #1565c0;
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        color: #1976d2;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        margin: 1rem 0 !important;
    }
    
    label {
        color: #424242 !important;
        font-weight: 500 !important;
    }
    
    /* Results styling */
    .results-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    
    .income-item {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }
    
    /* Increase font size for input labels */
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label {
        color: #000000 !important;
        font-weight: 500 !important;
        font-size: 20px !important;
        margin-bottom: 10px !important;
    }
    
    /* Increase font size for input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        font-size: 18px !important;
        color: #000000 !important;
    }
    
    /* Increase font size for tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 22px !important;
    }
    
    /* Make all regular text larger */
    .stMarkdown p {
        font-size: 20px !important;
        color: #000000 !important;
    }
    
    /* Button text size */
    .stButton > button {
        font-size: 20px !important;
    }

    /* Input field styling */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        background-color: white !important;
        color: black !important;
        font-size: 16px !important;
        padding: 8px 12px !important;
        border: 1px solid #ccc !important;
        border-radius: 4px !important;
        caret-color: black !important;
    }

    /* Dropdown/Selectbox styling */
    .stSelectbox > div > div {
        background-color: white !important;
        color: black !important;
    }

    /* Dropdown options styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
        color: black !important;
    }

    .stSelectbox [data-baseweb="popup"] ul {
        background-color: white !important;
    }

    .stSelectbox [data-baseweb="popup"] li {
        color: black !important;
    }

    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background-color: #e0e0e0 !important;
    }

    .stSlider [data-baseweb="thumb"] {
        background-color: #1565c0 !important;
    }

    /* Number input buttons */
    .stNumberInput button {
        color: black !important;
        background-color: #f0f0f0 !important;
        border: 1px solid #ccc !important;
    }

    /* Labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: black !important;
        font-weight: 500 !important;
    }

    /* Make error messages black with better visibility */
    div[data-baseweb="notification"] {
        background-color: #ffd9d9 !important;
    }

    div[data-baseweb="notification"] div {
        color: black !important;
        font-weight: 700 !important;
        font-size: 20px !important;
    }

    /* Style for error icon */
    div[data-baseweb="notification"] svg {
        fill: black !important;
    }

    /* Additional specific styling for Streamlit error messages */
    .element-container div[data-testid="stImage"] {
        background-color: #ffd9d9 !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 20px !important;
    }

    .stAlert > div {
        color: black !important;
        font-weight: bold !important;
        font-size: 20px !important;
        background-color: #ffd9d9 !important;
    }

    /* Ensure all error text is black and bold */
    .stAlert p {
        color: black !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

def home_page():
    st.markdown("""
        <h1 style='color: #1565c0; font-size: 40px; margin-bottom: 30px;'>
            🏠 Welcome to Income Tax Fraud Detection System
        </h1>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Key Features section
        st.markdown("""
            <h2 style='color: #1565c0; font-size: 30px; margin: 20px 0;'>Key Features</h2>
            <div style='font-size: 24px; color: #000000; margin-left: 20px;'>
                <p>🔍 <strong>Real-time fraud detection</strong></p>
                <p>💰 <strong>Multiple income source analysis</strong></p>
            </div>
            
            <h2 style='color: #1565c0; font-size: 30px; margin: 30px 0 20px 0;'>How it works</h2>
            <div style='font-size: 24px; color: #000000; margin-left: 20px;'>
                <ol>
                    <li>Enter your personal and financial information</li>
                    <li>System analyzes the data using ML models</li>
                    <li>Get instant fraud detection results</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            st.image("images/tax_fraud.png", 
                    caption="Tax Fraud Detection System",
                    use_column_width=True)
        except:
            st.info("Please add tax_fraud.png to images folder")

def detection_page():
    st.markdown("""
        <h1 style='color: #1565c0; font-size: 40px; margin-bottom: 30px;'>
            🔍 Fraud Detection Analysis
        </h1>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Personal Information
        st.markdown("""
            <h2 style='color: #1565c0; font-size: 30px; margin: 20px 0;'>
                Personal Information
            </h2>
        """, unsafe_allow_html=True)
        
        name = st.text_input("Full Name", key="name")
        col3, col4 = st.columns(2)
        with col3:
            pan_card = st.text_input("PAN Card Number")
        with col4:
            aadhar_card = st.text_input("Aadhar Card Number")
        bank_account_no = st.text_input("Bank Account Number")
        age = st.slider("Age", min_value=20, max_value=100, value=30)
        
        # Financial Information
        st.markdown("""
            <h2 style='color: #1565c0; font-size: 30px; margin: 30px 0 20px 0;'>
                Financial Information
            </h2>
        """, unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        with col5:
            occupation = st.selectbox("Occupation", ["Salaried", "Self-employed", "Business"])
        with col6:
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        
        children = st.selectbox("Do you have children?", ["No", "Yes"])
        reported_income = st.number_input("Reported Income (₹)", 
                                        min_value=0.0, 
                                        format="%0.2f", 
                                        key="reported_income",
                                        help="Enter your total reported income for the year")
        
        # Additional Income
        st.markdown("""
            <h2 style='color: #1565c0; font-size: 30px; margin: 30px 0 20px 0;'>
                Additional Income Sources
            </h2>
        """, unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        with col7:
            has_interest_income = st.selectbox("Interest Income?", ["No", "Yes"])
            if has_interest_income == "Yes":
                interest_income = st.number_input("Interest Income Amount (₹)", min_value=0.0, format="%0.2f")
            else:
                interest_income = 0.0
        
        with col8:
            has_capital_gains = st.selectbox("Capital Gains?", ["No", "Yes"])
            if has_capital_gains == "Yes":
                capital_gains = st.number_input("Capital Gains Amount (₹)", min_value=0.0, format="%0.2f")
            else:
                capital_gains = 0.0
        
        # Detect Fraud Button
        if st.button("Detect Fraud"):
            if not name or not pan_card or not aadhar_card or not bank_account_no:
                st.error("Please fill in all personal information fields")
            elif reported_income <= 0:
                st.error("Please enter a valid reported income")
            else:
                # Prepare input data for prediction
                input_data = np.array([
                    age,
                    label_encoder_occupation.transform([occupation])[0],
                    label_encoder_marital.transform([marital_status])[0],
                    label_encoder_children.transform([children])[0],
                    interest_income,
                    reported_income * 0.1,  # Business income estimate
                    capital_gains,
                    reported_income * 0.05,  # Other income estimate
                    40000,  # Educational expenses
                    30000,  # Healthcare costs
                    50000,  # Lifestyle expenditure
                    25000,  # Other expenses
                    reported_income * 0.6,  # Bank debited
                    reported_income * 0.3,  # Credit card debited
                    reported_income        # Base income (this was missing)
                ])
                
                # Load model and make prediction
                model = load_best_model()
                predicted_income = predict_income(model, input_data)
                
                # Calculate total reported income
                total_reported_income = reported_income + interest_income + capital_gains
                
                # Show results
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.markdown('<h2 style="color: #1565c0; font-size: 30px; margin: 20px 0;">Results</h2>', unsafe_allow_html=True)
                
                col9, col10 = st.columns(2)
                with col9:
                    st.markdown("""
                        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                            <h3 style='color: #1565c0; font-size: 24px;'>Income Analysis</h3>
                            <p style='font-size: 18px; margin: 10px 0;'><strong>Predicted Income:</strong> ₹{:,.2f}</p>
                            <p style='font-size: 18px; margin: 10px 0;'><strong>Reported Income:</strong> ₹{:,.2f}</p>
                            <p style='font-size: 18px; margin: 10px 0;'><strong>Interest Income:</strong> ₹{:,.2f}</p>
                            <p style='font-size: 18px; margin: 10px 0;'><strong>Capital Gains:</strong> ₹{:,.2f}</p>
                            <p style='font-size: 18px; margin: 10px 0;'><strong>Total Income:</strong> ₹{:,.2f}</p>
                        </div>
                    """.format(
                        predicted_income,
                        reported_income,
                        interest_income,
                        capital_gains,
                        total_reported_income
                    ), unsafe_allow_html=True)
                
                with col10:
                    # Get fraud status
                    fraud_status, color = classify_fraud(total_reported_income, predicted_income, age, occupation)
                    st.markdown(
                        f"""
                        <div style="padding: 20px; 
                                   border-radius: 10px; 
                                   background-color: {color}; 
                                   color: white;
                                   text-align: center;
                                   font-size: 24px;
                                   margin-top: 20px;">
                            {fraud_status}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    with col2:
        try:
            st.image("images/analysis.png", use_column_width=True)
        except:
            st.info("Add analysis.png to images folder")

def main():
    # Create tabs
    tabs = st.tabs(["🏠 Home", "🔍 Detection"])
    
    with tabs[0]:
        home_page()
    
    with tabs[1]:
        detection_page()

if __name__ == "__main__":
    main() 