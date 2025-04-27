import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Create and save label encoders
label_encoder_occupation = LabelEncoder()
label_encoder_occupation.fit(['Salaried', 'Self-employed', 'Business'])
joblib.dump(label_encoder_occupation, "label_encoder_occupation.joblib")

label_encoder_marital = LabelEncoder()
label_encoder_marital.fit(['Single', 'Married'])
joblib.dump(label_encoder_marital, "label_encoder_marital_status.joblib")

label_encoder_children = LabelEncoder()
label_encoder_children.fit(['No', 'Yes'])
joblib.dump(label_encoder_children, "label_encoder_children.joblib")

# Create model with better parameters
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=5,
    random_state=42
)

# Generate training data
n_samples = 5000  # Increased samples for better accuracy
np.random.seed(42)

# Generate base features
age = np.random.randint(25, 65, n_samples)
occupation = np.random.randint(0, 3, n_samples)
marital_status = np.random.randint(0, 2, n_samples)
children = np.random.randint(0, 2, n_samples)

# Generate income with more realistic ranges
base_income = np.zeros(n_samples)

# Realistic salary ranges based on Indian income standards
for i in range(n_samples):
    if occupation[i] == 0:  # Salaried
        if age[i] < 30:
            base_income[i] = np.random.normal(400000, 100000)  # 4L ± 1L
        elif age[i] < 40:
            base_income[i] = np.random.normal(600000, 150000)  # 6L ± 1.5L
        else:
            base_income[i] = np.random.normal(800000, 200000)  # 8L ± 2L
            
    elif occupation[i] == 1:  # Self-employed
        if age[i] < 30:
            base_income[i] = np.random.normal(500000, 150000)  # 5L ± 1.5L
        elif age[i] < 40:
            base_income[i] = np.random.normal(800000, 200000)  # 8L ± 2L
        else:
            base_income[i] = np.random.normal(1000000, 300000)  # 10L ± 3L
            
    else:  # Business
        if age[i] < 30:
            base_income[i] = np.random.normal(600000, 200000)  # 6L ± 2L
        elif age[i] < 40:
            base_income[i] = np.random.normal(1000000, 300000)  # 10L ± 3L
        else:
            base_income[i] = np.random.normal(1500000, 400000)  # 15L ± 4L
    
    # Realistic factor adjustments
    # Age factor (gradual increase)
    base_income[i] *= (1 + (age[i] - 25) * 0.003)  # 0.3% increase per year
    
    # Marital status and children factors
    if marital_status[i] == 1:  # Married
        base_income[i] *= 1.05  # 5% increase
    if children[i] == 1:  # Has children
        base_income[i] *= 1.02  # 2% increase

# Create feature matrix with realistic additional incomes
X = np.column_stack([
    age,
    occupation,
    marital_status,
    children,
    np.random.normal(base_income * 0.05, base_income * 0.01, n_samples),  # Interest income (5% of base)
    np.random.normal(base_income * 0.08, base_income * 0.02, n_samples),  # Business income
    np.random.normal(base_income * 0.06, base_income * 0.015, n_samples), # Capital gains
    np.random.normal(base_income * 0.03, base_income * 0.01, n_samples),  # Other income
    np.random.normal(base_income * 0.15, base_income * 0.03, n_samples),  # Educational expenses
    np.random.normal(base_income * 0.10, base_income * 0.02, n_samples),  # Healthcare costs
    np.random.normal(base_income * 0.20, base_income * 0.04, n_samples),  # Lifestyle expenditure
    np.random.normal(base_income * 0.12, base_income * 0.03, n_samples),  # Other expenses
    np.random.normal(base_income * 0.60, base_income * 0.10, n_samples),  # Bank debited
    np.random.normal(base_income * 0.30, base_income * 0.05, n_samples),  # Credit card debited
    base_income
])

# Ensure all values are positive
X = np.abs(X)
base_income = np.abs(base_income)

# Train the model
model.fit(X, base_income)

# Save the model
joblib.dump(model, "best_model.joblib")

print("Model trained and saved successfully!") 