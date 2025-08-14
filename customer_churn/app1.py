import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("final_gb_classifier.pkl")

model = load_model()

# Preprocessing function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })
    return df

# Page title
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to **churn** based on their service details.")

# Sidebar inputs
st.sidebar.header("📋 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 0 if gender == "Female" else 1

senior_citizen = st.sidebar.radio("Senior Citizen", ["No", "Yes"])
senior_citizen = 1 if senior_citizen == "Yes" else 0

partner = st.sidebar.radio("Has Partner?", ["No", "Yes"])
partner = 1 if partner == "Yes" else 0

dependents = st.sidebar.radio("Has Dependents?", ["No", "Yes"])
dependents = 1 if dependents == "Yes" else 0

phone_service = st.sidebar.radio("Phone Service", ["No", "Yes"])
phone_service = 1 if phone_service == "Yes" else 0

multiple_lines = st.sidebar.radio("Multiple Lines", ["No", "Yes"])
multiple_lines = 1 if multiple_lines == "Yes" else 0

internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

online_security = st.sidebar.radio("Online Security", ["No", "Yes", "Not Applicable"])
online_security = {"No": 0, "Yes": 1, "Not Applicable": 2}[online_security]

online_backup = st.sidebar.radio("Online Backup", ["No", "Yes", "Not Applicable"])
online_backup = {"No": 0, "Yes": 1, "Not Applicable": 2}[online_backup]

device_protection = st.sidebar.radio("Device Protection", ["No", "Yes", "Not Applicable"])
device_protection = {"No": 0, "Yes": 1, "Not Applicable": 2}[device_protection]

tech_support = st.sidebar.radio("Tech Support", ["No", "Yes", "Not Applicable"])
tech_support = {"No": 0, "Yes": 1, "Not Applicable": 2}[tech_support]

streaming_tv = st.sidebar.radio("Streaming TV", ["No", "Yes"])
streaming_tv = 1 if streaming_tv == "Yes" else 0

streaming_movies = st.sidebar.radio("Streaming Movies", ["No", "Yes"])
streaming_movies = 1 if streaming_movies == "Yes" else 0

contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

paperless_billing = st.sidebar.radio("Paperless Billing", ["No", "Yes"])
paperless_billing = 1 if paperless_billing == "Yes" else 0

payment_method = st.sidebar.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 
    'Bank transfer (automatic)', 'Credit card (automatic)'
])

monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, format="%1.0f")
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, format="%1.0f")
tenure_group = st.sidebar.number_input("Tenure Group", min_value=0)

# Prediction section
if st.button("🔍 Predict Churn"):
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }

    processed = preprocess_input(user_data)
    prediction = model.predict(processed)[0]

    st.subheader("📈 Prediction Result")
    if prediction == 1:
        st.error("❌ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **likely to stay**.")
