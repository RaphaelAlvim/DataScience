import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Carregar o modelo treinado
model = load('best_lgbm_model.joblib')

# Definir a interface no Streamlit
st.title("Churn Prediction App")

# Coletar dados do cliente
st.sidebar.header("Enter Customer Data:")

# Input do perfil do cliente
contract_duration = st.sidebar.slider('Contract Duration (days)', 0, 2000, 30)
paperless_billing = st.sidebar.selectbox('Paperless Billing', [0, 1])
monthly_charges = st.sidebar.slider('Monthly Charges ($)', 0.0, 150.0, 50.0)
avg_charge_per_duration = st.sidebar.slider('Average Charge Per Day ($)', 0.0, 5.0, 1.0)
senior_citizen = st.sidebar.selectbox('Is Senior Citizen?', [0, 1])
partner = st.sidebar.selectbox('Has Partner?', [0, 1])
dependents = st.sidebar.selectbox('Has Dependents?', [0, 1])
phone = st.sidebar.selectbox('Has Phone Service?', [0.0, 1.0])
multiple_lines = st.sidebar.selectbox('Has Multiple Lines?', [0.0, 1.0])
internet = st.sidebar.selectbox('Has Internet Service?', [0.0, 1.0])
online_security = st.sidebar.selectbox('Has Online Security?', [0.0, 1.0])
online_backup = st.sidebar.selectbox('Has Online Backup?', [0.0, 1.0])
device_protection = st.sidebar.selectbox('Has Device Protection?', [0.0, 1.0])
tech_support = st.sidebar.selectbox('Has Tech Support?', [0.0, 1.0])
streaming_tv = st.sidebar.selectbox('Has Streaming TV?', [0.0, 1.0])
streaming_movies = st.sidebar.selectbox('Has Streaming Movies?', [0.0, 1.0])

# Dummy encoding para tipo de contrato e método de pagamento
type_one_year = st.sidebar.selectbox('Contract Type: One Year', [False, True])
type_two_year = st.sidebar.selectbox('Contract Type: Two Year', [False, True])
payment_method_credit_card_automatic = st.sidebar.selectbox('Payment: Credit Card Auto', [False, True])
payment_method_electronic_check = st.sidebar.selectbox('Payment: Electronic Check', [False, True])
payment_method_mailed_check = st.sidebar.selectbox('Payment: Mailed Check', [False, True])
internet_service_fiber_optic = st.sidebar.selectbox('Internet Service: Fiber Optic', [False, True])
internet_service_no_internet = st.sidebar.selectbox('No Internet Service', [False, True])

# Organize os dados em um DataFrame
data = {
    'contract_duration': contract_duration,
    'paperless_billing': paperless_billing,
    'monthly_charges': monthly_charges,
    'avg_charge_per_duration': avg_charge_per_duration,
    'senior_citizen': senior_citizen,
    'partner': partner,
    'dependents': dependents,
    'phone': phone,
    'multiple_lines': multiple_lines,
    'internet': internet,
    'online_security': online_security,
    'online_backup': online_backup,
    'device_protection': device_protection,
    'tech_support': tech_support,
    'streaming_tv': streaming_tv,
    'streaming_movies': streaming_movies,
    'type_one_year': type_one_year,
    'type_two_year': type_two_year,
    'payment_method_credit_card_automatic': payment_method_credit_card_automatic,
    'payment_method_electronic_check': payment_method_electronic_check,
    'payment_method_mailed_check': payment_method_mailed_check,
    'internet_service_fiber_optic': internet_service_fiber_optic,
    'internet_service_no_internet': internet_service_no_internet,
}

df_input = pd.DataFrame([data])

# Previsão de churn
prediction_proba = model.predict_proba(df_input)[:, 1]  # Probabilidade de churn

st.subheader('Churn Probability:')
st.write(f"{prediction_proba[0]:.2%}")

if prediction_proba[0] > 0.5:
    st.warning("High risk of churn!")
else:
    st.success("Low risk of churn!")