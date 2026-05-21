import pickle
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open("Logistic.pkl", "rb"))

st.set_page_config(page_title="Customer Prediction App", layout="centered")

st.title("📊 Customer Churn Prediction App")

# ---------------- INPUT SECTION ---------------- #

# Numerical inputs
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72)
MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 10000.0)
Total_Charges = st.number_input("Total_Charges", 0.0, 100000.0)

# Binary categorical
partner = st.selectbox("Has Partner?", ["Yes", "No"])
Gender = st.selectbox("Gender", ["Male", "Female"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phoneService = st.selectbox("PhoneService?", ["Yes", "No"])
paperlessbilling = st.selectbox("Paperless Billing?", ["Yes", "No"])

# Example categorical (one-hot style)
MultipleLines= st.selectbox("Multiple Lines", [ "No", "No phone service","Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", 
                            ["Bank transfer (automatic)", "Credit card (automatic)", 
                             "Electronic check", "Mailed check"])
# scaling
scaler = StandardScaler()
#Encoding
Gender = 1 if Gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
phoneService = 1 if phoneService == "Yes" else 0
paperlessbilling = 1 if paperlessbilling == "Yes" else 0

# One-hot encoding
MultipleLines_No = 1 if MultipleLines == "No" else 0
MultipleLines_No_phone_service = 1 if MultipleLines == "No phone service" else 0
MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0
InternetService_DSL= 1 if InternetService == "DSL" else 0
InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0
InternetService_No = 1 if InternetService == "No" else 0
OnlineSecurity_No = 1 if OnlineSecurity == "No" else 0
OnlineSecurity_No_internet_service = 1 if OnlineSecurity == "No internet service" else 0
OnlineSecurity_Yes = 1 if OnlineSecurity == "Yes" else 0
OnlineBackup_No = 1 if OnlineBackup == "No" else 0
OnlineBackup_No_internet_service = 1 if OnlineBackup == "No internet service" else 0
OnlineBackup_Yes = 1 if OnlineBackup == "Yes" else 0
DeviceProtection_No = 1 if DeviceProtection == "No" else 0
DeviceProtection_No_internet_service = 1 if DeviceProtection == "No internet service" else 0
DeviceProtection_Yes = 1 if DeviceProtection == "Yes" else 0
TechSupport_No = 1 if TechSupport == "No" else 0
TechSupport_No_internet_service = 1 if TechSupport == "No internet service" else 0
TechSupport_Yes = 1 if TechSupport == "Yes" else 0
StreamingTV_No = 1 if StreamingTV == "No" else 0
StreamingTV_No_internet_service = 1 if StreamingTV == "No internet service" else 0
StreamingTV_Yes = 1 if StreamingTV == "Yes" else 0
StreamingMovies_No = 1 if StreamingMovies == "No" else 0
StreamingMovies_No_internet_service = 1 if StreamingMovies == "No internet service" else 0
StreamingMovies_Yes = 1 if StreamingMovies == "Yes" else 0
Contract_Month_to_month = 1 if Contract == "Month-to-month" else 0
Contract_One_year = 1 if Contract == "One year" else 0
Contract_Two_year = 1 if Contract == "Two year" else 0

PaymentMethod_Bank_transfer_automatic = 1 if PaymentMethod == "Bank transfer (automatic)" else 0
PaymentMethod_Credit_card_automatic = 1 if PaymentMethod == "Credit card (automatic)" else 0
PaymentMethod_Electronic_check = 1 if PaymentMethod == "Electronic check" else 0
PaymentMethod_Mailed_check = 1 if PaymentMethod == "Mailed check" else 0
# define dataframe
input_features = pd.DataFrame({'SeniorCitizen':[SeniorCitizen],'tenure':[tenure],'MonthlyCharges':[MonthlyCharges],'TotalCharges':[Total_Charges],
                               'Gender':[Gender],'Partner':[partner],'Dependents':[dependents],
                               'PhoneService':[phoneService],'PaperlessBilling':[paperlessbilling],
                               'MultipleLines_No':[MultipleLines_No],'MultipleLines_No_phone_service':[MultipleLines_No_phone_service],
                               'MultipleLines_Yes':[MultipleLines_Yes],'OnlineSecurity_No':[OnlineSecurity_No],
                               'InternetService_DSL':[InternetService_DSL],'InternetService_Fiber_optic':[InternetService_Fiber_optic],
                               'InternetService_No':[InternetService_No],
                               'OnlineSecurity_No_internet_service':[OnlineSecurity_No_internet_service],
                               'OnlineSecurity_Yes':[OnlineSecurity_Yes],'OnlineBackup_No':[OnlineBackup_No],
                               'OnlineBackup_No_internet_service':[OnlineBackup_No_internet_service],
                               'OnlineBackup_Yes':[OnlineBackup_Yes],'DeviceProtection_No':[DeviceProtection_No],
                               'DeviceProtection_No_internet_service':[DeviceProtection_No_internet_service],
                               'DeviceProtection_Yes':[DeviceProtection_Yes],'TechSupport_No':[TechSupport_No],
                               'TechSupport_No_internet_service':[TechSupport_No_internet_service],
                               'TechSupport_Yes':[TechSupport_Yes],'StreamingTV_No':[StreamingTV_No],
                               'StreamingTV_No_internet_service':[StreamingTV_No_internet_service],
                               'StreamingTV_Yes':[StreamingTV_Yes],'StreamingMovies_No':[StreamingMovies_No],
                               'StreamingMovies_No_internet_service':[StreamingMovies_No_internet_service],
                               'StreamingMovies_Yes':[StreamingMovies_Yes],'Contract_Month_to_month':[Contract_Month_to_month],
                               'Contract_One_year':[Contract_One_year],'Contract_Two_year':[Contract_Two_year],
                               'PaymentMethod_Bank_transfer_automatic':[PaymentMethod_Bank_transfer_automatic],
                               'PaymentMethod_Credit_card_automatic':[PaymentMethod_Credit_card_automatic],
                               'PaymentMethod_Electronic_check':[PaymentMethod_Electronic_check],
                               'PaymentMethod_Mailed_check':[PaymentMethod_Mailed_check]})
columns = pickle.load(open("columns.pkl", "rb"))

# Align columns
input_features = input_features.reindex(columns=columns, fill_value=0)


# ---------------- PREDICTION ---------------- #
if st.button('Predict'):
  predictions = model.predict(input_features)
  if predictions==1:
    st.error('⚠️ Customer will Churn')
  else:
    st.success('✅ Customer will Stay')
