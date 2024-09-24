import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv('Online_Transactions.csv', delimiter=';')

# Data preprocessing
data['amt_of_transactions'] = data['amt_of_transactions'].str.replace(',', '').astype(float)
data['no_of_active_costumers'] = pd.to_numeric(data['no_of_active_costumers'], errors='coerce')
data = data.dropna()

# Features and target variables
X = data[['no_of_transactions', 'no_of_active_costumers']]
y_transactions = data['amt_of_transactions']
y_customers = data['no_of_active_costumers']

# Split the data
X_train, X_test, y_train_transactions, y_test_transactions = train_test_split(X, y_transactions, test_size=0.2, random_state=42)
_, _, y_train_customers, y_test_customers = train_test_split(X, y_customers, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
transaction_model = RandomForestRegressor(random_state=42)
transaction_model.fit(X_train_scaled, y_train_transactions)

customer_model = RandomForestRegressor(random_state=42)
customer_model.fit(X_train_scaled, y_train_customers)

# Streamlit app interface
st.title('Bank Transaction and Customer Predictor')

# Get all unique bank names
bank_names = data['bank_name'].unique()

# Add an empty option to the list of bank names for the blank search bar
bank_names_with_blank = [''] + list(bank_names)

# Bank prediction input with dropdown (selectbox)
bank_name = st.selectbox('Select Bank Name', bank_names_with_blank)

# Submit button for prediction
if st.button('Submit'):
    def predict_by_bank_name(bank_name):
        bank_data = data[data['bank_name'] == bank_name]
        if bank_data.empty:
            return f"No data found for bank name: {bank_name}"

        no_of_transactions = bank_data['no_of_transactions'].values[0]
        no_of_active_customers = bank_data['no_of_active_costumers'].values[0]

        input_data = pd.DataFrame({
            'no_of_transactions': [no_of_transactions],
            'no_of_active_costumers': [no_of_active_customers]
        })

        input_scaled = scaler.transform(input_data)

        predicted_transactions = transaction_model.predict(input_scaled)[0]
        predicted_customers = customer_model.predict(input_scaled)[0]

        return {
            'Bank Name': bank_name,
            'Predicted Transaction Amount for Next Month': predicted_transactions,
            'Predicted Number of Active Customers for Next Month': int(predicted_customers)
        }

    if bank_name:  # Check if a bank is selected
        results = predict_by_bank_name(bank_name)

        if isinstance(results, dict):
            st.write(f"**Bank:** {results['Bank Name']}")
            st.write(f"**Predicted Transaction Amount for Next Month:** ₹{results['Predicted Transaction Amount for Next Month']:,.2f}")
            st.write(f"**Predicted Number of Active Customers for Next Month:** {results['Predicted Number of Active Customers for Next Month']}")
        else:
            st.write(results)


# C:\Users\CSP\AppData\Roaming\Python\Python312\Scripts\streamlit run app.py
