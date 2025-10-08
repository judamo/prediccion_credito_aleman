import streamlit as st
import pandas as pd
import joblib
import os

st.title("Credit Risk Prediction Form")


# Load the preprocessors and model
@st.cache_resource
def load_resources():
    le = joblib.load('label_encoder.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')
    knn_model = joblib.load('best_knn_model_2025-10-07.pkl')
    return le, ohe, knn_model

le, ohe, knn_model = load_resources()

# --- Input Form ---
st.header("Enter Customer Details:")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["female", "male"])
    job = st.selectbox("Job", [0.0, 1.0, 2.0, 3.0]) # Assuming these are the possible job categories based on your one-hot encoding
    housing = st.selectbox("Housing", ["own", "free", "rent"]) # Assuming these are the possible housing types
    saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"]) # Assuming these are the possible saving account levels
    checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"]) # Assuming these are the possible checking account levels
    credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
    duration = st.number_input("Duration (months)", min_value=0, value=12)

    submitted = st.form_submit_button("Get Prediction")

# --- Prediction ---
if submitted:
    # Create a DataFrame from input data
    """input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration]
    })"""
    
    input_data = pd.DataFrame([[age, sex, job, housing, saving_accounts, checking_account, credit_amount,duration]], columns=['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration'])
    
    print("******************************************************")
    print(input_data)
    # --- Preprocessing Input Data ---
    # Apply label encoding
    try:
        input_data['Housing_encoded'] = le.transform(input_data[['Housing']])
        input_data['Saving accounts_encoded'] = le.transform(input_data['Saving accounts'])
        input_data['Sex_encoded'] = le.transform(input_data['Sex'])
    except ValueError as e:
        st.error(f"jdm {housing}---{input_data['Housing']} ")
        st.error(f"Error during label encoding input: {e}. Please check the input values.")
        st.stop()

    # Apply one-hot encoding to 'Job'
    try:
        job_encoded_input = ohe.transform(input_data[['Job']])
        job_encoded_input_df = pd.DataFrame(job_encoded_input, columns=ohe.get_feature_names_out(['Job']), index=input_data.index)
        input_data = pd.concat([input_data.drop('Job', axis=1), job_encoded_input_df], axis=1)
    except ValueError as e:
         st.error(f"Error during one-hot encoding input: {e}. Please check the input value for 'Job'.")
         st.stop()


    # Drop original columns used for encoding
    input_data = input_data.drop(['Sex', 'Housing', 'Saving accounts', 'Checking account'], axis=1)

    # Reorder columns to match the training data
    # Assuming the model was trained on columns in this specific order.
    expected_columns = ['Age', 'Credit amount', 'Duration', 'Sex_encoded', 'Housing_encoded', 'Saving accounts_encoded', 'Job_0.0', 'Job_1.0', 'Job_2.0', 'Job_3.0']
    try:
        input_data = input_data.reindex(columns=expected_columns, fill_value=0) # Use fill_value=0 for missing one-hot encoded columns
    except Exception as e:
        st.error(f"Error reordering columns: {e}")
        st.stop()

    st.write("Processed Input Data:")
    st.write(input_data)

    # --- Make Prediction ---
    try:
        prediction = knn_model.predict(input_data)
        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.success("Credit Approved")
        else:
            st.error("Credit Denied")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
