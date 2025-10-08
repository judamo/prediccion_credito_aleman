import streamlit as st
import pandas as pd
import joblib
import os

st.title("Credit Risk Prediction Form")


# Load the preprocessors and model
@st.cache_resource
def load_resources():
    le_sex      = joblib.load('label_encoder_sex.joblib')
    le_housing  = joblib.load('label_encoder_housing.joblib')
    le_saving   = joblib.load('label_encoder_saving.joblib')
    ohe         = joblib.load('one_hot_encoder.joblib')
    knn_model   = joblib.load('best_knn_model_2025-10-07.pkl')
    return le_sex,le_housing,le_saving, ohe, knn_model

le_sex,le_housing,le_saving, ohe, knn_model = load_resources()

# --- Input Form ---
st.header("Enter Customer Details:")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=25, max_value=60, value=30)
    sex = st.selectbox("Sex", ["female", "male"])
    job = st.selectbox("Job", [0.0, 1.0, 2.0, 3.0]) # Assuming these are the possible job categories based on your one-hot encoding
    housing_options = ["own", "free", "rent"]
    housing = st.selectbox('Housing', housing_options)
    #housing = st.selectbox("Housing", ["own", "free", "rent"]) # Assuming these are the possible housing types
    saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"]) # Assuming these are the possible saving account levels
    checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"]) # Assuming these are the possible checking account levels
    credit_amount = st.number_input("Credit amount", min_value=500, max_value=10000,value=5000)
    duration = st.number_input("Duration (months)", min_value=6, value=36)

    submitted = st.form_submit_button("Get Prediction")

# --- Prediction ---
if submitted:
    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration]
    })
    
    #input_data = pd.DataFrame([[age, sex, job, housing, saving_accounts, checking_account, credit_amount,duration]], columns=['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration'])
    
    input_data['Housing'] = input_data['Housing'].astype('category')
    
    print("******************************************************")
    
    st.dataframe(input_data) 

    try:
        input_data['Sex_encoded'] = le_sex.transform(input_data[['Sex']])
        input_data['Housing_encoded'] = le_housing.transform( input_data[['Housing']] )
        input_data['Saving accounts_encoded'] = le_saving.transform(input_data[['Saving accounts']])
        
    except ValueError as e:
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
