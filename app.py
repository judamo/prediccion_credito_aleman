import streamlit as st
import pandas as pd
import joblib
import os

st.title("Credit Risk Prediction")

# Define the path to the directory containing the files
# Assuming the Streamlit app will be run from the same directory or a subdirectory
# where the 'deploy_credito' folder is accessible.
# You might need to adjust this path based on your deployment environment.
path = "deploy_credito" # Adjust this path if necessary

# --- Data Loading ---
filename_test = "datos_credito_alemania_qa_test.csv"
try:
    df = pd.read_csv(os.path.join(path, filename_test), sep=";")
    st.write("Original Data:")
    st.write(df)
except FileNotFoundError:
    st.error(f"Error: {filename_test} not found in {path}. Please ensure the file is in the correct location.")
    st.stop()

# --- Preprocessing ---
label_encoder_file = "label_encoder.joblib"
one_hot_encoder_file = "one_hot_encoder.joblib"

try:
    le = joblib.load(os.path.join(path, label_encoder_file))
    ohe = joblib.load(os.path.join(path, one_hot_encoder_file))
except FileNotFoundError:
    st.error(f"Error: Encoder files not found in {path}. Please ensure '{label_encoder_file}' and '{one_hot_encoder_file}' are in the correct location.")
    st.stop()


# Apply label encoding
try:
    df['Housing_encoded'] = le.transform(df['Housing'])
    df['Saving accounts_encoded'] = le.transform(df['Saving accounts'])
    # Ensure the label encoder has been trained on both 'male' and 'female' for the Sex column
    df['Sex_encoded'] = le.transform(df['Sex'])
except ValueError as e:
    st.error(f"Error during label encoding: {e}. This might be due to unseen labels in the data.")
    st.stop()


# Apply one-hot encoding to 'Job'
try:
    job_encoded = ohe.transform(df[['Job']])
    job_encoded_df = pd.DataFrame(job_encoded, columns=ohe.get_feature_names_out(['Job']), index=df.index)
    df = pd.concat([df.drop('Job', axis=1), job_encoded_df], axis=1)
except ValueError as e:
     st.error(f"Error during one-hot encoding: {e}. This might be due to unseen categories in the 'Job' column.")
     st.stop()


# Drop original columns
df = df.drop(['Sex', 'Housing', 'Saving accounts', 'Checking account'], axis=1)

# Reorder columns to match the training data if necessary
# This part is crucial for the model prediction
# Assuming the model was trained on columns in this specific order.
# You might need to explicitly define the expected column order.
expected_columns = ['Age', 'Credit amount', 'Duration', 'Sex_encoded', 'Housing_encoded', 'Saving accounts_encoded', 'Job_0.0', 'Job_1.0', 'Job_2.0', 'Job_3.0']
# Ensure all expected columns are in the dataframe after preprocessing
missing_cols = set(expected_columns) - set(df.columns)
if missing_cols:
    st.error(f"Error: Missing columns after preprocessing: {missing_cols}. Please check your preprocessing steps.")
    st.stop()

# Reindex the dataframe to ensure the columns are in the correct order for prediction
df = df.reindex(columns=expected_columns)


st.write("Processed Data (ready for prediction):")
st.write(df)

# --- Model Loading and Prediction ---
best_knn_model_file = "best_knn_model_2025-10-07.pkl"
try:
    knn_model = joblib.load(os.path.join(path, best_knn_model_file))
except FileNotFoundError:
    st.error(f"Error: Model file not found in {path}. Please ensure '{best_knn_model_file}' is in the correct location.")
    st.stop()


# Make predictions
try:
    y_pred = knn_model.predict(df)
    st.write("Predictions:")
    st.write(y_pred)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()
