import streamlit as st
import pandas as pd 
import joblib
import catboost

model = joblib.load('E:\MASTERS\Term 3-Winter 1\Advance topics in Inf Sys\Group Project\catboost.pkl')

st.title('Customer Churn Prediction')

# Create an input field for the user to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file of your customer to predit whether he/she will be churned", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded DataFrame
    st.write('Uploaded DataFrame:', df)
    
    # Assuming the model expects certain columns for prediction,
    # you can select the relevant columns from the DataFrame
    # and make predictions
    X = df[['Customer_Age', 'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Months_on_book', 'Total_Relationship_Count',
       'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
       'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
       'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
       'Avg_Utilization_Ratio']]  # Adjust column names as per your data
    predictions = model.predict(X)
    
    # Convert predictions to human-readable messages
    prediction_messages = ['The customer will retain' if pred == 0 else 'The customer will be churned' for pred in predictions]
    
    # Display the predictions
    st.write('Predictions:')
    for pred_msg in prediction_messages:
        st.write(pred_msg)