import streamlit as st
import pandas as pd 
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained CatBoost model
model = joblib.load(r'E:\MASTERS\Term 3-Winter 1\Advance topics in Inf Sys\Group Project\catboost.pkl')

# Title of the Streamlit app
st.title('Customer Churn Prediction')

# Create an input field for the user to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file of your customers to predict whether they will churn", type=["csv"])

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
    
    # Make predictions
    predictions = model.predict(X)
    
    # Convert predictions to churn messages
    churn_messages = ['Customer will churn' if pred == 1 else 'Customer will not churn' for pred in predictions]
    
    # Create a DataFrame to display churn messages with serial numbers
    churn_df = pd.DataFrame({'Serial': range(1, len(churn_messages) + 1), 'Churn Message': churn_messages})
    
    # Display Prediction label
    st.write('Prediction:')
    
    # Display churn messages in a table
    st.write(churn_df)
    
    # Button to generate SHAP summary plot
    if st.button('Generate Importance'):
        # Explain model predictions using SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Plot SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot()
