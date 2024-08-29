import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title of the app
st.title("Bank Loan Study and Prediction")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df_application = pd.read_csv(uploaded_file)
    st.write(df_application.head())  # Display the first few rows of the dataset

    # You can add more analysis here, for example, showing summaries:
    st.subheader("Dataset Summary")
    st.write(df_application.describe())

    # Example: Distribution of a specific feature
    st.subheader("Distribution of AMT_CREDIT")
    plt.figure(figsize=(10, 6))
    sns.histplot(df_application['AMT_CREDIT'], kde=True, bins=30)
    st.pyplot(plt)

    # Example: Correlation Heatmap
    '''st.subheader("Correlation Heatmap")
    corr = df_application.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)'''

    # Model Training and Prediction
    st.subheader("Loan Default Prediction")

    # Preprocess data
    X = df_application.drop(columns=['TARGET'])  # Features
    y = df_application['TARGET']  # Target variable

    # For simplicity, using only numerical features
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model (this can be modified according to your original notebook)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Allow users to input their data for prediction
    st.sidebar.subheader("Predict on New Data")
    input_data = {
        'AMT_INCOME_TOTAL': st.sidebar.number_input('Annual Income', min_value=0),
        'AMT_CREDIT': st.sidebar.number_input('Credit Amount', min_value=0),
        'AMT_ANNUITY': st.sidebar.number_input('Annuity Amount', min_value=0),
        'CNT_CHILDREN': st.sidebar.number_input('Number of Children', min_value=0, step=1)
        # Add more features as needed
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all columns match the model's training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)

    st.write(f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}")

else:
    st.info('Awaiting for CSV file to be uploaded.')

# You can add more features like filtering options, model predictions, etc.
