import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import gzip
import joblib

@st.cache_data
def load_data():
    return pd.read_csv('sepsis3.csv')

@st.cache_resource
def load_model():
    with gzip.open('sepsis_model.joblib.gz', 'rb') as f:
        model = joblib.load(f)
    return model

def main():
    st.title("Early Sepsis Detection for Burn Patients")

    data = load_data()

    X = data.drop(['SepsisLabel', 'Patient_ID'], axis=1)
    y = data['SepsisLabel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model()

    # Get prediction probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Add probabilities to the test data
    test_data_with_prob = X_test.copy()
    test_data_with_prob['Probability'] = probabilities
    test_data_with_prob['Patient_ID'] = data.loc[X_test.index, 'Patient_ID']
    test_data_with_prob['SepsisLabel'] = y_test

    # Select top 5 patients at risk
    top_risk_patients = test_data_with_prob.sort_values(by='Probability', ascending=False).head(5)

    st.subheader("Top 5 Patients at Risk of Sepsis")
    st.table(top_risk_patients[['Patient_ID', 'HR', 'Temp', 'Hour', 'WBC', 'MAP', 'Probability']])

if __name__ == "__main__":
    main()



