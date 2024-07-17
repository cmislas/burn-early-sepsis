import streamlit as st
import pandas as pd
import joblib
import gzip
from sklearn.ensemble import RandomForestClassifier

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
    st.subheader("Patient Data")
    st.write(data)
    
    features = ['HR', 'Temp', 'Hour', 'WBC', 'MAP']
    X = data[features]

    model = load_model()
    predictions = model.predict(X)
    data['Sepsis Risk'] = predictions

    st.subheader("Predictions")
    st.write(data[['HR', 'Temp', 'Hour', 'WBC', 'MAP', 'Sepsis Risk']])

    st.subheader("Sepsis Risk Count")
    st.bar_chart(data['Sepsis Risk'].value_counts())

    st.subheader("Feature Distribution")
    for feature in features:
        st.write(f"Distribution of {feature}")
        st.bar_chart(data[feature])

if __name__ == "__main__":
    main()


