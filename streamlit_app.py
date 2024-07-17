import streamlit as st
import pandas as pd
import gzip
import joblib

@st.cache_data
def load_data():
    return pd.read_csv('database3.csv')

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
    for feature in features:
        st.subheader(f"{feature} Distribution")
        st.bar_chart(data[feature])

if __name__ == "__main__":
    main()
