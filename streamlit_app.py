import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import joblib
import gzip

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
    st.write(data)

    st.subheader("Feature Importances")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    st.bar_chart(importance_df.set_index('Feature'))

    # Additional visualizations for individual features
    for feature in features:
        st.subheader(f'{feature} Distribution')
        st.bar_chart(data[feature])

if __name__ == "__main__":
    main()

