import streamlit as st
import pandas as pd
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
    model = load_model()

    features = ['HR', 'Temp', 'Hour', 'WBC', 'MAP']
    X = data[features]

    # Make predictions
    predictions = model.predict_proba(X)[:, 1]  # Get probability of positive class
    data['Sepsis Risk'] = predictions

    # Sort by sepsis risk in descending order and get top 5 patients
    top_5_patients = data.nlargest(5, 'Sepsis Risk')

    st.subheader("Top 5 Patients at Risk of Sepsis")
    st.write(top_5_patients[['HR', 'Temp', 'Hour', 'WBC', 'MAP', 'Sepsis Risk']])

    st.subheader("Patient Data")
    st.write(data)

    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importance.set_index('Feature'))

if __name__ == "__main__":
    main()



