#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import gzip
import shutil

# Load your dataset
data = pd.read_csv('/Users/claudiaislas/Desktop/Sepsis3.csv')  # Update the path to your actual file location

# Define features and target
features = ['HR', 'Temp', 'Hour', 'WBC', 'MAP']
target = 'SepsisLabel'  # Ensure this is the correct column name for the target variable

X = data[features]
y = data[target]

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'sepsis_model.joblib')
print("Model trained and saved as 'sepsis_model.joblib'")

# Compress the model file
with open('sepsis_model.joblib', 'rb') as f_in:
    with gzip.open('sepsis_model.joblib.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Model file compressed as 'sepsis_model.joblib.gz'")


# In[ ]:




