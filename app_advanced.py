import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the advanced pipeline file
pipeline = joblib.load("advanced_iris_pipeline.pkl")
target_names = ['Setosa', 'Versicolor', 'Virginica']

st.set_page_config(page_title="Advanced Iris AI", page_icon="📈", layout="centered")
st.title("📈 Advanced Iris Predictor with Analytics")
st.write("This enhanced version uses a scaled data pipeline and visualizes real-time prediction confidence.")
st.write("---")

# User Inputs
st.header("🎛️ Flower Features")
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width  = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width  = st.slider("Petal Width", 0.1, 2.5, 0.2)

st.write("---")

# Features array
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Calculate Prediction AND Probabilities
prediction = pipeline.predict(input_features)[0]
probabilities = pipeline.predict_proba(input_features)[0] # Returns array of percentages

# Display final answer
st.subheader(f"Predicted Species: **Iris-{target_names[prediction]}**")

st.write("### 📊 Model Confidence Analytics")
# Convert probabilities into a clean dataframe for Streamlit charts
prob_df = pd.DataFrame({
    'Species': target_names,
    'Confidence (%)': [p * 100 for p in probabilities]
})

# Display a beautiful built-in native bar chart
st.bar_chart(data=prob_df, x='Species', y='Confidence (%)')