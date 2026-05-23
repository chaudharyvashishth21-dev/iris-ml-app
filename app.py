import streamlit as st
import joblib
import numpy as np

# 1. Load the frozen model we just created
model = joblib.load("iris_model.pkl")

# Map the model's numeric output (0, 1, 2) back to human-readable names
target_names = ['Setosa', 'Versicolor', 'Virginica']

# 2. Configure the look and feel of the web page
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Species Predictor")
st.write("created by [Vashishth kumar] - Powered by Streamlit and scikit-learn")
st.write("This web application uses a live Random Forest Machine Learning model to predict the species of an Iris flower based on its physical dimensions.")
st.write("---")

# 3. Create a clean user input interface using Sliders
st.header("🎛️ Adjust Flower Measurements (in cm)")

# Format: st.slider("Label", min_value, max_value, default_value)
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width  = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width  = st.slider("Petal Width", 0.1, 2.5, 0.2)

st.write("---")

# 4. Define what happens when the user clicks the "Predict" button
if st.button("🔮 Predict Flower Species", type="primary"):
    
    # Pack the inputs into a 2D array that the scikit-learn model expects
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Run the prediction
    prediction_numeric = model.predict(input_features)[0]
    final_species = target_names[prediction_numeric]
    
    # Display the result in a beautiful green success box
    st.success(f"### Result: The model predicts this flower is an **Iris-{final_species}**!")