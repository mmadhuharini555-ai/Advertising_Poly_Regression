import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Advertising Sales Prediction", layout="centered")

st.title("📊 Advertising Sales Prediction")
st.write("Enter advertising budget to predict sales")

# Load poly + model
with open("advertising_poly_model.pkl", "rb") as file:
    poly, model = pickle.load(file)

# Inputs
tv = st.slider("📺 TV Budget", 0.0, 300.0, 50.0)
radio = st.slider("📻 Radio Budget", 0.0, 50.0, 10.0)
newspaper = st.slider("📰 Newspaper Budget", 0.0, 100.0, 20.0)

if st.button("🔮 Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    input_poly = poly.transform(input_data)   # ⭐ VERY IMPORTANT
    prediction = model.predict(input_poly)

    st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")
