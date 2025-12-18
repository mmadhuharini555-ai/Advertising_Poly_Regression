import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Advertising Sales Predictor",
    page_icon="📊",
    layout="centered"
)

# Load model
with open("advertising_poly_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.markdown("<h1 style='text-align:center;'>📊 Advertising Sales Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict product sales based on advertising budget</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
st.sidebar.header("📥 Enter Advertising Budget")
tv = st.sidebar.slider("TV Budget", 0.0, 300.0, 50.0)
radio = st.sidebar.slider("Radio Budget", 0.0, 50.0, 10.0)
newspaper = st.sidebar.slider("Newspaper Budget", 0.0, 120.0, 20.0)

# Main section
st.subheader("📌 Selected Inputs")
input_df = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper]
})

st.dataframe(input_df, use_container_width=True)

# Prediction
if st.button("🚀 Predict Sales"):
    input_array = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_array)

    st.success(f"📈 **Predicted Sales:** {prediction[0]:.2f}")

    st.divider()

    # Bar chart
    st.subheader("📊 Advertising Budget Distribution")
    chart_df = pd.DataFrame({
        "Medium": ["TV", "Radio", "Newspaper"],
        "Budget": [tv, radio, newspaper]
    })

    st.bar_chart(chart_df.set_index("Medium"))

    st.info(
        "ℹ️ Higher advertising budget usually leads to higher sales, "
        "but the impact depends on the medium."
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
