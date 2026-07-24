import streamlit as st
import joblib

# ==========================
# Load Model & Encoders
# ==========================
model = joblib.load("model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# ==========================
# Page Settings
# ==========================
st.set_page_config(
    page_title="Fertilizer Recommendation Agent",
    page_icon="🌱",
    layout="wide"
)

# ==========================
# Title
# ==========================
st.title("🌱 Fertilizer Recommendation Agent")
st.write("Enter the soil and crop details to get the best fertilizer recommendation.")

st.divider()

# ==========================
# Input Columns
# ==========================
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡 Temperature", value=25)
    humidity = st.number_input("💧 Humidity", value=50)
    moisture = st.number_input("🌾 Moisture", value=40)
    nitrogen = st.number_input("🧪 Nitrogen", value=20)

with col2:
    potassium = st.number_input("🧪 Potassium", value=20)
    phosphorous = st.number_input("🧪 Phosphorous", value=20)

    soil = st.selectbox(
        "🌍 Soil Type",
        soil_encoder.classes_
    )

    crop = st.selectbox(
        "🌿 Crop Type",
        crop_encoder.classes_
    )
    # ==========================
# Prediction
# ==========================

if st.button("🌱 Recommend Fertilizer"):

    soil_value = soil_encoder.transform([soil])[0]
    crop_value = crop_encoder.transform([crop])[0]

    input_data = [[
        temperature,
        humidity,
        moisture,
        soil_value,
        crop_value,
        nitrogen,
        potassium,
        phosphorous
    ]]

    prediction = model.predict(input_data)

    fertilizer = fertilizer_encoder.inverse_transform(prediction)

    st.success(f"✅ Recommended Fertilizer: **{fertilizer[0]}**")