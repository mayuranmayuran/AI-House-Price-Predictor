import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration (Sets the title and styling)
st.set_page_config(page_title="AI House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 AI House Price Prediction Dashboard")
st.write("Enter the area metrics below to estimate the median house value instantly using our trained Random Forest model.")
st.markdown("---")

# 2. Load the trained machine learning model from Phase 2
@st.cache_resource
def load_model():
    with open("housing_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# 3. Create the Input Fields UI (Grouped into clean columns)
st.subheader("📊 Property & Location Characteristics")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in $10k blocks, e.g. 5.0 = $50,000)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
    house_age = st.slider("Median House Age (Years)", min_value=1, max_value=52, value=28)
    ave_rooms = st.number_input("Average Rooms per Dwelling", min_value=1.0, max_value=10.0, value=5.2, step=0.1)
    ave_bedrms = st.number_input("Average Bedrooms per Dwelling", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

with col2:
    population = st.number_input("Block Population", min_value=3, max_value=35000, value=1400, step=50)
    ave_occup = st.number_input("Average House Occupancy (People/Household)", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    latitude = st.number_input("Latitude Coordinate", min_value=32.5, max_value=42.5, value=35.6, step=0.01)
    longitude = st.number_input("Longitude Coordinate", min_value=-124.3, max_value=-114.3, value=-119.5, step=0.01)

st.markdown("---")

# 4. Handle the Prediction Action
if st.button("🔮 Calculate Estimated House Value", type="primary", use_container_width=True):
    # Format inputs exactly like the original training data features layout
    input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]], 
                              columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    
    # Run prediction
    prediction = model.predict(input_data)[0]
    
    # Convert from dataset base units ($100,000s) to standard dollars
    actual_price = prediction * 100000
    
    # Display Result beautifully
    st.success(f"### 🎉 Estimated Value: **${actual_price:,.2f}**")