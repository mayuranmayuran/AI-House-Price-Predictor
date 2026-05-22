import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(page_title="AI House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 AI House Price Prediction Dashboard")
st.write("Enter the area metrics below to estimate the median house value instantly.")
st.markdown("---")

# 🤖 AUTOMATED CLOUD TRAINER (Bypasses pickle version errors)
@st.cache_resource
def get_trained_model():
    # Load the framework data directly on the cloud server
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    # Train the exact optimized lightweight model matching the server environment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Initialize our model seamlessly
with st.spinner("🤖 Initializing AI Model in the cloud... Please wait..."):
    model = get_trained_model()

# 2. User Input Elements Layout
st.header("📋 Enter Property Characteristics")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in $10,000s)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
    house_age = st.number_input("House Age (Years)", min_value=1.0, max_value=52.0, value=28.0, step=1.0)
    ave_rooms = st.number_input("Average Rooms per Dwelling", min_value=1.0, max_value=10.0, value=5.4, step=0.1)
    ave_bedrms = st.number_input("Average Bedrooms per Dwelling", min_value=0.5, max_value=5.0, value=1.1, step=0.1)

with col2:
    population = st.number_input("Neighborhood Population", min_value=3.0, max_value=35000.0, value=1425.0, step=10.0)
    ave_occup = st.number_input("Average Household Occupants", min_value=1.0, max_value=6.0, value=3.0, step=0.1)
    latitude = st.number_input("Property Latitude (°N)", min_value=32.5, max_value=42.5, value=35.6, step=0.1)
    longitude = st.number_input("Property Longitude (°W)", min_value=-124.5, max_value=-114.3, value=-119.5, step=0.1)

# 3. Handle Calculations
if st.button("Calculate Estimated House Value", type="primary"):
    # Format inputs exactly like original training metrics layout
    input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]],
                              columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    
    # Generate live inference
    prediction = model.predict(input_data)[0]
    actual_price = prediction * 100000
    
    st.markdown("### 🎯 Prediction Analysis")
    st.success(f"### 🏡 Estimated Value: **${actual_price:,.2f}**")