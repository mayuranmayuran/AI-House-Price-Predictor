import pandas as pd
import numpy as np
import pickle # Used to save our trained model to a file
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("📦 Loading California Housing Data...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 1. Separate Features (X) and Target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# 2. Split data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training with {X_train.shape[0]} rows, testing with {X_test.shape[0]} rows.")

# 3. Initialize and Train the Random Forest Model
print("🤖 Training the Random Forest Regressor (This might take 10-20 seconds)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Training complete!")

# 4. Evaluate the Model's Accuracy
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=====================================")
print("🎯 MODEL PERFORMANCE METRICS")
print("=====================================")
print(f"• Mean Absolute Error (MAE): ${mae * 100000:.2f}")
print(f"• R2 Score (Accuracy):        {r2 * 100:.2f}%")
print("=====================================")

# 5. Export and Save the trained model file for our UI later
with open("housing_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\n💾 Model saved successfully as 'housing_model.pkl'!")