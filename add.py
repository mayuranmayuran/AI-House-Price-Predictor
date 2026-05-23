import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("📦 Loading California Housing Data...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🛠️ OPTIMIZED FOR GITHUB: Shrinking tree depth to bring file size below 25MB
print("🤖 Training an optimized, lightweight Random Forest...")
model = RandomForestRegressor(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Training complete!")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=====================================")
print("🎯 NEW MODEL PERFORMANCE METRICS")
print("=====================================")
print(f"• Mean Absolute Error (MAE): ${mae * 100000:.2f}")
print(f"• R2 Score (Accuracy):        {r2 * 100:.2f}%")
print("=====================================")

with open("housing_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\n💾 New lightweight model saved successfully as 'housing_model.pkl'!")