import pandas as pd
import joblib

# Load new data (replace with your new CSV if different)
print("Loading new data...")
new_data = pd.read_csv("new_profiles.csv")

# Drop irrelevant or non-numeric columns
drop_cols = ["LCLid", "DateTime"] if "DateTime" in new_data.columns else ["LCLid"]
new_data_clean = new_data.drop(columns=drop_cols, errors="ignore")

# Select only numeric features
X_new = new_data_clean.select_dtypes(include="number")

# Load the trained model
print("Loading trained model...")
rf = joblib.load("rf_model.pkl")

# Predict
print("Making predictions...")
predictions = rf.predict(X_new)

# Add prediction to the original data
new_data["Theft_Prediction"] = predictions

# Save the results
new_data.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")