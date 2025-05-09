import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load balanced training data
data = pd.read_csv("balanced_training_data.csv")

# Features and target
X = data[["KWH/hh (per half hour)"]]
y = data["Theft"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "rf_model.pkl")
print("✅ Model trained and saved as rf_model.pkl")