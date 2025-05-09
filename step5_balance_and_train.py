import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading labeled dataset...")
df = pd.read_csv("labeled_dataset.csv")

# Drop irrelevant or non-numeric columns
drop_cols = ["LCLid", "DateTime"] if "DateTime" in df.columns else ["LCLid"]
df = df.drop(columns=drop_cols, errors='ignore')

# Ensure "Theft" column exists
if "Theft" not in df.columns:
    raise ValueError("Missing 'Theft' column in dataset.")

# Separate classes
df_majority = df[df.Theft == 0]
df_minority = df[df.Theft == 1]

# Upsample the minority class
print("Balancing dataset with upsampling...")
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Separate features and label
X = df_balanced.drop(columns=["Theft"]).select_dtypes(include="number")
y = df_balanced["Theft"]

# Check if the feature matrix is valid
if X.empty or y.empty:
    raise ValueError("No numeric features found or labels are empty.")

print("Training balanced Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Save model
joblib.dump(rf, "rf_model.pkl")
print("Model saved to rf_model.pkl")