import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load labeled dataset
print("Loading labeled dataset...")
df = pd.read_csv("labeled_dataset.csv")

# Drop non-numeric or ID columns
df = df.drop(columns=["LCLid", "DateTime"], errors="ignore")

# Check which columns are numeric
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Show warning if 'Theft' column is missing
if "Theft" not in numeric_cols:
    raise ValueError("Theft column not found or not numeric.")

# Split features and target
y = df["Theft"]
X = df[numeric_cols].drop(columns=["Theft"])

# Drop rows with missing values
initial_len = len(X)
X = X.dropna()
y = y.loc[X.index]
print(f"Dropped {initial_len - len(X)} rows due to missing values.")

# Stop if dataset is still empty
if X.empty:
    raise ValueError("No data left after cleaning. Check your labeled_dataset.csv for valid numeric features.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest model...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, "theft_model.pkl")
print("Model saved to theft_model.pkl")