import pandas as pd

# Load the selected user data from Step 1
df = pd.read_csv("sample_households.csv", parse_dates=["DateTime"])

# Clean column names: remove extra spaces
df.columns = df.columns.str.strip()

# Convert energy usage to numeric (handle non-numeric values safely)
df["KWH/hh (per half hour)"] = pd.to_numeric(df["KWH/hh (per half hour)"], errors="coerce")

# Drop rows with missing energy values
df = df.dropna(subset=["KWH/hh (per half hour)"])

# Optionally, drop rows with negative or zero usage if you want to clean anomalies
df = df[df["KWH/hh (per half hour)"] > 0]

# Save cleaned data for next step
df.to_csv("cleaned_profiles.csv", index=False)

print("Step 2 completed: Cleaned data saved to cleaned_profiles.csv")