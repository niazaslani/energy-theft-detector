import pandas as pd

# Load the cleaned dataset from step 2
df = pd.read_csv("cleaned_profiles.csv", parse_dates=["DateTime"])

# Clean column names
df.columns = df.columns.str.strip()

# Convert energy column to float
df["KWH/hh (per half hour)"] = pd.to_numeric(df["KWH/hh (per half hour)"], errors="coerce")

# Add 'Theft' column initialized to 0
df["Theft"] = 0

# Pick one household ID to simulate theft
target_id = df["LCLid"].unique()[0]

# Define theft time period
start_date = "2013-07-01"
end_date = "2013-07-15"

# Create a mask for the target household and date range
mask = (
    (df["LCLid"] == target_id) &
    (df["DateTime"] >= start_date) &
    (df["DateTime"] <= end_date)
)

# Simulate theft by reducing energy usage
df.loc[mask, "KWH/hh (per half hour)"] *= 0.2
df.loc[mask, "Theft"] = 1

# Save to new labeled dataset
df.to_csv("labeled_dataset.csv", index=False)

print(f"Step 3 completed: Theft simulated for household {target_id} from {start_date} to {end_date}.")