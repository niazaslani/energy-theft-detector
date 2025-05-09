import pandas as pd

# Load the full dataset
df = pd.read_csv("CC_LCL-FullData.csv", parse_dates=["DateTime"])

# Clean column names (remove trailing spaces)
df.columns = df.columns.str.strip()

# Pick the first 10 unique users (you can change this number)
unique_users = df["LCLid"].unique()[:10]

# Filter data to only include these selected users
sample_df = df[df["LCLid"].isin(unique_users)]

# Save the sample to a new file for further processing
sample_df.to_csv("sample_households.csv", index=False)

print(f"Step 1 completed: Sample of {len(unique_users)} users saved to sample_households.csv")