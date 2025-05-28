import os
import glob
import pandas as pd

# Set the root directory containing all test results
root_dir = "/home/grace/Documents/ce340-lars-semantic/output/test"

# Find all test_summary.csv files recursively
csv_files = glob.glob(os.path.join(root_dir, "**/test_summary.csv"), recursive=True)

# Read and concatenate all CSVs
df_list = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Optionally add a column for model version/folder
    df["ModelFolder"] = os.path.basename(os.path.dirname(csv_file))
    df_list.append(df)

# Combine all into one DataFrame
summary_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV
summary_df.to_csv(os.path.join(root_dir, "all_test_summary.csv"), index=False)
print(f"Combined summary saved to {os.path.join(root_dir, 'all_test_summary.csv')}")