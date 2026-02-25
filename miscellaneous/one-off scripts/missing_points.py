# ONE-OFF SCRIPT; IGNORE

import torch
import pandas as pd
from factors_config import get_design_points_robot, is_valid_point
import os

save_dir = 'results/uprightcup_bruteforce'
csv_path = os.path.join(save_dir, 'results.csv')
cleaned_csv_path = os.path.join(save_dir, 'results_cleaned.csv')
missing_csv_path = os.path.join(save_dir, 'missing_from_results.csv')

# --- 1. Generate Current Valid Points ---
task_name = 'uprightcup'
all_points = get_design_points_robot()
valid_points_list = [p for p in all_points if is_valid_point(p, task_name=task_name)]

if not valid_points_list:
    print("Error: No valid points found.")
    exit()

# Store valid points in a set for O(1) lookup speed
# We round to 4 decimal places to handle floating point noise
points_tensor = torch.stack(valid_points_list)[:, :3]
valid_set = {tuple(round(val, 4) for val in row) for row in points_tensor.tolist()}

# --- 2. Process the CSV ---
df = pd.read_csv(csv_path)

# Create a boolean mask to identify valid rows
# We round the CSV values on the fly to match the set format
def check_row_validity(row):
    point = (round(row['x'], 4), round(row['y'], 4), round(row['table_height'], 4))
    return point in valid_set

# Split the dataframe into valid and invalid
is_valid_mask = df.apply(check_row_validity, axis=1)
df_cleaned = df[is_valid_mask]
df_invalid = df[~is_valid_mask]

# --- 3. Output and Save ---
print(f"Original Rows: {len(df)}")
print(f"Valid Rows:    {len(df_cleaned)}")
print(f"Removed Rows:  {len(df_invalid)}")

# Save the cleaned data
df_cleaned.to_csv(cleaned_csv_path, index=False)
print(f"\n✅ Cleaned dataset saved as {cleaned_csv_path}")

# --- 4. Export Missing Points (The To-Do List) ---
# Find points in code but NOT in the original CSV
csv_points_set = {tuple(round(val, 4) for val in row) for row in df[['x', 'y', 'table_height']].values}
missing_from_csv = valid_set - csv_points_set

if missing_from_csv:
    pd.DataFrame(list(missing_from_csv), columns=['x', 'y', 'table_height']).to_csv(missing_csv_path, index=False)
    print(f"✅ Exported {len(missing_from_csv)} missing points to {missing_csv_path}")