# ONE-OFF SCRIPT; IGNORE
import pandas as pd

# Load your cleaned results
csv_path = './results/uprightcup_bruteforce/results_cleaned.csv'
df = pd.read_csv(csv_path)

# Overwrite the 'trial' column with consecutive numbers starting from 1
df['trial'] = range(1, len(df) + 1)

# Save it back to the file
# Set index=False so pandas doesn't add an extra 'unnamed' column
df.to_csv('results_final.csv', index=False)

print(f"✅ Re-indexed {len(df)} trials.")
print("The 'trial' column now follows a perfect 1 -> 2 -> ... sequence.")

# Quick verification
print("\nFirst 5 rows of updated trials:")
print(df[['trial', 'x', 'y', 'table_height']].head())