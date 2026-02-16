import os
import pandas as pd

file_path = '/home/andrewliao/active-testing/results/pickblueblock_bruteforce/results_rightview.csv'  # Change this to your target directory


df = pd.read_csv(file_path)

# Check if 'trial' exists in this CSV to avoid errors
if 'trial' in df.columns:
    df['trial'] = df['trial'] + 268
    
    # Overwrite the original file with updated values
    df.to_csv(file_path, index=False)
    print(f"Updated: {file_path}")