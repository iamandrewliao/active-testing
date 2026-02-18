# one off script, IGNORE

import os
import pandas as pd

file_path = '/home/andrewliao/active-testing/results/pickblueblock_bruteforce/results_rightview.csv'

df = pd.read_csv(file_path)

if 'trial' in df.columns:
    df['trial'] = df['trial'] + 268
    
    # Overwrite the original file with updated values
    df.to_csv(file_path, index=False)
    print(f"Updated: {file_path}")