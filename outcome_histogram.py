'''
Quick visualization of continuous outcome as a histogram.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# Change as needed
filepath = 'results/pAll_bf_95_results.csv'
savepath = 'visualizations/robo_eval/pAll_outcome_histogram.png'

try:
    df = pd.read_csv(filepath)
except FileNotFoundError:
    print(f"Error: The file {filepath} was not found. Please check the file name and path.")
    exit()

# Create the histogram with centered bars
column_name = 'continuous_outcome'
increment = 0.5
min_outcome = 0.0
max_outcome = 4.0

# Calculate the number of desired ticks/bins centers:
# (Max - Min) / Increment + 1 = (4.0 - 0.0) / 0.5 + 1 = 9 points
num_points = int((max_outcome - min_outcome) / increment) + 1

# Define the desired tick locations using np.linspace (guaranteed inclusion of endpoints)
tick_locations = np.linspace(min_outcome, max_outcome, num_points)
tick_labels = [str(x) for x in tick_locations]

# Define bin edges: 1 more edge than the number of points (9 points -> 10 edges)
# The edges must be centered relative to the points: from min - (inc/2) to max + (inc/2)
min_edge = min_outcome - (increment / 2)  # -0.25
max_edge = max_outcome + (increment / 2)  # 4.25
bins = np.linspace(min_edge, max_edge, num_points + 1)


plt.figure(figsize=(10, 6))

if column_name in df.columns:
    # Plot the histogram using the clean bins
    plt.hist(df[column_name].dropna(), bins=bins, edgecolor='black', alpha=0.7, rwidth=0.9)

    # Set the custom x-axis ticks and labels
    plt.xticks(ticks=tick_locations, labels=tick_labels)

    # Set titles and labels
    plt.title(f'Histogram of {column_name} from {filepath}', fontsize=16)
    plt.xlabel(column_name, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.5)

    # Save the plot to a file
    plt.savefig(savepath)
    plt.close()
    print(f"\nSuccessfully generated histogram and saved as {savepath}")
else:
    print(f"\nError: Column '{column_name}' not found.")