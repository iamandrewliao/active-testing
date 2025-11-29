import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

def fit_density_estimator(csv_path):
    """
    Loads x,y data from a CSV and fits a Kernel Density Estimator.
    """
    print(f"\nLoading prior training distribution from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Take x and y columns
    train_data = df[['x', 'y']].values
    
    # gaussian_kde expects data shape (dims, num_points), so we transpose
    kde = gaussian_kde(train_data.T)
    print("Kernel Density Estimator fitted successfully.")
    return kde, train_data # Return both KDE and raw data for plotting


def plot_kde(kde, data, save_path='kde_plot.png'):
    """
    Plots the 2D Kernel Density Estimation using a contour map 
    and overlays the original data points, then saves the plot to a file.
    """
    # Define the bounds of the plot grid
    xmin = data[:, 0].min()
    xmax = data[:, 0].max()
    ymin = data[:, 1].min()
    ymax = data[:, 1].max()

    # Create a grid of points to evaluate the KDE at
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the KDE on the grid
    Z = np.reshape(kde(positions).T, X.shape)

    # Plotting
    plt.figure(figsize=(10, 8))
    # Plot original points
    plt.plot(data[:, 0], data[:, 1], 'o', color='red', markersize=3, 
             alpha=0.5, label='Original Data Points')
    # Plot the KDE as filled contours (density map)
    # Store the contour object returned by contourf 
    # and use its mapping for the colorbar.
    contour_set = plt.contourf(
        X, Y, Z, 
        cmap=cm.viridis, 
        norm=Normalize(vmin=Z.min(), vmax=Z.max())
    )
    # Optionally, plot contour lines for clarity
    plt.contour(X, Y, Z, colors='k', linewidths=0.5)
    # Pass the contour set object to the colorbar function
    plt.colorbar(contour_set, label='Estimated Density (PDF)')
    plt.title('Kernel Density Estimate')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--')
    plt.legend()

    # Save the figure
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
    print(f"Plot saved successfully to: {save_path}")


# --- Main execution block ---
csv_path = '/home/liao0241/initial_conditions/bottomleft_topright_initialconds.csv'
plot_save_path = 'visualizations/train_data_KDE.png'

# Fit the density estimator
kde_result, train_data = fit_density_estimator(csv_path)

if kde_result is not None:
    # Plot the KDE and the data
    # plot_kde(kde_result, train_data, save_path=plot_save_path)
    
    test_point = np.array([[.9], [.1]])
    density_at_point = kde_result(test_point)
    
    print("\n--- KDE Evaluation ---")
    print(f"Density at test point {test_point.T[0]}: {density_at_point[0]:.4f}")