import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import torch

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

def compute_knn_distance(target_points, reference_points, k=5):
    """
    Computes the distance from each point in target_points to its 
    k-th nearest neighbor in reference_points.
    
    Args:
        target_points: (N, 2) tensor of points to evaluate
        reference_points: (M, 2) tensor of training data
        k: Which neighbor to use (default 5 for smoother results)
        
    Returns:
        (N, 1) tensor of distances
    """
    # Compute pairwise distances: (N, M) matrix
    dists = torch.cdist(target_points, reference_points)
    
    # Get the k-th smallest distance for each target point
    # topk returns the smallest 'k' values when largest=False
    # values shape is (N, k)
    values, _ = torch.topk(dists, k=k, dim=1, largest=False)
    
    # The last column of values is the k-th nearest neighbor distance
    # shape (N, k), we want the k-th one (index k-1)
    kth_dist = values[:, k-1].unsqueeze(1) # Shape (N, 1)
    
    return kth_dist

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
    print(f"KDE plot saved successfully to: {save_path}")

def plot_knn_distance(reference_data, k, save_path='knn_distance_map.png'):
    """
    Plots the k-NN distance field across the 2D space.
    """
    print(f"\nComputing k-NN distance field (k={k})...")
    
    # 1. Define the grid and reference data
    xmin, xmax = reference_data[:, 0].min(), reference_data[:, 0].max()
    ymin, ymax = reference_data[:, 1].min(), reference_data[:, 1].max()

    # Create evaluation grid (NumPy)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    target_points_np = np.vstack([X.ravel(), Y.ravel()]).T # (N*N, 2)

    # Convert NumPy arrays to PyTorch Tensors
    target_points_torch = torch.from_numpy(target_points_np).float()
    reference_points_torch = torch.from_numpy(reference_data).float()
    
    # 2. Compute k-NN distance field
    distances_torch = compute_knn_distance(
        target_points_torch, 
        reference_points_torch, 
        k=k
    )
    # Convert distances back to NumPy and reshape to the grid
    Z_distances = distances_torch.numpy().reshape(X.shape)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    # Plotting the distance field. 
    # Since smaller distance means closer to data (higher "density"), we use 
    # a reversed color map (e.g., 'Reds_r') where dark means high density (low distance).
    # Use vmin and vmax for normalization.
    contour_set = plt.contourf(
        X, Y, Z_distances, 
        levels=20, 
        cmap=cm.Reds_r, # Reversed colormap
        norm=Normalize(vmin=Z_distances.min(), vmax=Z_distances.max())
    )
    # Plot original data points on top
    plt.plot(reference_data[:, 0], reference_data[:, 1], 'o', color='gray', markersize=3, 
             alpha=0.6, label='Reference Data Points')
    
    plt.contour(X, Y, Z_distances, colors='k', linewidths=0.5, alpha=0.5)
    plt.colorbar(contour_set, label=f'Distance to {k}-th Nearest Neighbor')
    plt.title(f'{k}-Nearest Neighbor (k-NN) Distance Field')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"KNN plot saved successfully to: {save_path}")

# --- Main execution block ---
csv_path = '/home/liao0241/initial_conditions/bottomleft_topright_initialconds.csv'
kde_save_path = '/home/liao0241/active_testing/visualizations/bottomleft_topright_KDE.png'
knn_K = 2
knn_save_path = f'/home/liao0241/active_testing/visualizations/bottomleft_topright_KNN_k{knn_K}.png'

# Fit the density estimator
kde_result, train_data = fit_density_estimator(csv_path)

if kde_result is not None:
    plot_kde(kde_result, train_data, save_path=kde_save_path)
    # plot_knn_distance(train_data, knn_K, save_path=knn_save_path)
    
    # test_point = np.array([[.9], [.1]])
    # density_at_point = kde_result(test_point)
    # print("\n--- KDE Evaluation ---")
    # print(f"Density at test point {test_point.T[0]}: {density_at_point[0]:.4f}")