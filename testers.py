'''
Different testing strategies
-Active testing (surrogate model + acquisition function)
-IID testing (uniform random sampling)

BoTorch sources:
https://botorch.org/docs/overview
https://botorch.readthedocs.io/en/latest/index.html
'''
import pandas as pd
import numpy as np
import torch
import time

from utils import fit_surrogate_model, get_acquisition_function, optimize_acq_func, load_vla_data, compute_knn_distance, compute_kde_density

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cuda" if torch.cuda.is_available() else "cpu"}

class ActiveTester:
    def __init__(self, initial_X, initial_Y, bounds, grid_points, mc_points, model_name, acq_func_name, vla_data_path=None, ood_metric="knn"):
        self.train_X = initial_X
        self.train_Y = initial_Y
        self.bounds = bounds
        self.grid_points = grid_points
        self.model_name = model_name
        self.acq_func_name = acq_func_name
        self.model = None
        self.acq_func = None
        if mc_points is None:
            self.mc_points = self.grid_points # integrate over entire design space (no sampling)
        else:
            self.mc_points = mc_points
        self.available_design_space = self.grid_points.clone()
        self.vla_points = load_vla_data(vla_data_path) # contains factor values of the training data as columns
        if self.vla_points is not None:
            self.vla_points = self.vla_points.to(**tkwargs)
        self.ood_metric = ood_metric # likelihood of drawing evaluation factor combo f from the training data distribution
    
    def add_feature(self, points):
        """
        Manually adds the OOD feature (distance or density) to a set of points.
        """
        if self.vla_points is None:
            return points
        if self.ood_metric == "knn":
            # Calculate distance to nearest VLA neighbor
            feature = compute_knn_distance(points, self.vla_points, k=1)
        elif self.ood_metric == "kde":
            # Calculate probability density
            feature = compute_kde_density(points, self.vla_points)
        else:
            # Fallback or unknown metric, return unaugmented
            return points
        # Concatenate [points, feature]
        return torch.cat([points, feature], dim=-1)

    def get_next_point(self):
        """
        Fits the surrogate model and optimizes the acquisition function to find the
        next best point to sample.
        """
        print(f"Fitting surrogate model {self.model_name}")
        start_time = time.time()

        # Augment the evaluation points (train_X) with the OOD feature
        train_X_final = self.train_X
        bounds_final = self.bounds
        if self.vla_points is not None:
            train_X_final = self.add_feature(self.train_X)
            # We need to update bounds for the feature.
            # We can calculate the feature on the grid to find the max range.
            with torch.no_grad():
                grid_features = self.add_feature(self.grid_points)[:, -1] # get just the feature col
                max_val = grid_features.max().item()
                min_val = grid_features.min().item()
            # Bounds: x=[0,1], y=[0,1], feature=[min, max*buffer]
            # Expanding slightly helps avoid edge effects in optimization
            # buffer = 1.1
            # feature_bounds = torch.tensor([[min_val], [max_val * buffer]], **tkwargs)
            feature_bounds = torch.tensor([[min_val], [max_val]], **tkwargs)
            bounds_final = torch.cat([self.bounds, feature_bounds], dim=1)
        else:
            train_X_final = self.train_X
            bounds_final = self.bounds

        self.model = fit_surrogate_model(train_X_final, self.train_Y, bounds_final, model_name=self.model_name)

        print(f"Optimizing acquisition function {self.acq_func_name}")
        # We must also augment the design space so it matches the model's expected input size
        design_space_input = self.available_design_space
        mc_points_input = self.mc_points

        if self.vla_points is not None:
            design_space_input = self._augment_data(self.available_design_space)
            if self.mc_points is not None:
                mc_points_input = self._augment_data(self.mc_points)

        self.acq_func = get_acquisition_function(model=self.model, acq_func_name=self.acq_func_name, mc_points=mc_points_input)
        acquired_point_aug = optimize_acq_func(acq_func=self.acq_func, design_space=design_space_input, discrete=True, normalized_bounds=None)
        acquired_point = acquired_point_aug[:2] # just return the factor values
        
        end_time = time.time()
        print(f"Active sample selection took {end_time - start_time:.2f} seconds.")

        # Remove the acquired point from the design space
        self.available_design_space = self.available_design_space[~torch.all(self.available_design_space == acquired_point, dim=1)]
        return acquired_point

    def update(self, new_x, new_y):
        """Adds a new data point to the training set."""
        self.train_X = torch.cat([self.train_X, new_x.unsqueeze(0)])
        self.train_Y = torch.cat([self.train_Y, new_y.unsqueeze(0)])


class IIDSampler:
    """
    A sampler that samples *with replacement* from a discrete grid of points.
    """
    def __init__(self, grid_points):
        self.grid_points = grid_points
        self.num_points = self.grid_points.shape[0]
        # print(f"Initialized IIDSampler with {self.num_points} discrete points.")

    def get_next_point(self):
        """Generates a new point by sampling uniformly from the grid."""
        # Randomly select an index
        idx = torch.randint(low=0, high=self.num_points, size=(1,)).item()
        point = self.grid_points[idx]
        return point
    
    def update(self, new_x, new_y):
        """Does nothing."""
        pass


class ListIteratorSampler:
    """
    A 'sampler' that iterates through a provided list of points.
    The list can come from a filepath (for 'loaded' mode)
    or a pre-computed tensor (for 'brute_force' mode).
    """
    def __init__(self, source):
        if isinstance(source, str):
            # --- This is the 'loaded' mode logic (from PointLoader) ---
            filepath = source
            print(f"Loading evaluation points from {filepath}...")
            try:
                df = pd.read_csv(filepath)
                if not {'x', 'y'}.issubset(df.columns):
                    raise ValueError("CSV file must contain 'x' and 'y' columns.")
                
                # Convert directly to a single [N, 2] tensor
                points_np = df[['x', 'y']].values
                self.points = torch.tensor(points_np, **tkwargs)
                
            except FileNotFoundError:
                print(f"Error: The file '{filepath}' was not found.")
                exit()
            except Exception as e:
                print(f"Error reading the file: {e}")
                exit()
            print(f"Successfully loaded {self.points.shape[0]} points.")
            
        elif isinstance(source, torch.Tensor):
            # --- This is the 'brute_force' mode logic ---
            self.points = source
            # print(f"Initialized ListIteratorSampler with {self.points.shape[0]} points.")
        
        else:
            raise TypeError(f"ListIteratorSampler must be initialized with a filepath (str) or a tensor, not {type(source)}")
        
        self.index = 0
        self.num_points = self.points.shape[0]


    def get_next_point(self):
        """Returns the next point from the pre-loaded list."""
        if self.index >= self.num_points:
            print("Error: (ListIteratorSampler) Ran out of points to evaluate.")
            return None 
        
        point = self.points[self.index]
        self.index += 1
        return point

    def update(self, new_x, new_y):
        """Does nothing."""
        pass
