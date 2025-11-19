'''
Different testing strategies
-Active testing (surrogate model + acquisition function)
-IID testing (uniform random sampling)

BoTorch sources:
https://botorch.org/docs/overview
https://botorch.readthedocs.io/en/latest/index.html
'''
import pandas as pd
import torch
import time

from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from utils import fit_surrogate_model, get_acquisition_function, optimize_acq_func

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cuda" if torch.cuda.is_available() else "cpu"}

class ActiveTester:
    def __init__(self, initial_X, initial_Y, bounds, grid_points, mc_points, model_name, acq_func_name):
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

    def get_next_point(self):
        """
        Fits the surrogate model and optimizes the acquisition function to find the
        next best point to sample.
        """
        print(f"Fitting surrogate model {self.model_name}")
        start_time = time.time()
        self.model = fit_surrogate_model(self.train_X, self.train_Y, self.bounds, model_name=self.model_name)

        print(f"Optimizing acquisition function {self.acq_func_name}")
        self.acq_func = get_acquisition_function(model=self.model, acq_func_name=self.acq_func_name, mc_points=self.mc_points)
        acquired_point = optimize_acq_func(acq_func=self.acq_func, design_space=self.available_design_space, discrete=True, normalized_bounds=None)
        
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
