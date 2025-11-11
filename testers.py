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

from botorch.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from utils import fit_surrogate_model

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cpu"}

class ActiveTester:
    def __init__(self, initial_X, initial_Y, bounds, grid_points, mc_points=None):
        self.train_X = initial_X
        self.train_Y = initial_Y
        self.bounds = bounds
        self.grid_points = grid_points
        self.model = None
        if mc_points is None:
            self.mc_points = self.grid_points
        else:
            self.mc_points = mc_points
        self.acq_func = None

    def get_next_point(self):
        """
        Fits the model and optimizes the acquisition function to find the
        next best point to sample.
        """
        print("Fitting surrogate model")
        start_time = time.time()
        self.model = fit_surrogate_model(self.train_X, self.train_Y, self.bounds, model_name="SaasFullyBayesianSingleTaskGP")
        # self.model = fit_surrogate_model(self.train_X, self.train_Y, self.bounds, model_name="SingleTaskGP")

        print("Optimizing acquisition function")
        # Construct the acquisition function
        self.acq_func = qBayesianActiveLearningByDisagreement(model=self.model)
        # self.acq_func = qNegIntegratedPosteriorVariance(
        #     model=self.model, 
        #     mc_points=self.mc_points
        # )

        # Optimize the acquisition function (continuous and discrete versions)
        # there is also a mixed version: optimize_acqf_mixed()
        # candidate, _ = optimize_acqf(
        #     acq_function=jes,
        #     bounds=normalized_bounds,
        #     q=1,
        #     num_restarts=4,
        #     raw_samples=256,
        # )
        candidate, _ = optimize_acqf_discrete(
            acq_function=self.acq_func,
            q=1,
            choices=self.grid_points,
        )
        
        point = candidate.squeeze(0) # Return a 1D tensor
        
        end_time = time.time()
        print(f"Active sample selection took {end_time - start_time:.2f} seconds.")

        return point

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
