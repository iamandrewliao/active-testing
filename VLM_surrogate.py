'''
A more advanced surrogate model that combines an vision-language encoder with a prediction head.
Input: scene and wrist camera images, language instruction
Output: continuous outcome prediction
(WORK IN PROGRESS)
'''

import torch
import numpy as np
import safetensors.torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# OpenPi Imports
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.model import Observation

# --- 1. The Pi0 Feature Extractor ---

class Pi0FeatureExtractor:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        
        # Initialize Config and Model
        # Note: You might need to adjust default config parameters if your checkpoint differs
        self.config = Pi0Config() 
        self.model = PI0Pytorch(self.config).to(device)
        self.model.eval()

        # Load Weights
        print(f"Loading Pi0 weights from {checkpoint_path}...")
        safetensors.torch.load_model(self.model, checkpoint_path)
        print("Pi0 model loaded successfully.")

    @torch.no_grad()
    def get_embeddings(self, images, tokenized_prompts):
        """
        Args:
            images: Tensor of shape [B, 3, H, W] (normalized -1 to 1)
            tokenized_prompts: Tensor of shape [B, SeqLen]
        Returns:
            embeddings: Tensor of shape [B, D]
        """
        # 1. Construct Mock Observation for Preprocessing
        # We need to create the specific structure PI0 expects
        B = images.shape[0]
        
        # Create masks (assuming full validity for inputs)
        image_mask = torch.ones(B, dtype=torch.bool, device=self.device)
        prompt_mask = torch.ones_like(tokenized_prompts, dtype=torch.bool, device=self.device)
        
        # PI0 expects specific keys
        obs_dict = {
            "images": {
                "base_0_rgb": images.to(self.device),
                "left_wrist_0_rgb": torch.zeros_like(images).to(self.device), # Dummy if not used
                "right_wrist_0_rgb": torch.zeros_like(images).to(self.device) # Dummy if not used
            },
            "image_masks": {
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": torch.zeros(B, dtype=torch.bool, device=self.device),
                "right_wrist_0_rgb": torch.zeros(B, dtype=torch.bool, device=self.device)
            },
            "state": torch.zeros((B, self.config.action_dim), device=self.device), # Dummy state
            "tokenized_prompt": tokenized_prompts.to(self.device),
            "tokenized_prompt_mask": prompt_mask
        }
        
        # 2. Call internal embedding methods directly
        # We use the internal helper to handle the list formatting PI0Pytorch expects
        # Note: We skip the full forward() because we only want the encoder (prefix) output
        
        # Prepare inputs exactly as PI0Pytorch._preprocess_observation expects, 
        # but since we manually built the dict, we can skip to embedding.
        
        # Embed Images & Text
        # Note: embed_prefix expects lists of tensors for images/masks
        img_list = [obs_dict["images"][k] for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]]
        mask_list = [obs_dict["image_masks"][k] for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]]
        
        prefix_embs, prefix_pad_masks, _ = self.model.embed_prefix(
            img_list, 
            mask_list, 
            obs_dict["tokenized_prompt"], 
            obs_dict["tokenized_prompt_mask"]
        )
        
        # prefix_embs: [B, SeqLen, D]
        # prefix_pad_masks: [B, SeqLen]
        
        # 3. Pooling Strategy
        # We compute the mean of the embeddings, ignoring padding
        mask_expanded = prefix_pad_masks.unsqueeze(-1).float() # [B, S, 1]
        sum_embeddings = torch.sum(prefix_embs * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        pooled_embeddings = sum_embeddings / sum_mask # [B, D]
        
        return pooled_embeddings


# 2. Prediction head: on top of the Pi0 encoder, there is a prediction head to predict outcomes and uncertainties
# Two options: a GP and an MLP (deep ensemble to get uncertainties)

# --- GP prediction head ---

def fit_pi0_gp_surrogate(train_embeddings, train_Y, bounds=None):
    """
    Fits a SingleTaskGP on top of the fixed Pi0 embeddings.
    
    Args:
        train_embeddings: [N, D] tensor of Pi0 features.
        train_Y: [N, 1] tensor of outcomes.
    """
    # Standardize outputs
    outcome_transform = Standardize(m=1)
    
    # We treat the embeddings as the "input features". 
    # Since embeddings are high-dim, we typically don't normalize them or use a linear kernel,
    # but RBF (default in SingleTaskGP) works well for "similarity" in latent space.
    
    model = SingleTaskGP(
        train_X=train_embeddings,
        train_Y=train_Y,
        outcome_transform=outcome_transform,
    )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    return model


# --- MLP prediction head ---

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. BoTorch-Compatible Ensemble Wrapper ---

class EnsemblePosterior(Posterior):
    """A custom Posterior that represents the distribution of the ensemble."""
    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

class Pi0EnsembleSurrogate(Model):
    def __init__(self, models, outcome_mean, outcome_std):
        """
        Args:
            models: List of trained PyTorch MLPs.
            outcome_mean, outcome_std: Statistics for un-normalizing predictions.
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.outcome_mean = outcome_mean
        self.outcome_std = outcome_std

    @property
    def num_outputs(self):
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        """
        Computes the mean and variance of the ensemble predictions.
        X shape: (batch_shape, q, d)
        Output shapes: (batch_shape, q, m)
        """
        self.eval()
        with torch.no_grad():
            # X comes in as [batch, q, d]
            preds = []
            for model in self.models:
                # Output: [batch, q, 1]
                pred = model(X)
                preds.append(pred)
            
            preds = torch.stack(preds, dim=0) # [ensemble_size, batch, q, 1]
            
            # 1. Calculate stats in normalized space
            mean_pred = preds.mean(dim=0)
            var_pred = preds.var(dim=0, unbiased=True)
            
            # If ensemble size is 1, variance is 0 (or NaN). Handle gracefully.
            if len(self.models) == 1:
                var_pred = torch.zeros_like(mean_pred)

            # 2. Un-standardize
            # Mean: mu_y + sigma_y * pred
            # Var: sigma_y^2 * var_pred
            
            final_mean = self.outcome_mean + self.outcome_std * mean_pred
            final_var = (self.outcome_std ** 2) * var_pred
            
            return EnsemblePosterior(final_mean, final_var)

# --- 4. Training Function ---

def fit_pi0_mlp_surrogate(train_embeddings, train_Y, ensemble_size=5, hidden_dim=256, epochs=100, lr=1e-3, device="cuda"):
    """
    Trains an ensemble of MLPs on the provided embeddings.
    """
    # 1. Normalize Targets
    outcome_mean = train_Y.mean()
    outcome_std = train_Y.std()
    if outcome_std < 1e-6: 
        outcome_std = torch.tensor(1.0) # Avoid division by zero
        
    train_Y_norm = (train_Y - outcome_mean) / outcome_std
    
    # Move to device
    X_data = train_embeddings.to(device)
    Y_data = train_Y_norm.to(device)
    
    dataset = TensorDataset(X_data, Y_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = train_embeddings.shape[-1]
    trained_models = []

    print(f"Training MLP Ensemble (Size={ensemble_size})...")
    
    for i in range(ensemble_size):
        model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_Y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        trained_models.append(model)
        
    return Pi0EnsembleSurrogate(trained_models, outcome_mean.to(device), outcome_std.to(device))