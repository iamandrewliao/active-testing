'''
Deep Ensemble: Ensemble of DNNs, each of which outputs mean and variance.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.layers(x)
        mean = self.mean_head(h)
        var = F.softplus(self.var_head(h)) + 1e-6
        return mean, var


def train_ensemble(models, X, y, epochs=500, lr=0.01):
    """
    Trains each model in the list independently.
    """
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in models]
    
    # Gaussian NLL Loss: L = 0.5 * (log(var) + (y - mean)^2 / var)
    loss_fn = nn.GaussianNLLLoss()
    
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            mean, var = model(X)
            # GaussianNLLLoss expects variance, not std dev
            loss = loss_fn(mean, y, var)
            loss.backward()
            optimizer.step()
        print(f"Model {i+1} finished. Final Loss: {loss.item():.4f}")


def predict_ensemble(models, X_test):
    """
    Combines predictions from multiple models.
    Returns ensemble mean and total uncertainty (standard deviation).
    """
    means_list = []
    vars_list = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            mean, var = model(X_test)
            means_list.append(mean)
            vars_list.append(var)
    
    # Shape: (Ensemble_Size, N_Samples, 1)
    means = torch.stack(means_list)
    vars = torch.stack(vars_list)
    
    # Combination
    # 1. Ensemble mean: avg of individual means
    # E[y] = (1/M) * sum(mu_m)
    ensemble_mean = means.mean(dim=0)
    # 2. Ensemble variance: 
    # Var(y) = (1/M) * sum(var_m + mu_m^2) - E[y]^2
    # This captures both aleatoric (avg data noise) and epistemic (disagreement between models) uncertainty.
    # Mean of variances (aleatoric)
    avg_var = vars.mean(dim=0)
    # Variance of means (epistemic)
    var_of_means = means.var(dim=0, unbiased=False)
    # Total variance
    total_var = avg_var + var_of_means
    # Return mean and standard deviation (sqrt of variance)
    return ensemble_mean, torch.sqrt(total_var)