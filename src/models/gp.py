import torch
import gpytorch
import numpy as np

# Your TanimotoKernel implementation
def batch_tanimoto_sim(x1: torch.Tensor, x2: torch.Tensor):
    """Tanimoto similarity between two batched tensors, across last 2 dimensions."""
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2 ** 2, dim=-1, keepdims=True)
    return (dot_prod) / (x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod)

class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto coefficient kernel."""

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        return batch_tanimoto_sim(x1, x2)

# Gaussian Process Model using the TanimotoKernel
class TanimotoGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TanimotoGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = TanimotoKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Wrapper class with .fit and .predict methods
class TanimotoGPRegressor:
    def __init__(self, uncertainty_method='variance', device='cpu'):
        self.uncertainty_method = uncertainty_method
        self.device = torch.device(device)
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

    def fit(self, train_x, train_y, training_iter=50, lr=0.1):

        # Convert NumPy arrays to torch tensors if needed
        if isinstance(train_x, np.ndarray):
            train_x = torch.from_numpy(train_x).float()
        if isinstance(train_y, np.ndarray):
            train_y = torch.from_numpy(train_y).float()

        # Ensure inputs are of appropriate dimensions
        if train_x.dim() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_y.dim() == 1:
            train_y = train_y.unsqueeze(-1)

        # Move data to the specified device
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        self.model = TanimotoGPModel(train_x, train_y.squeeze(), self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Loss function (marginal log likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y.squeeze())
            loss.backward()
            optimizer.step()
            # if (i + 1) % 10 == 0:
            #     print(f'Iteration {i + 1}/{training_iter} - Loss: {loss.item():.3f}')

    def predict(self, test_x):
        if isinstance(test_x, np.ndarray):
            test_x = torch.from_numpy(test_x).float()
        # Ensure inputs are of appropriate dimensions
        if test_x.dim() == 1:
            test_x = test_x.unsqueeze(-1)

        # Move test data to the device
        test_x = test_x.to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))
            mean = preds.mean
            lower, upper = preds.confidence_region() # = mean +- 2*sqrt(variance), interval 95% by default
            variance = preds.variance
            covariance = preds.covariance_matrix

            return {
                'mean': mean.cpu().numpy(),
                'lower': lower.cpu().numpy(),
                'upper': upper.cpu().numpy(),
                'variance': variance.cpu().numpy(),
                'covariance': covariance.cpu().numpy()
            }

    def predict_with_uncertainty(self, test_x):

        if self.uncertainty_method == 'variance':
            if isinstance(test_x, np.ndarray):
                test_x = torch.from_numpy(test_x).float()
            # Ensure inputs are of appropriate dimensions
            if test_x.dim() == 1:
                test_x = test_x.unsqueeze(-1)

            # Move test data to the device
            test_x = test_x.to(self.device)

            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = self.likelihood(self.model(test_x))
                mean = preds.mean
                lower, upper = preds.confidence_region() # = mean +- 2*sqrt(variance), interval 95% by default
                variance = preds.variance
                covariance = preds.covariance_matrix

                return mean.cpu().numpy(), variance.cpu().numpy()