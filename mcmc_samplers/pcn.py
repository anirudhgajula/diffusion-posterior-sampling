import torch
import numpy as np
from .base import MCMCSampler

class PCNSampler(MCMCSampler):
    """Pre-conditioned Crank-Nicolson (pCN) MCMC sampler.
    
    This sampler is particularly efficient for linear inverse problems with Gaussian prior.
    It can handle all linear measurement operators:
    - Super-resolution
    - Gaussian deblurring
    - Motion deblurring
    - Inpainting
    """
    
    def __init__(self, measurement_op, noise_model, prior_model, beta=0.1):
        """
        Args:
            measurement_op: Forward measurement operator
            noise_model: Noise model (e.g., Gaussian, Poisson)
            prior_model: Prior distribution model
            beta: Step size parameter (0 < beta < 1)
        """
        super().__init__(measurement_op, noise_model, prior_model)
        self.beta = torch.tensor(beta)
        
        # Verify that we have a linear operator
        if not hasattr(measurement_op, 'transpose'):
            raise ValueError("PCN sampler requires a linear measurement operator with transpose method")
    
    def log_likelihood(self, x, y, **kwargs):
        """Compute log likelihood p(y|x).
        
        For Gaussian noise: -1/(2σ²) ||y - Ax||²
        """
        forward_pred = self.measurement_op.forward(x, **kwargs)
        if isinstance(self.noise_model, dict) and self.noise_model['name'] == 'gaussian':
            sigma = self.noise_model['sigma']
            return -0.5 * torch.sum((y - forward_pred)**2) / (sigma**2)
        else:
            raise NotImplementedError("Only Gaussian noise model is implemented")
    
    def log_prior(self, x):
        """Compute log prior p(x).
        
        For Gaussian prior: -1/2 ||x||²
        """
        return -0.5 * torch.sum(x**2)
    
    def propose(self, x_current):
        """Generate proposal using pCN scheme:
        x' = √(1-β²)x + βw, where w ~ N(0,I)
        """
        w = torch.randn_like(x_current)
        sqrt_factor = torch.sqrt(1 - self.beta**2).to(x_current.device)
        beta = self.beta.to(x_current.device)
        return sqrt_factor * x_current + beta * w
    
    def acceptance_probability(self, x_proposed, x_current, y, **kwargs):
        """Compute acceptance probability.
        
        For pCN: min(1, exp(Φ(x) - Φ(x')))
        where Φ(x) = -log p(y|x) is the negative log likelihood
        """
        current_likelihood = self.log_likelihood(x_current, y, **kwargs)
        proposed_likelihood = self.log_likelihood(x_proposed, y, **kwargs)
        
        # Note: Prior terms cancel out in pCN
        log_alpha = proposed_likelihood - current_likelihood
        return torch.min(torch.tensor(1.0).to(x_current.device), torch.exp(log_alpha)) 