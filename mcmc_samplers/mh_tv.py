import torch
import torch.nn.functional as F
from .base import MCMCSampler

class MHTVSampler(MCMCSampler):
    """Metropolis-Hastings sampler with Total Variation prior.
    
    This sampler can handle both linear and non-linear inverse problems:
    Linear:
    - Super-resolution
    - Gaussian deblurring
    - Motion deblurring
    - Inpainting
    
    Non-linear:
    - Non-linear deblurring
    - Phase retrieval
    """
    
    def __init__(self, measurement_op, noise_model, prior_model, step_size=0.1, tv_lambda=0.1):
        """
        Args:
            measurement_op: Forward measurement operator
            noise_model: Noise model (e.g., Gaussian, Poisson)
            prior_model: Prior distribution model
            step_size: Step size for random walk proposals
            tv_lambda: Weight of TV prior
        """
        super().__init__(measurement_op, noise_model, prior_model)
        self.step_size = step_size
        self.tv_lambda = tv_lambda
    
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
    
    def total_variation(self, x):
        """Compute total variation of image x.
        
        For color images, computes TV separately for each channel.
        """
        if x.dim() == 4:  # batch x channels x height x width
            # Compute horizontal and vertical differences
            diff_h = torch.abs(x[..., :, 1:] - x[..., :, :-1])
            diff_v = torch.abs(x[..., 1:, :] - x[..., :-1, :])
            
            # Sum over spatial dimensions and channels
            return torch.sum(diff_h) + torch.sum(diff_v)
        else:
            raise ValueError("Input must be a 4D tensor [batch x channels x height x width]")
    
    def log_prior(self, x):
        """Compute log prior p(x).
        
        TV prior: -λ * TV(x)
        """
        return -self.tv_lambda * self.total_variation(x)
    
    def propose(self, x_current):
        """Generate proposal using random walk:
        x' = x + εw, where w ~ N(0,I)
        
        For better mixing, we use a multi-scale proposal:
        - 80% of the time: local random walk
        - 20% of the time: global random walk
        """
        if torch.rand(1) < 0.8:
            # Local random walk
            w = torch.randn_like(x_current)
            return x_current + self.step_size * w
        else:
            # Global random walk (larger step size)
            w = torch.randn_like(x_current)
            return x_current + (5 * self.step_size) * w
    
    def acceptance_probability(self, x_proposed, x_current, y, **kwargs):
        """Compute acceptance probability.
        
        α = min(1, p(y|x')p(x')/p(y|x)p(x))
        """
        # Compute log posterior ratio
        log_ratio = (self.log_likelihood(x_proposed, y, **kwargs) + self.log_prior(x_proposed)) - \
                   (self.log_likelihood(x_current, y, **kwargs) + self.log_prior(x_current))
                   
        return torch.min(torch.tensor(1.0).to(x_current.device), torch.exp(log_ratio)) 