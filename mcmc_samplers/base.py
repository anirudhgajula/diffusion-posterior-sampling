import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import os

class MCMCSampler(ABC):
    """Base class for MCMC samplers."""
    
    def __init__(self, measurement_op, noise_model, prior_model):
        """
        Args:
            measurement_op: Forward measurement operator (from guided_diffusion.measurements)
            noise_model: Noise model (e.g., Gaussian, Poisson)
            prior_model: Prior distribution model
        """
        self.measurement_op = measurement_op
        self.noise_model = noise_model
        self.prior_model = prior_model
        
    def log_likelihood(self, x, y, **kwargs):
        """Compute log likelihood p(y|x).
        
        Args:
            x: Current state/image
            y: Measurement/observation
            **kwargs: Additional arguments for measurement operator
        """
        forward_pred = self.measurement_op.forward(x, **kwargs)
        
        if isinstance(self.noise_model, dict):
            if self.noise_model['name'] == 'gaussian':
                sigma = self.noise_model['sigma']
                return -0.5 * torch.sum((y - forward_pred)**2) / (sigma**2)
            elif self.noise_model['name'] == 'poisson':
                # For Poisson noise, we use a Gaussian approximation
                # The variance at each point is equal to the mean
                diff = y - forward_pred
                variance = torch.abs(forward_pred) + 1e-8  # Add small constant for stability
                return -0.5 * torch.sum(diff**2 / variance)
        
        raise NotImplementedError(f"Noise model {self.noise_model['name']} not implemented")
    
    @abstractmethod
    def log_prior(self, x):
        """Compute log prior p(x)."""
        pass
    
    def log_posterior(self, x, y, **kwargs):
        """Compute log posterior p(x|y) ∝ p(y|x)p(x)."""
        return self.log_likelihood(x, y, **kwargs) + self.log_prior(x)
    
    @abstractmethod
    def propose(self, x_current):
        """Generate proposal for next state."""
        pass
    
    @abstractmethod
    def acceptance_probability(self, x_proposed, x_current, y, **kwargs):
        """Compute acceptance probability for proposed state."""
        pass
    
    def sample(self, y, x_init, n_steps, record=False, save_root=None, **kwargs):
        """Run MCMC sampling.
        
        Args:
            y: Measurement/observation
            x_init: Initial state
            n_steps: Number of MCMC steps
            record: Whether to save intermediate states
            save_root: Directory to save intermediate states
            **kwargs: Additional arguments for measurement operator
            
        Returns:
            Final sample and acceptance rate
        """
        x_current = x_init
        accepted = 0
        
        # Create progress directory if needed
        if record and save_root is not None:
            os.makedirs(os.path.join(save_root, 'progress'), exist_ok=True)
        
        pbar = tqdm(range(n_steps))
        for step in pbar:
            # Propose new state
            x_proposed = self.propose(x_current)
            
            # Compute acceptance probability
            alpha = self.acceptance_probability(x_proposed, x_current, y, **kwargs)
            
            # Accept/reject
            if torch.rand(1) < alpha:
                x_current = x_proposed
                accepted += 1
            
            # Record if requested
            if record and step % 10 == 0 and save_root is not None:
                self.save_state(x_current, save_root, step)
            
            # Update progress bar
            pbar.set_postfix({
                'acceptance_rate': accepted/(step+1),
                'log_posterior': self.log_posterior(x_current, y, **kwargs).item()
            }, refresh=False)
            
        return x_current, accepted/n_steps
    
    def save_state(self, x, save_root, step):
        """Save current state."""
        import matplotlib.pyplot as plt
        from util.img_utils import clear_color
        
        file_path = os.path.join(save_root, f"progress/x_{str(step).zfill(4)}.png")
        plt.imsave(file_path, clear_color(x)) 