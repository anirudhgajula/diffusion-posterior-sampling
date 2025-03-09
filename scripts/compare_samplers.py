import os
import torch
import yaml
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA

from guided_diffusion.measurements import get_operator
from mcmc_samplers import PCNSampler, MHTVSampler
from util.img_utils import clear_color, mask_generator
from sample_condition import load_model_and_diffusion
from data.dataloader import get_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_measurement_kwargs(task_name, device):
    """Get task-specific measurement kwargs."""
    if task_name == 'inpainting':
        # Generate random mask for inpainting
        mask = mask_generator(1, 256, 256, 0.5).to(device)  # 50% missing pixels
        return {'mask': mask}
    return {}

def visualize_sample_distributions(all_samples, posterior_mean, posterior_cov, save_dir):
    """Visualize sample distributions using PCA and compare with true posterior.
    
    Args:
        all_samples: Dict of samples from different methods
        posterior_mean: True posterior mean
        posterior_cov: True posterior covariance
        save_dir: Directory to save plots
    """
    # Combine all samples for PCA
    all_combined = []
    labels = []
    for name, samples in all_samples.items():
        if len(samples) > 0:
            all_combined.append(samples)
            labels.extend([name] * len(samples))
    all_combined = torch.cat(all_combined, dim=0).cpu().numpy()
    
    # Fit PCA to all samples
    pca = PCA(n_components=2)
    samples_2d = pca.fit_transform(all_combined)
    
    # Project true posterior parameters to PCA space
    posterior_mean_2d = pca.transform(posterior_mean.cpu().numpy().reshape(1, -1))
    posterior_cov_2d = pca.transform(posterior_cov.cpu().numpy()) @ pca.components_
    
    # Create grid for posterior contour
    x, y = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x, y))
    rv = stats.multivariate_normal(posterior_mean_2d[0], posterior_cov_2d)
    
    # Plot samples and posterior contours
    plt.figure(figsize=(12, 8))
    
    # Plot true posterior contours
    plt.contour(x, y, rv.pdf(pos), levels=10, alpha=0.3, colors='k')
    
    # Plot samples from each method
    start_idx = 0
    for name, samples in all_samples.items():
        if len(samples) > 0:
            end_idx = start_idx + len(samples)
            plt.scatter(samples_2d[start_idx:end_idx, 0], 
                       samples_2d[start_idx:end_idx, 1],
                       alpha=0.5, label=name)
            start_idx = end_idx
    
    plt.scatter(posterior_mean_2d[0, 0], posterior_mean_2d[0, 1],
                color='red', marker='*', s=200, label='True Mean')
    
    plt.title('Sample Distributions (PCA projection)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'sample_distributions.png'))
    plt.close()
    
    # Plot marginal distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for name, samples in all_samples.items():
        if len(samples) > 0:
            samples_proj = pca.transform(samples.cpu().numpy())
            for i in range(2):
                sns.kdeplot(data=samples_proj[:, i], ax=axes[i], label=name)
    
    # Plot true posterior marginals
    x_range = np.linspace(-5, 5, 100)
    for i in range(2):
        pdf = stats.norm.pdf(x_range, posterior_mean_2d[0, i], np.sqrt(posterior_cov_2d[i, i]))
        axes[i].plot(x_range, pdf, 'k--', label='True Posterior')
        axes[i].set_title(f'PC {i+1} Marginal Distribution')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'marginal_distributions.png'))
    plt.close()

def compute_gaussian_kl_divergence(samples, true_mean, true_cov):
    """Compute KL divergence between sample distribution and true Gaussian posterior.
    
    KL[q(x)||p(x)] = 1/2 [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - d + log(|Σ₂|/|Σ₁|)]
    
    Args:
        samples: Tensor of samples [N, D] where N is number of samples, D is dimension
        true_mean: True posterior mean [D]
        true_cov: True posterior covariance [D, D]
    
    Returns:
        KL divergence estimate
    """
    # Compute sample statistics
    sample_mean = torch.mean(samples, dim=0)
    sample_cov = torch.cov(samples.T)
    
    # For numerical stability
    eps = 1e-8
    sample_cov = sample_cov + eps * torch.eye(sample_cov.shape[0], device=sample_cov.device)
    true_cov = true_cov + eps * torch.eye(true_cov.shape[0], device=true_cov.device)
    
    # Dimension
    d = true_mean.shape[0]
    
    # 1. Trace term: tr(Σ₂⁻¹Σ₁)
    true_cov_inv = torch.inverse(true_cov)
    trace_term = torch.trace(true_cov_inv @ sample_cov)
    
    # 2. Mean difference term: (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁)
    mean_diff = sample_mean - true_mean
    quad_term = mean_diff @ true_cov_inv @ mean_diff
    
    # 3. Log determinant term: log(|Σ₂|/|Σ₁|)
    logdet_term = torch.logdet(true_cov) - torch.logdet(sample_cov)
    
    # Combine terms
    kl = 0.5 * (trace_term + quad_term - d + logdet_term)
    
    return kl

def compute_true_posterior(H, y, prior_mean, prior_cov, noise_var):
    """Compute true posterior for linear Gaussian case.
    
    For y = Hx + ε where:
    - x ~ N(m, Σ)
    - ε ~ N(0, σ²I)
    
    The posterior is Gaussian with:
    - μ_post = Σ_post(Hᵀy/σ² + Σ⁻¹m)
    - Σ_post = (Hᵀ H/σ² + Σ⁻¹)⁻¹
    """
    H = torch.as_tensor(H)
    y = torch.as_tensor(y)
    prior_mean = torch.as_tensor(prior_mean)
    prior_cov = torch.as_tensor(prior_cov)
    
    # Compute posterior precision and mean
    noise_precision = 1.0 / noise_var
    posterior_precision = noise_precision * H.T @ H + torch.inverse(prior_cov)
    posterior_cov = torch.inverse(posterior_precision)
    posterior_mean = posterior_cov @ (noise_precision * H.T @ y + torch.inverse(prior_cov) @ prior_mean)
    
    return posterior_mean, posterior_cov

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--n_mcmc_steps', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--image_path', type=str, help='Path to test image. If not provided, will use random sample from dataset.')
    parser.add_argument('--compute_kl', action='store_true', help='Compute KL divergence for linear-Gaussian case')
    args = parser.parse_args()
    
    # Load configs
    model_config = load_config(args.model_config)
    diffusion_config = load_config(args.diffusion_config)
    task_config = load_config(args.task_config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load measurement operator and noise model
    measurement_config = task_config['measurement']
    noise_config = task_config.get('noise', {'name': 'gaussian', 'sigma': 0.1})
    measurement_op = get_operator(device=device, **measurement_config['operator'])
    
    # Get task-specific measurement kwargs
    measurement_kwargs = get_measurement_kwargs(measurement_config['operator']['name'], device)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    for subdir in ['pcn', 'mh_tv', 'diffusion']:
        os.makedirs(os.path.join(args.save_dir, subdir, 'progress'), exist_ok=True)
    
    # Load or create test image
    if args.image_path:
        # Load and preprocess custom image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        x_true = transform(Image.open(args.image_path)).unsqueeze(0).to(device)
    else:
        # Get image from dataset
        dataset = get_dataset(data_config=task_config['data'])
        x_true = dataset[0]['image'].unsqueeze(0).to(device)
    
    # Create measurement
    y = measurement_op.forward(x_true, **measurement_kwargs)
    if noise_config['name'] == 'gaussian':
        y = y + noise_config['sigma'] * torch.randn_like(y)
    elif noise_config['name'] == 'poisson':
        # Handle Poisson noise according to your noise implementation
        noiser = get_noise(**noise_config)
        y = noiser(y)
    
    # Initialize samplers with appropriate parameters
    samplers = {}
    
    # Only use pCN for linear operators
    if hasattr(measurement_op, 'transpose'):
        samplers['pcn'] = PCNSampler(measurement_op, noise_config, None, beta=0.1)
    
    # MH-TV works for both linear and non-linear
    samplers['mh_tv'] = MHTVSampler(measurement_op, noise_config, None, 
                                   step_size=0.1, tv_lambda=0.1)
    
    # Store samples for KL computation
    if args.compute_kl and hasattr(measurement_op, 'H'):  # Only for linear operators with explicit matrix
        all_samples = {
            'pcn': [],
            'mh_tv': [],
            'diffusion': []
        }
    
    # Run MCMC sampling
    x_init = torch.randn_like(x_true)
    results = {}
    
    for name, sampler in samplers.items():
        print(f"Running {name} sampler...")
        x_final, rate = sampler.sample(y, x_init, args.n_mcmc_steps, 
                                     record=True, 
                                     save_root=os.path.join(args.save_dir, name),
                                     **measurement_kwargs)
        results[name] = (x_final, rate)
        
        if args.compute_kl and hasattr(measurement_op, 'H'):
            # Collect samples during chain
            samples = sampler.get_samples(thin=10)  # Get every 10th sample
            all_samples[name] = samples.reshape(samples.shape[0], -1)  # Flatten spatial dimensions
    
    # Run diffusion sampling
    print("Running diffusion sampler...")
    model, diffusion = load_model_and_diffusion(model_config, diffusion_config)
    model.to(device)
    
    x_diffusion = diffusion.p_sample_loop(model, x_init.shape, y,
                                         measurement_op,
                                         record=True,
                                         save_root=os.path.join(args.save_dir, 'diffusion'))
    
    if args.compute_kl and hasattr(measurement_op, 'H'):
        # For diffusion, we need to run multiple chains to get samples
        diffusion_samples = []
        n_chains = 100
        for _ in tqdm(range(n_chains), desc="Collecting diffusion samples"):
            sample = diffusion.p_sample_loop(model, x_init.shape, y, measurement_op)
            diffusion_samples.append(sample.reshape(1, -1))
        all_samples['diffusion'] = torch.cat(diffusion_samples, dim=0)
        
        # Compute true posterior
        H = measurement_op.H
        prior_mean = torch.zeros(H.shape[1], device=device)
        prior_cov = torch.eye(H.shape[1], device=device)
        noise_var = noise_config['sigma']**2
        
        posterior_mean, posterior_cov = compute_true_posterior(
            H, y.reshape(-1), prior_mean, prior_cov, noise_var)
        
        # Compute KL divergence for each method
        kl_divergences = {}
        for name, samples in all_samples.items():
            if len(samples) > 0:
                kl = compute_gaussian_kl_divergence(samples, posterior_mean, posterior_cov)
                kl_divergences[name] = kl.item()
                print(f"KL divergence for {name}: {kl:.4f}")
        
        # Save KL divergences
        with open(os.path.join(args.save_dir, 'kl_divergences.txt'), 'w') as f:
            for name, kl in kl_divergences.items():
                f.write(f"{name}: {kl}\n")
        
        # Visualize sample distributions
        visualize_sample_distributions(all_samples, posterior_mean, posterior_cov, args.save_dir)
    
    # Save results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(clear_color(x_true))
    plt.title('Ground Truth')
    plt.axis('off')
    
    col = 2
    for name, (x_final, rate) in results.items():
        plt.subplot(1, 4, col)
        plt.imshow(clear_color(x_final))
        plt.title(f'{name}\nAcceptance Rate: {rate:.3f}')
        plt.axis('off')
        col += 1
    
    plt.subplot(144)
    plt.imshow(clear_color(x_diffusion))
    plt.title('Diffusion')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'comparison.png'))
    plt.close()
    
    # Save measurements
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(clear_color(x_true))
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(122)
    if y.shape == x_true.shape:  # For operators that preserve size
        plt.imshow(clear_color(y))
    else:  # For operators that change size (e.g., super-resolution)
        plt.imshow(clear_color(measurement_op.transpose(y)))
    plt.title('Measurement')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'measurement.png'))
    plt.close()

if __name__ == '__main__':
    main() 