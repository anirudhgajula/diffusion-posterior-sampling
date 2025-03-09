import os
import torch
import yaml
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--n_mcmc_steps', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--image_path', type=str, help='Path to test image. If not provided, will use random sample from dataset.')
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
    
    # Run MCMC sampling
    x_init = torch.randn_like(x_true)
    results = {}
    
    for name, sampler in samplers.items():
        print(f"Running {name} sampler...")
        results[name] = sampler.sample(y, x_init, args.n_mcmc_steps, 
                                     record=True, 
                                     save_root=os.path.join(args.save_dir, name),
                                     **measurement_kwargs)
    
    # Run diffusion sampling
    print("Running diffusion sampler...")
    model, diffusion = load_model_and_diffusion(model_config, diffusion_config)
    model.to(device)
    
    # The diffusion sampling part should be implemented according to your existing codebase
    x_diffusion = diffusion.p_sample_loop(model, x_init.shape, y,
                                         measurement_op,
                                         record=True,
                                         save_root=os.path.join(args.save_dir, 'diffusion'))
    
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