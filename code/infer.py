"""
Model infer
"""
import torch
from CVAE import ConditionalVAE
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
current_work_dir = os.path.dirname(os.path.dirname(__file__))


def save_image(tensor, fp):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ndarr)
    ax.axis('off')
    plt.savefig(fp, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
    model = ConditionalVAE(num_classes=1000, embedding_dim=32*32)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(os.path.join(current_work_dir, 'checkpoints/CVAE-MI-099.pt'))['model_state_dict'] )
    model = model.to(device)
    vae = vae.to(device)
    
    # sample
    z = torch.randn(1, 4, 32, 32).to(device)
    c = torch.tensor(15).to(device)
    
    recon_x = model.decode(z, c)
    sample = vae.decode(recon_x / 0.18215).sample
    
    save_image(sample, os.path.join(current_work_dir, 'sample.png'))
    
    
    
    
    
