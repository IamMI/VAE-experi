"""
Build CVAE model
"""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class ConditionalVAE(nn.Module):
    def __init__(self, num_classes, embedding_dim=32*32):
        super(ConditionalVAE, self).__init__()
        self.vae = AutoencoderKL(in_channels=4, out_channels=4, latent_channels=4)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        
    def encode(self, x, condition):  
        # Merge x with condition
        cond_embed = self.embedding(condition).view(-1, 1, 32, 32).repeat(1, 4, 1, 1)
        x = x+cond_embed
        # Encode
        posterior = self.vae.encode(x).latent_dist
        
        return posterior

    def decode(self, z, condition):
        cond_embed = self.embedding(condition).view(-1, 1, 32, 32).repeat(1, 4, 1, 1)
        z = z+cond_embed
        
        return self.vae.decode(z).sample

    def forward(self, x, condition):
        posterior = self.encode(x, condition)
        z = posterior.sample()
        recon_x = self.decode(z, condition)
        return recon_x, posterior, z


if __name__ == "__main__":
    # Load VAE encoder
    model = ConditionalVAE(num_classes=10)
    x = torch.randn(3, 4, 32, 32)
    y, z = model(x, torch.tensor([1, 2, 3]))
    print(y.shape, z.shape)
    print("Done!")
    
