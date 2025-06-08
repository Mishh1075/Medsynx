import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from monai.networks.nets import AutoEncoder
from monai.losses import PerceptualLoss

class ImageGenerator:
    """Generates synthetic medical images using various GAN architectures."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (256, 256),
                 latent_dim: int = 128,
                 channels: int = 1,
                 epsilon: float = 1.0,
                 delta: float = 1e-5):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.channels = channels
        self.epsilon = epsilon
        self.delta = delta
        
        # Initialize models
        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()
        self.autoencoder = AutoEncoder(
            spatial_dims=2,
            in_channels=channels,
            out_channels=channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2, 2)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.perceptual_loss = PerceptualLoss(spatial_dims=2)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    
    def _create_generator(self) -> nn.Module:
        """Create the generator network."""
        return nn.Sequential(
            # Latent vector input
            nn.Linear(self.latent_dim, 128 * (self.image_size[0] // 8) * (self.image_size[1] // 8)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128 * (self.image_size[0] // 8) * (self.image_size[1] // 8)),
            
            # Reshape
            lambda x: x.view(-1, 128, self.image_size[0] // 8, self.image_size[1] // 8),
            
            # Upsampling layers
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, self.channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def _create_discriminator(self) -> nn.Module:
        """Create the discriminator network."""
        return nn.Sequential(
            nn.Conv2d(self.channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),
            nn.Linear(128 * (self.image_size[0] // 8) * (self.image_size[1] // 8), 1),
            nn.Sigmoid()
        )
    
    def add_noise(self, tensor: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise for differential privacy."""
        noise = torch.randn_like(tensor) * noise_level * self.epsilon
        return tensor + noise
    
    def train_step(self, real_images: torch.Tensor) -> dict:
        """Perform one training step."""
        batch_size = real_images.size(0)
        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_output = self.discriminator(real_images)
        d_loss_real = self.adversarial_loss(real_output, real_label)
        
        # Fake images
        z = torch.randn(batch_size, self.latent_dim)
        fake_images = self.generator(z)
        fake_images = self.add_noise(fake_images)  # Add DP noise
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.adversarial_loss(fake_output, fake_label)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        fake_output = self.discriminator(fake_images)
        g_loss = self.adversarial_loss(fake_output, real_label)
        
        # Add perceptual loss
        p_loss = self.perceptual_loss(fake_images, real_images)
        total_g_loss = g_loss + 0.1 * p_loss
        
        total_g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_loss': total_g_loss.item(),
            'd_loss': d_loss.item(),
            'p_loss': p_loss.item()
        }
    
    def generate(self, num_images: int = 1) -> torch.Tensor:
        """Generate synthetic images."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim)
            fake_images = self.generator(z)
            fake_images = self.add_noise(fake_images)
        return fake_images
    
    def save_models(self, path: str):
        """Save generator and discriminator models."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, path)
    
    def load_models(self, path: str):
        """Load generator and discriminator models."""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict']) 