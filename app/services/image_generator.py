import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import monai
from monai.networks.nets import Generator, Discriminator
from monai.data import ImageDataset
from pathlib import Path
import nibabel as nib
import pydicom
from typing import Dict, Any, Union, List, Optional
import cv2
import os
import torch.nn.functional as F
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 1):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Initial size: (batch_size, latent_dim, 1, 1)
        self.conv_blocks = nn.Sequential(
            # Block 1: (batch_size, latent_dim, 1, 1) -> (batch_size, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Block 2: (batch_size, 512, 4, 4) -> (batch_size, 256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Block 3: (batch_size, 256, 8, 8) -> (batch_size, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Block 4: (batch_size, 128, 16, 16) -> (batch_size, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 5: (batch_size, 64, 32, 32) -> (batch_size, channels, 64, 64)
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Reshape noise: (batch_size, latent_dim) -> (batch_size, latent_dim, 1, 1)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.conv_blocks(z)

class Discriminator(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1: (batch_size, channels, 64, 64) -> (batch_size, 64, 32, 32)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: (batch_size, 64, 32, 32) -> (batch_size, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: (batch_size, 128, 16, 16) -> (batch_size, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4: (batch_size, 256, 8, 8) -> (batch_size, 512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 5: (batch_size, 512, 4, 4) -> (batch_size, 1, 1, 1)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(x).view(-1, 1).squeeze(1)

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir: Union[str, Path], transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_files = list(self.image_dir.glob('*.dcm'))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_files[idx]
        image = self.load_dicom(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    @staticmethod
    def load_dicom(path: Union[str, Path]) -> np.ndarray:
        ds = pydicom.dcmread(str(path))
        return ds.pixel_array

class MedicalImageGenerator:
    def __init__(
        self,
        latent_dim: int = 100,
        image_size: int = 64,
        channels: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.device = device
        
        # Initialize generator
        self.generator = Generator(latent_dim, channels).to(device)
        
        # Load pre-trained weights if available
        weights_path = Path(__file__).parent / 'weights' / 'generator.pth'
        if weights_path.exists():
            self.generator.load_state_dict(
                torch.load(weights_path, map_location=device)
            )
        
        self.generator.eval()
    
    def generate_images(
        self,
        num_images: int = 1,
        noise: Optional[torch.Tensor] = None
    ) -> List[np.ndarray]:
        """Generate synthetic medical images"""
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(
                    num_images,
                    self.latent_dim,
                    device=self.device
                )
            
            # Generate images
            fake_images = self.generator(noise)
            
            # Convert to numpy arrays
            images = []
            for img in fake_images:
                # Denormalize from [-1, 1] to [0, 255]
                img = ((img.cpu().numpy() + 1) * 127.5).astype(np.uint8)
                img = img.squeeze()  # Remove channel dimension for grayscale
                images.append(img)
            
            return images
    
    @staticmethod
    def load_dicom(path: Union[str, Path]) -> np.ndarray:
        """Load a DICOM image"""
        ds = pydicom.dcmread(str(path))
        return ds.pixel_array
    
    @staticmethod
    def save_dicom(
        image: np.ndarray,
        output_path: Union[str, Path],
        template_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Save an image as DICOM"""
        if template_path:
            # Copy metadata from template
            ds = pydicom.dcmread(str(template_path))
            ds.PixelData = image.tobytes()
            ds.Rows, ds.Columns = image.shape
        else:
            # Create new DICOM dataset
            ds = pydicom.Dataset()
            ds.PixelData = image.tobytes()
            ds.Rows, ds.Columns = image.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            
            # Add required DICOM tags
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.Modality = 'OT'  # Other
        
        ds.save_as(str(output_path))
    
    def train(
        self,
        data_dir: Union[str, Path],
        num_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.0002,
        beta1: float = 0.5,
        save_interval: int = 10
    ) -> Dict[str, List[float]]:
        """Train the GAN on medical images"""
        # Initialize discriminator
        discriminator = Discriminator(self.channels).to(self.device)
        
        # Setup optimizers
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        
        # Create dataset
        dataset = MedicalImageDataset(
            data_dir,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Training loop
        g_losses = []
        d_losses = []
        
        for epoch in range(num_epochs):
            for i, real_images in enumerate(dataloader):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Train discriminator
                d_optimizer.zero_grad()
                label_real = torch.ones(batch_size, device=self.device)
                label_fake = torch.zeros(batch_size, device=self.device)
                
                output_real = discriminator(real_images)
                d_loss_real = F.binary_cross_entropy(output_real, label_real)
                
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise)
                output_fake = discriminator(fake_images.detach())
                d_loss_fake = F.binary_cross_entropy(output_fake, label_fake)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train generator
                g_optimizer.zero_grad()
                output_fake = discriminator(fake_images)
                g_loss = F.binary_cross_entropy(output_fake, label_real)
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'D_loss: {d_loss.item():.4f} '
                  f'G_loss: {g_loss.item():.4f}')
            
            # Save model periodically
            if (epoch + 1) % save_interval == 0:
                weights_dir = Path(__file__).parent / 'weights'
                weights_dir.mkdir(exist_ok=True)
                torch.save(
                    self.generator.state_dict(),
                    weights_dir / 'generator.pth'
                )
        
        return {
            'generator_losses': g_losses,
            'discriminator_losses': d_losses
        }
    
    @staticmethod
    def load_nifti(path: Union[str, Path]) -> np.ndarray:
        """
        Load NIfTI image
        """
        return nib.load(path).get_fdata()
    
    @staticmethod
    def save_nifti(image: np.ndarray, path: Union[str, Path], affine: np.ndarray = None):
        """
        Save image as NIfTI
        """
        if affine is None:
            affine = np.eye(4)
        nib.save(nib.Nifti1Image(image, affine), path)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for the model
        """
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            
        return image
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output to numpy array
        """
        # Convert to numpy
        image = tensor.cpu().numpy()
        
        # Remove channel dimension if needed
        if image.shape[0] == 1:
            image = image.squeeze(0)
            
        # Rescale to original range
        image = (image * 255).astype(np.uint8)
            
        return image 