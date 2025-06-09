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
from typing import Dict, Any, Union, List
import cv2

class MedicalImageGenerator:
    def __init__(self, config: Dict[str, Any] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {
            "image_size": 256,
            "channels": 1,
            "latent_dim": 100,
            "num_epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "epsilon": 1.0,  # DP parameter
            "delta": 1e-5    # DP parameter
        }
        
        # Initialize GAN components
        self.generator = Generator(
            latent_shape=(self.config["latent_dim"],),
            channels=self.config["channels"],
            num_channels=[64, 128, 256, 512],
            strides=[2, 2, 2, 2],
        ).to(self.device)
        
        self.discriminator = Discriminator(
            in_shape=(self.config["channels"], self.config["image_size"], self.config["image_size"]),
            channels=[64, 128, 256, 512],
            strides=[2, 2, 2, 2],
        ).to(self.device)
        
    def train(self, data_loader: DataLoader, callback=None) -> Dict[str, List[float]]:
        """
        Train the GAN model on medical images
        """
        criterion = nn.BCELoss()
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["beta1"], 0.999)
        )
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["beta1"], 0.999)
        )
        
        history = {
            "d_losses": [],
            "g_losses": [],
            "real_scores": [],
            "fake_scores": []
        }
        
        for epoch in range(self.config["num_epochs"]):
            for i, real_images in enumerate(data_loader):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)
                
                output_real = self.discriminator(real_images)
                d_loss_real = criterion(output_real, label_real)
                
                noise = torch.randn(batch_size, self.config["latent_dim"]).to(self.device)
                fake_images = self.generator(noise)
                output_fake = self.discriminator(fake_images.detach())
                d_loss_fake = criterion(output_fake, label_fake)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                output_fake = self.discriminator(fake_images)
                g_loss = criterion(output_fake, label_real)
                g_loss.backward()
                g_optimizer.step()
                
                # Record losses
                history["d_losses"].append(d_loss.item())
                history["g_losses"].append(g_loss.item())
                history["real_scores"].append(output_real.mean().item())
                history["fake_scores"].append(output_fake.mean().item())
                
                if callback:
                    callback(epoch, i, history)
                
        return history
    
    def generate_images(self, num_images: int) -> torch.Tensor:
        """
        Generate synthetic medical images
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, self.config["latent_dim"]).to(self.device)
            synthetic_images = self.generator(noise)
        return synthetic_images
    
    @staticmethod
    def load_dicom(path: Union[str, Path]) -> np.ndarray:
        """
        Load DICOM image
        """
        return pydicom.dcmread(path).pixel_array
    
    @staticmethod
    def load_nifti(path: Union[str, Path]) -> np.ndarray:
        """
        Load NIfTI image
        """
        return nib.load(path).get_fdata()
    
    @staticmethod
    def save_dicom(image: np.ndarray, path: Union[str, Path], metadata: Dict = None):
        """
        Save image as DICOM
        """
        ds = pydicom.Dataset()
        if metadata:
            for key, value in metadata.items():
                setattr(ds, key, value)
        ds.PixelData = image.tobytes()
        ds.save_as(path)
    
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
        image = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))
        
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