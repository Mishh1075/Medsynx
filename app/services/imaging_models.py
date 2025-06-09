import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import pydicom
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ChestXrayDataset(Dataset):
    """Dataset class for chest X-ray images"""
    def __init__(self, 
                 data_dir: Union[str, Path], 
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (320, 320)):
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.image_files = list(self.data_dir.glob('*.dcm'))
        
        # Disease labels from CheXpert
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_files[idx]
        image = self.load_and_preprocess(image_path)
        return image, str(image_path)
    
    def load_and_preprocess(self, path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess DICOM image"""
        ds = pydicom.dcmread(str(path))
        image = ds.pixel_array
        
        # Convert to RGB (3 channels)
        image = Image.fromarray(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class ChestXrayModel:
    """Wrapper for chest X-ray classification models"""
    def __init__(self, 
                 model_type: str = 'densenet121',
                 pretrained: bool = True,
                 num_classes: int = 14,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        
        # Initialize model
        if model_type == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = self.model.to(device)
        
        # Load pretrained weights if available
        weights_path = Path(__file__).parent / 'weights' / f'{model_type}_chexpert.pth'
        if weights_path.exists():
            self.model.load_state_dict(
                torch.load(weights_path, map_location=device)
            )
        
        self.model.eval()
    
    def predict(self, 
                images: Union[torch.Tensor, np.ndarray, List[Union[str, Path]]],
                batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Predict disease probabilities for chest X-ray images
        
        Args:
            images: Input images (tensor, numpy array, or list of paths)
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary with disease probabilities
        """
        if isinstance(images, (str, Path)):
            images = [images]
        
        if isinstance(images, list):
            dataset = ChestXrayDataset(images)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            predictions = []
            
            with torch.no_grad():
                for batch, _ in dataloader:
                    batch = batch.to(self.device)
                    output = torch.sigmoid(self.model(batch))
                    predictions.append(output.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
        else:
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            images = images.to(self.device)
            
            with torch.no_grad():
                predictions = torch.sigmoid(self.model(images))
                predictions = predictions.cpu().numpy()
        
        return {
            label: predictions[:, i]
            for i, label in enumerate(ChestXrayDataset.labels)
        }
    
    def fine_tune(self,
                  train_dir: Union[str, Path],
                  val_dir: Optional[Union[str, Path]] = None,
                  epochs: int = 10,
                  batch_size: int = 32,
                  learning_rate: float = 1e-4,
                  save_path: Optional[Union[str, Path]] = None) -> Dict[str, List[float]]:
        """
        Fine-tune the model on custom data
        
        Args:
            train_dir: Directory containing training images
            val_dir: Directory containing validation images
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_path: Path to save model weights
            
        Returns:
            Dictionary with training history
        """
        train_dataset = ChestXrayDataset(train_dir)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        if val_dir:
            val_dataset = ChestXrayDataset(val_dir)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [],
            'val_loss': [] if val_dir else None
        }
        
        self.model.train()
        for epoch in range(epochs):
            # Training
            train_loss = 0
            for images, _ in train_loader:
                images = images.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_dir:
                val_loss = 0
                self.model.eval()
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                self.model.train()
            
            print(f'Epoch {epoch+1}/{epochs}:',
                  f'train_loss = {train_loss:.4f}',
                  f'val_loss = {val_loss:.4f}' if val_dir else '')
        
        # Save model
        if save_path:
            torch.save(self.model.state_dict(), save_path)
        
        return history

class MIMICCXRModel(ChestXrayModel):
    """Extension of ChestXrayModel for MIMIC-CXR specific features"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # MIMIC-CXR specific labels
        self.mimic_labels = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
            'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Fracture'
        ]
        
        # Load MIMIC-CXR specific weights if available
        weights_path = Path(__file__).parent / 'weights' / 'mimic_cxr.pth'
        if weights_path.exists():
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
    
    def predict(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Override predict to use MIMIC-CXR labels"""
        predictions = super().predict(*args, **kwargs)
        return {
            label: predictions[label]
            for label in self.mimic_labels
        }

def load_pretrained_model(model_name: str = 'chexpert',
                         model_type: str = 'densenet121',
                         device: str = None) -> Union[ChestXrayModel, MIMICCXRModel]:
    """
    Load a pretrained chest X-ray model
    
    Args:
        model_name: 'chexpert' or 'mimic'
        model_type: 'densenet121' or 'resnet50'
        device: Device to load model on
        
    Returns:
        Pretrained model instance
    """
    if model_name == 'chexpert':
        return ChestXrayModel(model_type=model_type, device=device)
    elif model_name == 'mimic':
        return MIMICCXRModel(model_type=model_type, device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}") 