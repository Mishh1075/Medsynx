import os
from typing import List, Tuple, Union
import numpy as np
import pydicom
import nibabel as nib
import cv2
from monai.transforms import (
    LoadImage,
    ScaleIntensity,
    Resize,
    ToTensor,
    Compose
)

class ImageProcessor:
    """Handles processing of medical images (DICOM, NIfTI, PNG, JPG)."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.transforms = Compose([
            ScaleIntensity(),
            Resize(spatial_size=target_size),
            ToTensor()
        ])
    
    def load_image(self, file_path: str) -> np.ndarray:
        """Load medical image from various formats."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.dcm':
            return self._load_dicom(file_path)
        elif ext in ['.nii', '.nii.gz']:
            return self._load_nifti(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return self._load_standard_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _load_dicom(self, file_path: str) -> np.ndarray:
        """Load DICOM image."""
        dcm = pydicom.dcmread(file_path)
        return dcm.pixel_array
    
    def _load_nifti(self, file_path: str) -> np.ndarray:
        """Load NIfTI image."""
        nifti = nib.load(file_path)
        return nifti.get_fdata()
    
    def _load_standard_image(self, file_path: str) -> np.ndarray:
        """Load standard image formats (PNG, JPG)."""
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        return self.transforms(image)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values to [0, 1] range."""
        return (image - image.min()) / (image.max() - image.min())
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.target_size)
    
    def apply_windowing(self, image: np.ndarray, 
                       window_center: float, 
                       window_width: float) -> np.ndarray:
        """Apply windowing to adjust image contrast."""
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        return np.clip(image, min_value, max_value)
    
    def extract_metadata(self, file_path: str) -> dict:
        """Extract metadata from medical image files."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.dcm':
            dcm = pydicom.dcmread(file_path)
            return {
                'PatientID': getattr(dcm, 'PatientID', 'Unknown'),
                'Modality': getattr(dcm, 'Modality', 'Unknown'),
                'StudyDate': getattr(dcm, 'StudyDate', 'Unknown'),
                'PixelSpacing': getattr(dcm, 'PixelSpacing', None)
            }
        elif ext in ['.nii', '.nii.gz']:
            nifti = nib.load(file_path)
            return {
                'Dimensions': nifti.header.get_data_shape(),
                'Affine': nifti.affine.tolist(),
                'Voxel_Size': nifti.header.get_zooms()
            }
        return {} 