import numpy as np
import pydicom
import nibabel as nib
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from monai.transforms import (
    LoadImage,
    ScaleIntensity,
    Resize,
    NormalizeIntensity,
    SpatialPad
)

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """Process and normalize medical images for synthetic generation."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.supported_formats = {
            '.dcm': self._load_dicom,
            '.nii': self._load_nifti,
            '.nii.gz': self._load_nifti,
            '.png': self._load_standard_image,
            '.jpg': self._load_standard_image,
            '.jpeg': self._load_standard_image
        }
        
    def process_image(self, file_path: str) -> Dict[str, Any]:
        """
        Process medical image and extract metadata.
        
        Args:
            file_path: Path to the medical image file
            
        Returns:
            Dictionary containing processed image and metadata
        """
        try:
            file_path = Path(file_path)
            extension = ''.join(file_path.suffixes)
            
            if extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {extension}")
                
            # Load and process image
            image_data = self.supported_formats[extension](file_path)
            
            # Normalize and resize
            processed_image = self._preprocess_image(image_data['image'])
            
            return {
                'processed_image': processed_image,
                'original_size': image_data.get('original_size'),
                'metadata': image_data.get('metadata', {}),
                'modality': image_data.get('modality', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error processing medical image: {str(e)}")
            raise
            
    def _load_dicom(self, file_path: Path) -> Dict[str, Any]:
        """Load and process DICOM image."""
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(str(file_path))
            
            # Extract image data
            image = dcm.pixel_array
            
            # Extract relevant metadata
            metadata = {
                'PatientID': str(getattr(dcm, 'PatientID', 'unknown')),
                'Modality': str(getattr(dcm, 'Modality', 'unknown')),
                'StudyDescription': str(getattr(dcm, 'StudyDescription', 'unknown')),
                'SeriesDescription': str(getattr(dcm, 'SeriesDescription', 'unknown')),
                'PixelSpacing': getattr(dcm, 'PixelSpacing', [1.0, 1.0]),
                'ImageOrientation': getattr(dcm, 'ImageOrientationPatient', [1,0,0,0,1,0])
            }
            
            return {
                'image': image,
                'original_size': image.shape,
                'metadata': metadata,
                'modality': metadata['Modality']
            }
            
        except Exception as e:
            logger.error(f"Error loading DICOM image: {str(e)}")
            raise
            
    def _load_nifti(self, file_path: Path) -> Dict[str, Any]:
        """Load and process NIfTI image."""
        try:
            # Load NIfTI file
            nifti = nib.load(str(file_path))
            
            # Get image data
            image = nifti.get_fdata()
            
            # If 3D/4D, take middle slice
            if len(image.shape) > 2:
                middle_idx = image.shape[2] // 2
                image = image[:, :, middle_idx]
            
            # Extract metadata
            metadata = {
                'affine': nifti.affine.tolist(),
                'dimensions': nifti.header['dim'].tolist(),
                'voxel_size': nifti.header['pixdim'].tolist(),
                'description': str(getattr(nifti, 'description', 'unknown'))
            }
            
            return {
                'image': image,
                'original_size': image.shape,
                'metadata': metadata,
                'modality': 'MR'  # Assume MR for NIfTI
            }
            
        except Exception as e:
            logger.error(f"Error loading NIfTI image: {str(e)}")
            raise
            
    def _load_standard_image(self, file_path: Path) -> Dict[str, Any]:
        """Load and process standard image formats (PNG, JPG)."""
        try:
            # Read image
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
                
            metadata = {
                'format': file_path.suffix[1:].upper(),
                'color_space': 'grayscale'
            }
            
            return {
                'image': image,
                'original_size': image.shape,
                'metadata': metadata,
                'modality': 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error loading standard image: {str(e)}")
            raise
            
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        try:
            # Ensure float32
            image = image.astype(np.float32)
            
            # Normalize intensity
            normalizer = NormalizeIntensity()
            image = normalizer(image)
            
            # Resize to target size
            if image.shape != self.target_size:
                image = cv2.resize(image, self.target_size, 
                                 interpolation=cv2.INTER_LINEAR)
            
            # Scale to [-1, 1]
            scaler = ScaleIntensity(minv=-1.0, maxv=1.0)
            image = scaler(image)
            
            # Add channel dimension if needed
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
                
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
            
    def save_image(self, image: np.ndarray, file_path: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save processed/synthetic image."""
        try:
            file_path = Path(file_path)
            extension = ''.join(file_path.suffixes)
            
            # Ensure image is in correct format
            if len(image.shape) == 3 and image.shape[0] == 1:
                image = image[0]  # Remove channel dimension
                
            # Scale to [0, 255]
            image = ((image + 1) * 127.5).astype(np.uint8)
            
            if extension == '.dcm' and metadata:
                # Create DICOM file
                dcm = pydicom.Dataset()
                
                # Set required DICOM attributes
                dcm.PatientID = metadata.get('PatientID', 'synthetic')
                dcm.Modality = metadata.get('Modality', 'OT')
                dcm.StudyDescription = metadata.get('StudyDescription', 'Synthetic Image')
                dcm.SeriesDescription = metadata.get('SeriesDescription', 'Synthetic Series')
                dcm.PixelSpacing = metadata.get('PixelSpacing', [1.0, 1.0])
                dcm.ImageOrientationPatient = metadata.get('ImageOrientation', [1,0,0,0,1,0])
                
                # Set image data
                dcm.PixelData = image.tobytes()
                dcm.save_as(str(file_path))
                
            elif extension in ['.nii', '.nii.gz'] and metadata:
                # Create NIfTI file
                affine = np.array(metadata.get('affine', np.eye(4)))
                nifti_image = nib.Nifti1Image(image, affine)
                nib.save(nifti_image, str(file_path))
                
            else:
                # Save as standard image format
                cv2.imwrite(str(file_path), image)
                
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise 