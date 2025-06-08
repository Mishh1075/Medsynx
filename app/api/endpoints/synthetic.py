from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from ...services.imaging import ImageProcessor, ImageGenerator, ImageEvaluator
from ...core.security import get_current_user
from ...db.models import User
import io
import json
import pandas as pd

router = APIRouter()

# Initialize services
image_processor = ImageProcessor()
image_generator = ImageGenerator()
image_evaluator = ImageEvaluator()

@router.post("/upload/medical-image")
async def upload_medical_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload and process medical image."""
    try:
        contents = await file.read()
        file_path = f"temp/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process image
        image = image_processor.load_image(file_path)
        metadata = image_processor.extract_metadata(file_path)
        
        # Preprocess for model
        processed_image = image_processor.preprocess(image)
        
        return {
            "status": "success",
            "filename": file.filename,
            "metadata": metadata,
            "shape": image.shape
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate/medical-image")
async def generate_synthetic_images(
    num_images: int,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate synthetic medical images."""
    try:
        # Update privacy parameters
        image_generator.epsilon = epsilon
        image_generator.delta = delta
        
        # Generate synthetic images
        synthetic_images = image_generator.generate(num_images)
        
        # Convert to base64 for transfer
        images_list = []
        for i in range(num_images):
            img = synthetic_images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img_bytes = io.BytesIO()
            np.save(img_bytes, img)
            images_list.append(img_bytes.getvalue())
        
        return {
            "status": "success",
            "num_generated": num_images,
            "images": images_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate/medical-image")
async def evaluate_synthetic_images(
    real_images: List[UploadFile] = File(...),
    synthetic_images: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Evaluate synthetic medical images."""
    try:
        # Load and process images
        real_tensors = []
        synthetic_tensors = []
        
        for real_file, syn_file in zip(real_images, synthetic_images):
            # Process real images
            real_contents = await real_file.read()
            real_path = f"temp/{real_file.filename}"
            with open(real_path, "wb") as f:
                f.write(real_contents)
            real_img = image_processor.load_image(real_path)
            real_tensor = image_processor.preprocess(real_img)
            real_tensors.append(real_tensor)
            
            # Process synthetic images
            syn_contents = await syn_file.read()
            syn_path = f"temp/{syn_file.filename}"
            with open(syn_path, "wb") as f:
                f.write(syn_contents)
            syn_img = image_processor.load_image(syn_path)
            syn_tensor = image_processor.preprocess(syn_img)
            synthetic_tensors.append(syn_tensor)
        
        # Stack tensors
        real_batch = torch.stack(real_tensors)
        synthetic_batch = torch.stack(synthetic_tensors)
        
        # Calculate metrics
        metrics = image_evaluator.calculate_metrics(real_batch, synthetic_batch)
        
        # Generate report
        report = image_evaluator.generate_report(save_path=f"reports/{current_user.id}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "report_path": f"reports/{current_user.id}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/visualization/medical-image/{report_id}")
async def get_visualizations(
    report_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get visualization data for a specific report."""
    try:
        report_path = f"reports/{report_id}"
        
        # Load metrics from CSV
        metrics_df = pd.read_csv(f"{report_path}/metrics.csv")
        
        # Load summary
        with open(f"{report_path}/summary.txt", 'r') as f:
            summary = f.read()
        
        return {
            "status": "success",
            "metrics_data": metrics_df.to_dict(orient='records'),
            "summary": summary,
            "visualization_paths": {
                "metrics_over_time": f"{report_path}/metrics_over_time.png",
                "privacy_risks": f"{report_path}/privacy_risks.png"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 