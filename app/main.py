from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import cv2
from pathlib import Path

from app.core.config import settings
from app.core.security_middleware import SecurityMiddleware
from app.core.audit import AuditLogger
from app.db.base import get_db, engine, Base
from app.db.models import User
from app.services.data_processor import DataProcessor
from app.services.synthetic_generator import SyntheticGenerator
from app.services.evaluation_service import EvaluationService
from app.api.auth import router as auth_router, get_current_user
from app.api.v1 import auth as auth_v1, synthetic_data
from app.services.privacy_evaluator import PrivacyEvaluator
from app.services.utility_evaluator import UtilityEvaluator
from app.services.image_generator import MedicalImageGenerator

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="A scalable synthetic data generation platform for healthcare with differential privacy",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityMiddleware)

# Initialize services
data_processor = DataProcessor(settings.UPLOAD_DIR)
synthetic_generator = SyntheticGenerator()
evaluation_service = EvaluationService()
audit_logger = AuditLogger()
privacy_evaluator = PrivacyEvaluator()
utility_evaluator = UtilityEvaluator()
image_generator = MedicalImageGenerator()

# Include routers
app.include_router(auth_v1.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(synthetic_data.router, prefix=f"{settings.API_V1_STR}/synthetic", tags=["synthetic"])

@app.middleware("http")
async def audit_requests(request: Request, call_next):
    """Middleware to audit all requests."""
    start_time = datetime.utcnow()

@app.get("/")
def read_root():
    return {"message": "Welcome to MedSynX API"}

@app.post("/api/v1/upload")
async def upload_data(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload a dataset for synthetic data generation.
    """
    try:
        # Save the uploaded file
        file_path = data_processor.save_uploaded_file(file, file.filename)
        
        # Load the data
        data = data_processor.load_data(file_path)
        
        # Validate the data
        validation_results = data_processor.validate_data(data)
        if not validation_results["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data: {validation_results['errors']}"
            )
            
        # Get data summary
        data_summary = data_processor.get_data_summary(data)
        
        # Create dataset record in database
        from app.db.models import Dataset
        dataset = Dataset(
            name=file.filename,
            file_path=file_path,
            owner_id=current_user.id
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return {
            "message": "Data uploaded successfully",
            "dataset_id": dataset.id,
            "file_path": file_path,
            "validation_results": validation_results,
            "data_summary": data_summary
        }
        
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/generate")
async def generate_synthetic_data(
    dataset_id: int,
    epsilon: float = settings.DEFAULT_EPSILON,
    delta: float = settings.DEFAULT_DELTA,
    model_type: str = "dpgan",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate synthetic data from an uploaded dataset.
    """
    try:
        # Get dataset
        from app.db.models import Dataset
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == current_user.id
        ).first()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Load the data
        data = data_processor.load_data(dataset.file_path)
        
        # Preprocess the data
        processed_data = data_processor.preprocess_data(data)
        
        # Create synthetic job record
        from app.db.models import SyntheticJob
        job = SyntheticJob(
            dataset_id=dataset.id,
            status="running",
            epsilon=epsilon,
            delta=delta,
            model_type=model_type
        )
        db.add(job)
        db.commit()
        
        try:
            # Initialize generator with parameters
            generator = SyntheticGenerator(
                epsilon=epsilon,
                delta=delta,
                model_type=model_type
            )
            
            # Generate synthetic data
            synthetic_result = generator.generate_synthetic_data(processed_data)
            
            # Evaluate privacy
            privacy_metrics = privacy_evaluator.evaluate_privacy(
                processed_data.values,
                synthetic_result["synthetic_data"].values
            )
            
            # Evaluate utility
            utility_metrics = utility_evaluator.evaluate_utility(
                processed_data,
                synthetic_result["synthetic_data"]
            )
            
            # Update job status and metrics
            job.status = "completed"
            job.completed_at = pd.Timestamp.now()
            job.privacy_metrics = privacy_metrics
            job.utility_metrics = utility_metrics
            db.commit()
            
            return {
                "message": "Synthetic data generated successfully",
                "job_id": job.id,
                "synthetic_data": synthetic_result["synthetic_data"].to_dict(),
                "privacy_metrics": privacy_metrics,
                "utility_metrics": utility_metrics
            }
            
        except Exception as e:
            # Update job status on failure
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            raise
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/models")
async def list_available_models(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List available synthetic data generation models.
    """
    try:
        plugins = synthetic_generator.plugins.list()
        return {
            "models": plugins
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/datasets")
async def list_user_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all datasets belonging to the current user.
    """
    try:
        from app.db.models import Dataset
        datasets = db.query(Dataset).filter(Dataset.owner_id == current_user.id).all()
        return {
            "datasets": [
                {
                    "id": d.id,
                    "name": d.name,
                    "created_at": d.created_at,
                    "is_synthetic": d.is_synthetic
                }
                for d in datasets
            ]
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/generate/image")
async def generate_synthetic_images(
    file: UploadFile = File(...),
    num_images: int = 1,
    epsilon: float = settings.DEFAULT_EPSILON,
    delta: float = settings.DEFAULT_DELTA,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate synthetic medical images.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format"
            )
        
        # Save uploaded file
        file_path = f"data/original/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create image generation job
        job = ImageGenerationJob(
            user_id=current_user.id,
            original_file=file_path,
            status="running",
            num_images=num_images,
            epsilon=epsilon,
            delta=delta
        )
        db.add(job)
        db.commit()
        
        try:
            # Load and preprocess image
            if file.filename.endswith('.dcm'):
                image = image_generator.load_dicom(file_path)
            elif file.filename.endswith(('.nii', '.nii.gz')):
                image = image_generator.load_nifti(file_path)
            else:
                image = cv2.imread(file_path)
            
            # Preprocess image
            processed_image = image_generator.preprocess_image(image)
            
            # Generate synthetic images
            synthetic_images = image_generator.generate_images(num_images)
            
            # Save synthetic images
            output_paths = []
            for i, img in enumerate(synthetic_images):
                output_path = f"data/synthetic/synthetic_{job.id}_{i}{Path(file.filename).suffix}"
                
                if file.filename.endswith('.dcm'):
                    image_generator.save_dicom(img, output_path)
                elif file.filename.endswith(('.nii', '.nii.gz')):
                    image_generator.save_nifti(img, output_path)
                else:
                    cv2.imwrite(output_path, img)
                    
                output_paths.append(output_path)
            
            # Update job status
            job.status = "completed"
            job.completed_at = pd.Timestamp.now()
            job.output_files = output_paths
            db.commit()
            
            return {
                "message": "Synthetic images generated successfully",
                "job_id": job.id,
                "output_paths": output_paths
            }
            
        except Exception as e:
            # Update job status on failure
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            raise
            
    except Exception as e:
        logger.error(f"Error generating synthetic images: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 