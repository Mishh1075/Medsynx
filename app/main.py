from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.config import settings
from app.core.security_middleware import SecurityMiddleware
from app.core.audit import AuditLogger
from app.db.base import get_db, engine, Base
from app.db.models import User
from app.services.data_processor import DataProcessor
from app.services.synthetic_generator import SyntheticGenerator
from app.services.evaluation_service import EvaluationService
from app.api.auth import router as auth_router, get_current_user

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="A scalable synthetic data generation platform for healthcare with differential privacy"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# Include authentication router
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])

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
            result = generator.generate_synthetic_data(processed_data)
            
            # Update job status
            job.status = "completed"
            job.completed_at = pd.Timestamp.now()
            db.commit()
            
            return {
                "message": "Synthetic data generated successfully",
                "job_id": job.id,
                "synthetic_data": result["synthetic_data"].to_dict(),
                "privacy_metrics": result["privacy_metrics"],
                "performance_metrics": result["performance_metrics"]
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