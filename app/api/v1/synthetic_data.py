from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Dict, Any
import pandas as pd
import io
import json
from app.services.synthetic_generator import SyntheticGenerator
from app.core.auth import get_current_user
from app.models.user import User
from app.models.job import Job
from app.db.session import get_db
from sqlalchemy.orm import Session

router = APIRouter()
synthetic_generator = SyntheticGenerator()

@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload data for synthetic generation"""
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Create a new job
        job = Job(
            user_id=current_user.id,
            status="processing",
            original_filename=file.filename
        )
        db.add(job)
        db.commit()
        
        # Generate synthetic data
        result = synthetic_generator.generate_synthetic_data(df)
        
        # Save synthetic data
        output_path = f"data/synthetic/{job.id}.csv"
        result["synthetic_data"].to_csv(output_path, index=False)
        
        # Update job status
        job.status = "completed"
        job.result_path = output_path
        job.metrics = json.dumps({
            "privacy_metrics": result["privacy_metrics"],
            "performance_metrics": result["performance_metrics"]
        })
        db.commit()
        
        return {
            "job_id": job.id,
            "status": "completed",
            "metrics": {
                "privacy_metrics": result["privacy_metrics"],
                "performance_metrics": result["performance_metrics"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get job status and results"""
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return {
        "job_id": job.id,
        "status": job.status,
        "metrics": json.loads(job.metrics) if job.metrics else None
    }

@router.get("/download/{job_id}")
async def download_synthetic_data(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download generated synthetic data"""
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job or not job.result_path:
        raise HTTPException(status_code=404, detail="Synthetic data not found")
        
    return FileResponse(
        job.result_path,
        filename=f"synthetic_{job.original_filename}",
        media_type="text/csv"
    ) 