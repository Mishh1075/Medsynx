from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import List

from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    datasets = relationship("Dataset", back_populates="owner")
    synthetic_jobs = relationship("SyntheticJob", back_populates="user")
    image_jobs = relationship("ImageGenerationJob", back_populates="user")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String, nullable=True)
    file_path = Column(String)
    file_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_synthetic = Column(Boolean, default=False)
    epsilon = Column(Float, nullable=True)  # DP parameter
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="datasets")
    synthetic_jobs = relationship("SyntheticJob", back_populates="dataset")

class SyntheticJob(Base):
    __tablename__ = "synthetic_jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String)  # running, completed, failed
    model_type = Column(String)
    epsilon = Column(Float)
    delta = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    
    # New fields for enhanced metrics
    privacy_metrics = Column(JSON, nullable=True)
    utility_metrics = Column(JSON, nullable=True)
    
    dataset = relationship("Dataset", back_populates="synthetic_jobs")
    user = relationship("User", back_populates="synthetic_jobs")

class ImageGenerationJob(Base):
    __tablename__ = "image_generation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    original_file = Column(String)
    status = Column(String)  # running, completed, failed
    num_images = Column(Integer)
    epsilon = Column(Float)
    delta = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    output_files = Column(JSON, nullable=True)  # List of generated image paths
    
    user = relationship("User", back_populates="image_jobs") 