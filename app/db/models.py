from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    
    datasets = relationship("Dataset", back_populates="owner")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    file_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_synthetic = Column(Boolean, default=False)
    epsilon = Column(Float, nullable=True)  # DP parameter
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="datasets")

class SyntheticJob(Base):
    __tablename__ = "synthetic_jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    status = Column(String)  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    epsilon = Column(Float)
    delta = Column(Float)
    model_type = Column(String)  # dpgan, pategan, etc.
    error_message = Column(String, nullable=True)
    
    dataset = relationship("Dataset") 