from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedSynX"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    RATE_LIMIT_SECONDS: int = 1  # 1 request per second per IP
    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 15
    PASSWORD_MIN_LENGTH: int = 12
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./medsynx.db")
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: list = [".csv", ".xlsx", ".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg"]
    
    # Synthetic Data Generation
    DEFAULT_EPSILON: float = 1.0  # Default differential privacy parameter
    DEFAULT_DELTA: float = 1e-5   # Default delta parameter for DP
    MIN_EPSILON: float = 0.1      # Minimum allowed epsilon for strong privacy
    MAX_EPSILON: float = 10.0     # Maximum allowed epsilon
    MIN_DATASET_SIZE: int = 100   # Minimum records needed for synthesis
    MAX_TRAINING_TIME: int = 3600 # Maximum training time in seconds
    
    # Privacy Settings
    PRIVACY_METRICS_THRESHOLD: float = 0.7  # Minimum privacy score required
    K_ANONYMITY_MIN: int = 5               # Minimum k-anonymity value
    L_DIVERSITY_MIN: int = 3               # Minimum l-diversity value
    MAX_QUERY_RESULTS: int = 1000         # Maximum results per query
    DATA_RETENTION_DAYS: int = 30         # Number of days to retain uploaded data
    
    # Monitoring
    ENABLE_AUDIT_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"
    MONITORING_WEBHOOK_URL: Optional[str] = os.getenv("MONITORING_WEBHOOK_URL")
    
    class Config:
        case_sensitive = True

settings = Settings() 