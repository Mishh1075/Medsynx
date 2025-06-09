from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from dotenv import load_dotenv
import secrets

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Synthetic Data Generator"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "synthetic_data"
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: list = [".csv", ".xlsx", ".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg"]
    
    # Synthetic Data Generation
    DEFAULT_EPSILON: float = 1.0  # Differential privacy parameter
    DEFAULT_DELTA: float = 1e-5   # DP relaxation parameter
    DEFAULT_MODEL: str = "dpgan"   # Default synthetic data model
    SUPPORTED_MODELS: List[str] = ["dpgan", "pategan", "ctgan"]
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".csv"]
    
    # Data Storage
    UPLOAD_FOLDER: str = "data/original"
    SYNTHETIC_FOLDER: str = "data/synthetic"
    
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
        env_file = ".env"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
                f"{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
            )

settings = Settings() 