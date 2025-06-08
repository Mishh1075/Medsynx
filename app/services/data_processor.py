import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, upload_dir: str):
        """
        Initialize the data processor.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        
    def save_uploaded_file(self, file: Any, filename: str) -> str:
        """
        Save an uploaded file to disk.
        
        Args:
            file: File object from FastAPI
            filename: Name of the file
            
        Returns:
            Path where the file was saved
        """
        try:
            file_path = os.path.join(self.upload_dir, filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
            
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file into a pandas DataFrame.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the loaded data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation_results = {
                "row_count": len(data),
                "column_count": len(data.columns),
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": data.dtypes.astype(str).to_dict(),
                "is_valid": True,
                "errors": []
            }
            
            # Check for empty dataset
            if len(data) == 0:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Dataset is empty")
                
            # Check for too many missing values
            missing_ratio = data.isnull().sum().mean() / len(data)
            if missing_ratio > 0.5:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Too many missing values")
                
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise
            
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for synthetic generation.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
                
            # Fill categorical missing values with mode
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                
            # Convert categorical variables to numeric
            for col in categorical_columns:
                df[col] = pd.Categorical(df[col]).codes
                
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
            
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing dataset summary
        """
        try:
            summary = {
                "shape": data.shape,
                "columns": list(data.columns),
                "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(data.select_dtypes(exclude=[np.number]).columns),
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": data.dtypes.astype(str).to_dict(),
                "summary_stats": data.describe().to_dict()
            }
            return summary
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            raise 