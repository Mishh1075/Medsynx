from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.core.schema import Schema
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
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
        self.plugins = Plugins()
        
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
            
    def infer_schema(self, data: pd.DataFrame) -> Schema:
        """
        Infer the schema of the input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Schema object with column types and constraints
        """
        try:
            # Infer data types
            categorical_columns = []
            numerical_columns = []
            datetime_columns = []
            
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numerical_columns.append(col)
                elif pd.api.types.is_datetime64_any_dtype(data[col]):
                    datetime_columns.append(col)
                else:
                    categorical_columns.append(col)
            
            return Schema(
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                datetime_columns=datetime_columns
            )
        except Exception as e:
            logger.error(f"Error inferring schema: {str(e)}")
            raise
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        schema: Optional[Schema] = None,
        target_column: Optional[str] = None,
        sensitive_features: Optional[List[str]] = None
    ) -> Tuple[GenericDataLoader, Dict[str, Any]]:
        """
        Preprocess the input data for synthetic generation.
        
        Args:
            data: Input DataFrame
            schema: Optional Schema object
            target_column: Optional target column for supervised learning
            sensitive_features: Optional list of sensitive columns
            
        Returns:
            Tuple of (GenericDataLoader, preprocessing_info)
        """
        try:
            if schema is None:
                schema = self.infer_schema(data)
            
            # Handle missing values
            for col in schema.numerical_columns:
                data[col] = data[col].fillna(data[col].mean())
            
            for col in schema.categorical_columns:
                data[col] = data[col].fillna(data[col].mode()[0])
            
            # Create dataloader
            loader = GenericDataLoader(
                data,
                sensitive_features=sensitive_features or [],
                target_column=target_column
            )
            
            preprocessing_info = {
                "schema": {
                    "categorical_columns": schema.categorical_columns,
                    "numerical_columns": schema.numerical_columns,
                    "datetime_columns": schema.datetime_columns
                },
                "target_column": target_column,
                "sensitive_features": sensitive_features,
                "n_samples": len(data)
            }
            
            return loader, preprocessing_info
            
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            bool indicating if data is valid
        """
        try:
            # Check if data is empty
            if data.empty:
                raise ValueError("Data is empty")
            
            # Check for all missing columns
            if data.columns.empty:
                raise ValueError("No columns in data")
            
            # Check for too many missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.5:
                raise ValueError("Too many missing values (>50%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
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