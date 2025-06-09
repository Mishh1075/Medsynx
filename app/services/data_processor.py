"""
Data Processor Service

This module provides functionality for preprocessing and validating data before synthetic generation.
It handles data type inference, schema validation, missing value handling, and data transformation.

Classes:
    DataProcessor: Main class for data preprocessing and validation.
"""

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
    """
    A class for preprocessing and validating data before synthetic generation.
    
    This class provides methods to:
    - Validate input data format and quality
    - Infer data schema and types
    - Handle missing values
    - Transform data for model input
    - Create data loaders for synthetic generation
    
    Attributes:
        max_missing_ratio (float): Maximum allowed ratio of missing values (default: 0.3)
        min_rows (int): Minimum required number of rows (default: 10)
    """
    
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
        Infer data schema including column types and constraints.
        
        Args:
            data (pd.DataFrame): Input data to analyze
            
        Returns:
            Schema: Inferred schema object with column information
        """
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return Schema(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        sensitive_features: Optional[List[str]] = None
    ) -> Tuple[GenericDataLoader, Dict]:
        """
        Preprocess data for synthetic generation.
        
        This method:
        1. Validates input data
        2. Infers data schema
        3. Handles missing values
        4. Creates a data loader
        5. Returns preprocessing information
        
        Args:
            data (pd.DataFrame): Input data to preprocess
            target_column (str, optional): Name of target column for supervised learning
            sensitive_features (List[str], optional): List of sensitive feature columns
            
        Returns:
            Tuple[GenericDataLoader, Dict]: Tuple containing:
                - GenericDataLoader: Preprocessed data loader for synthetic generation
                - Dict: Information about preprocessing including schema and statistics
                
        Raises:
            ValueError: If data validation fails or column names are invalid
        """
        # Validate data
        self.validate_data(data)
        
        # Infer schema
        schema = self.infer_schema(data)
        
        # Handle missing values
        data = self._handle_missing_values(data, schema)
        
        # Create data loader
        loader = GenericDataLoader(
            data=data,
            target_column=target_column,
            sensitive_features=sensitive_features
        )
        
        # Prepare info dict
        info = {
            'schema': {
                'numerical_columns': schema.numerical_columns,
                'categorical_columns': schema.categorical_columns
            },
            'n_samples': len(data),
            'target_column': target_column,
            'sensitive_features': sensitive_features
        }
        
        return loader, info
    
    def _handle_missing_values(self, data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data with missing values
            schema (Schema): Data schema with column information
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        # Fill numerical missing values with median
        for col in schema.numerical_columns:
            data[col] = data[col].fillna(data[col].median())
            
        # Fill categorical missing values with mode
        for col in schema.categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0])
            
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for quality and format requirements.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
            
        Raises:
            ValueError: If data does not meet requirements
        """
        if len(data) < self.min_rows:
            raise ValueError(f"Data must have at least {self.min_rows} rows")
            
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > self.max_missing_ratio:
            raise ValueError(f"Data has too many missing values: {missing_ratio:.2%}")
            
        return True
    
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