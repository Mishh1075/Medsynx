from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval import evaluate_privacy, evaluate_performance
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SyntheticGenerator:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, model_type: str = "dpgan"):
        """
        Initialize the synthetic data generator.
        
        Args:
            epsilon: Differential privacy parameter
            delta: Differential privacy relaxation parameter
            model_type: Type of synthetic data model to use (dpgan, pategan, etc.)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.model_type = model_type
        self.plugins = Plugins()
        
    def preprocess_data(self, data: pd.DataFrame) -> GenericDataLoader:
        """
        Preprocess the input data and create a GenericDataLoader.
        
        Args:
            data: Input DataFrame
            
        Returns:
            GenericDataLoader instance
        """
        try:
            loader = GenericDataLoader(
                data,
                sensitive_features=[],  # Add sensitive columns if needed
                target_column=None  # Add target column if needed
            )
            return loader
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            raise
            
    def generate_synthetic_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate synthetic data using the specified model.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing synthetic data and evaluation metrics
        """
        try:
            # Preprocess data
            loader = self.preprocess_data(data)
            
            # Get the plugin
            plugin = self.plugins.get(
                self.model_type,
                epsilon=self.epsilon,
                delta=self.delta
            )
            
            # Train the model
            plugin.fit(loader)
            
            # Generate synthetic data
            synthetic_data = plugin.generate(count=len(data))
            
            # Convert to DataFrame
            synthetic_df = pd.DataFrame(
                synthetic_data,
                columns=data.columns
            )
            
            # Evaluate the synthetic data
            privacy_metrics = evaluate_privacy(loader, synthetic_data)
            performance_metrics = evaluate_performance(loader, synthetic_data)
            
            return {
                "synthetic_data": synthetic_df,
                "privacy_metrics": privacy_metrics,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in generating synthetic data: {str(e)}")
            raise
            
    def evaluate_synthetic_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the quality of synthetic data.
        
        Args:
            real_data: Original DataFrame
            synthetic_data: Generated synthetic DataFrame
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            real_loader = self.preprocess_data(real_data)
            
            privacy_metrics = evaluate_privacy(real_loader, synthetic_data)
            performance_metrics = evaluate_performance(real_loader, synthetic_data)
            
            return {
                "privacy_metrics": privacy_metrics,
                "performance_metrics": performance_metrics
            }
        except Exception as e:
            logger.error(f"Error in evaluating synthetic data: {str(e)}")
            raise 