"""
Synthetic Generator Service

This module provides functionality for generating synthetic data using various models
with differential privacy guarantees. It supports multiple synthetic data generation
methods and provides privacy and utility metrics evaluation.

Classes:
    SyntheticGenerator: Main class for synthetic data generation and evaluation.
"""

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval import evaluate_privacy, evaluate_performance
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class SyntheticGenerator:
    """
    A class for generating synthetic data with privacy guarantees.
    
    This class provides methods to:
    - Generate synthetic data using various models
    - Evaluate privacy metrics
    - Calculate utility metrics
    - Manage model parameters and configurations
    
    Attributes:
        epsilon (float): Privacy parameter epsilon for differential privacy
        delta (float): Privacy parameter delta for differential privacy
        model_type (str): Type of synthetic data generation model
        model_params (Dict): Additional model parameters
    """
    
    def __init__(
        self,
        epsilon: float = settings.DEFAULT_EPSILON,
        delta: float = settings.DEFAULT_DELTA,
        model_type: str = settings.DEFAULT_MODEL
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            epsilon: Differential privacy parameter
            delta: Differential privacy relaxation parameter
            model_type: Type of synthetic data model to use
        """
        if model_type not in settings.SUPPORTED_MODELS:
            raise ValueError(f"Model type {model_type} not supported. Choose from {settings.SUPPORTED_MODELS}")
            
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
            
    def generate_synthetic_data(
        self,
        data_loader: GenericDataLoader,
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic data using the specified model.
        
        Args:
            data_loader: GenericDataLoader instance
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Dictionary containing synthetic data and evaluation metrics
        """
        try:
            # Get the plugin
            plugin = self.plugins.get(
                self.model_type,
                epsilon=self.epsilon,
                delta=self.delta
            )
            
            # Train the model
            plugin.fit(data_loader)
            
            # Generate synthetic data
            if n_samples is None:
                n_samples = len(data_loader)
                
            synthetic_data = plugin.generate(count=n_samples)
            
            # Convert to DataFrame
            synthetic_df = pd.DataFrame(
                synthetic_data,
                columns=data_loader.columns
            )
            
            # Evaluate the synthetic data
            privacy_metrics = self._evaluate_privacy(data_loader, synthetic_data)
            utility_metrics = self._evaluate_utility(data_loader, synthetic_data)
            
            return {
                "synthetic_data": synthetic_df,
                "privacy_metrics": privacy_metrics,
                "utility_metrics": utility_metrics,
                "generation_config": {
                    "model_type": self.model_type,
                    "epsilon": self.epsilon,
                    "delta": self.delta,
                    "n_samples": n_samples
                }
            }
            
        except Exception as e:
            logger.error(f"Error in generating synthetic data: {str(e)}")
            raise
            
    def _evaluate_privacy(
        self,
        real_data: GenericDataLoader,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate privacy metrics of synthetic data.
        
        Args:
            real_data: Original data loader
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary of privacy metrics
        """
        try:
            privacy_metrics = evaluate_privacy(real_data, synthetic_data)
            
            # Extract relevant metrics
            metrics = {
                "epsilon_score": privacy_metrics.get("epsilon", None),
                "delta_score": privacy_metrics.get("delta", None),
                "membership_disclosure_score": privacy_metrics.get("membership_disclosure", None),
                "attribute_disclosure_score": privacy_metrics.get("attribute_disclosure", None)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in privacy evaluation: {str(e)}")
            raise
            
    def _evaluate_utility(
        self,
        real_data: GenericDataLoader,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate utility metrics of synthetic data.
        
        Args:
            real_data: Original data loader
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary of utility metrics
        """
        try:
            utility_metrics = evaluate_performance(real_data, synthetic_data)
            
            # Extract relevant metrics
            metrics = {
                "statistical_similarity": utility_metrics.get("statistical_similarity", None),
                "feature_correlation": utility_metrics.get("correlation", None),
                "predictive_score": utility_metrics.get("predictive_score", None),
                "distribution_similarity": utility_metrics.get("distribution", None)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in utility evaluation: {str(e)}")
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