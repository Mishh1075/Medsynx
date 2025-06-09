"""
Evaluation Service

This module provides comprehensive evaluation metrics for synthetic data,
including statistical tests, privacy assessments, and utility measurements.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from synthcity.metrics import Metrics
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.metrics.eval_privacy import PrivacyEvaluator
import logging

logger = logging.getLogger(__name__)

class EvaluationService:
    """
    Service for evaluating synthetic data quality, privacy, and utility.
    """
    
    def __init__(self):
        self.metrics = Metrics()
        self.privacy_evaluator = PrivacyEvaluator()
    
    def evaluate_all(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Generated synthetic dataset
            sensitive_features: List of sensitive columns
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        return {
            "statistical_metrics": self.evaluate_statistical(original_data, synthetic_data),
            "privacy_metrics": self.evaluate_privacy(original_data, synthetic_data, sensitive_features),
            "utility_metrics": self.evaluate_utility(original_data, synthetic_data),
            "ml_metrics": self.evaluate_ml_utility(original_data, synthetic_data)
        }
    
    def evaluate_statistical(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate statistical similarity between original and synthetic data.
        """
        stats_metrics = {}
        
        # Kolmogorov-Smirnov test for numerical columns
        for col in original_data.select_dtypes(include=['number']).columns:
            ks_stat, p_value = stats.ks_2samp(
                original_data[col].dropna(),
                synthetic_data[col].dropna()
            )
            stats_metrics[f"ks_test_{col}"] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value)
            }
        
        # Chi-square test for categorical columns
        for col in original_data.select_dtypes(include=['object', 'category']).columns:
            orig_counts = original_data[col].value_counts()
            synth_counts = synthetic_data[col].value_counts()
            
            # Align categories
            all_categories = list(set(orig_counts.index) | set(synth_counts.index))
            orig_counts = orig_counts.reindex(all_categories, fill_value=0)
            synth_counts = synth_counts.reindex(all_categories, fill_value=0)
            
            chi2_stat, p_value = stats.chisquare(
                synth_counts.values,
                orig_counts.values * (len(synthetic_data) / len(original_data))
            )
            stats_metrics[f"chi2_test_{col}"] = {
                "statistic": float(chi2_stat),
                "p_value": float(p_value)
            }
        
        # Correlation similarity
        orig_corr = original_data.corr().fillna(0)
        synth_corr = synthetic_data.corr().fillna(0)
        corr_diff = np.abs(orig_corr - synth_corr).mean().mean()
        stats_metrics["correlation_difference"] = float(corr_diff)
        
        return stats_metrics
    
    def evaluate_privacy(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate privacy preservation in synthetic data.
        """
        # Create data loaders
        original_loader = DataLoader(data=original_data)
        synthetic_loader = DataLoader(data=synthetic_data)
        
        privacy_metrics = {}
        
        # Basic privacy metrics from SynthCity
        basic_metrics = self.metrics.evaluate(
            original_loader,
            synthetic_loader,
            metrics=["privacy_metrics"]
        )
        privacy_metrics.update(basic_metrics)
        
        # Advanced privacy evaluation
        privacy_metrics.update(self._evaluate_advanced_privacy(
            original_data,
            synthetic_data,
            sensitive_features
        ))
        
        return privacy_metrics
    
    def _evaluate_advanced_privacy(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_features: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform advanced privacy evaluation including membership inference attacks.
        """
        metrics = {}
        
        # Membership inference attack
        mia_score = self.privacy_evaluator.membership_disclosure(
            original_data,
            synthetic_data
        )
        metrics["membership_inference_risk"] = float(mia_score)
        
        # Attribute disclosure risk for sensitive features
        if sensitive_features:
            for feature in sensitive_features:
                disclosure_risk = self.privacy_evaluator.attribute_disclosure(
                    original_data,
                    synthetic_data,
                    target_column=feature
                )
                metrics[f"attribute_disclosure_{feature}"] = float(disclosure_risk)
        
        # k-anonymity approximation
        metrics["k_anonymity_estimate"] = self._estimate_k_anonymity(synthetic_data)
        
        return metrics
    
    def _estimate_k_anonymity(self, data: pd.DataFrame) -> float:
        """
        Estimate k-anonymity level of the synthetic data.
        """
        # Convert data to string representation for grouping
        str_data = data.astype(str)
        
        # Count occurrences of each unique combination
        counts = str_data.groupby(list(str_data.columns)).size()
        
        # k-anonymity is the minimum group size
        return float(counts.min())
    
    def evaluate_utility(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate utility preservation in synthetic data.
        """
        # Create data loaders
        original_loader = DataLoader(data=original_data)
        synthetic_loader = DataLoader(data=synthetic_data)
        
        # Get comprehensive utility metrics from SynthCity
        utility_metrics = self.metrics.evaluate(
            original_loader,
            synthetic_loader,
            metrics=[
                "statistical_similarity",
                "feature_correlation",
                "data_mismatch",
                "performance_metrics"
            ]
        )
        
        # Add mutual information scores
        for col in original_data.columns:
            mi_score = mutual_info_score(
                original_data[col].fillna('missing'),
                synthetic_data[col].fillna('missing')
            )
            utility_metrics[f"mutual_info_{col}"] = float(mi_score)
        
        return utility_metrics
    
    def evaluate_ml_utility(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate machine learning utility of synthetic data.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        ml_metrics = {}
        
        # For each column, try to predict it using other columns
        for target_col in original_data.columns:
            try:
                # Prepare data
                X_orig = original_data.drop(columns=[target_col])
                X_synth = synthetic_data.drop(columns=[target_col])
                
                # Handle categorical features
                for col in X_orig.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X_orig[col] = le.fit_transform(X_orig[col].fillna('missing'))
                    X_synth[col] = le.transform(synthetic_data[col].fillna('missing'))
                
                # Handle target variable
                y_orig = original_data[target_col]
                y_synth = synthetic_data[target_col]
                if y_orig.dtype == 'object' or y_orig.dtype.name == 'category':
                    le = LabelEncoder()
                    y_orig = le.fit_transform(y_orig.fillna('missing'))
                    y_synth = le.transform(y_synth.fillna('missing'))
                
                # Train and evaluate on original data
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                orig_scores = cross_val_score(clf, X_orig, y_orig, cv=5)
                
                # Train on synthetic, test on original
                clf.fit(X_synth, y_synth)
                synth_score = clf.score(X_orig, y_orig)
                
                ml_metrics[f"ml_utility_{target_col}"] = {
                    "original_cv_score": float(orig_scores.mean()),
                    "synthetic_score": float(synth_score),
                    "score_ratio": float(synth_score / orig_scores.mean())
                }
                
            except Exception as e:
                logger.warning(f"Could not evaluate ML utility for {target_col}: {str(e)}")
                continue
        
        return ml_metrics 