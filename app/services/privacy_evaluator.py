import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import tensorflow_privacy
from diffprivlib.mechanisms import Gaussian
from opacus.utils.batch_memory_manager import BatchMemoryManager

logger = logging.getLogger(__name__)

class PrivacyEvaluator:
    """Comprehensive privacy evaluation for synthetic data."""
    
    def __init__(self):
        self.metrics_history = []
        self.membership_inference_model = RandomForestClassifier()
        self.dp_mechanism = Gaussian()
        
    def evaluate_privacy(self, original_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive privacy evaluation of synthetic data
        """
        metrics = {
            "differential_privacy": self._evaluate_dp_guarantees(original_data, synthetic_data),
            "membership_inference": self._evaluate_membership_inference(original_data, synthetic_data),
            "attribute_disclosure": self._evaluate_attribute_disclosure(original_data, synthetic_data),
            "k_anonymity": self._calculate_k_anonymity(synthetic_data),
            "l_diversity": self._calculate_l_diversity(synthetic_data)
        }
        self.metrics_history.append(metrics)
        return metrics

    def _evaluate_dp_guarantees(self, original_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate epsilon and delta guarantees
        """
        # Calculate sensitivity
        sensitivity = np.max(np.abs(original_data - np.mean(original_data, axis=0)))
        
        # Calculate epsilon based on noise level and sensitivity
        noise_std = np.std(synthetic_data - np.mean(synthetic_data, axis=0))
        epsilon = sensitivity / noise_std
        
        # Calculate delta based on sample size and dimension
        n_samples = len(original_data)
        delta = 1.0 / (n_samples * np.log(n_samples))
        
        return {
            "epsilon": float(epsilon),
            "delta": float(delta)
        }

    def _evaluate_membership_inference(self, original_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Perform membership inference attack evaluation
        """
        # Prepare training data
        X_train = np.vstack([original_data, synthetic_data])
        y_train = np.hstack([np.ones(len(original_data)), np.zeros(len(synthetic_data))])
        
        # Split for testing
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        
        # Train membership inference model
        self.membership_inference_model.fit(X_train, y_train)
        
        # Evaluate attack success rate
        attack_success_rate = self.membership_inference_model.score(X_test, y_test)
        
        return {
            "attack_success_rate": float(attack_success_rate),
            "privacy_risk_score": 1.0 - float(attack_success_rate)
        }

    def _evaluate_attribute_disclosure(self, original_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate attribute disclosure risk
        """
        # Calculate distance between closest records
        min_distances = []
        for orig_record in original_data:
            distances = np.linalg.norm(synthetic_data - orig_record, axis=1)
            min_distances.append(np.min(distances))
        
        # Calculate disclosure risk metrics
        avg_min_distance = np.mean(min_distances)
        max_disclosure_risk = 1.0 / (1.0 + np.min(min_distances))
        
        return {
            "average_minimum_distance": float(avg_min_distance),
            "maximum_disclosure_risk": float(max_disclosure_risk)
        }

    def _calculate_k_anonymity(self, data: np.ndarray) -> Dict[str, int]:
        """
        Calculate k-anonymity of the synthetic data
        """
        # Convert to string representation for exact matching
        records = [tuple(row) for row in data]
        counts = {}
        for record in records:
            counts[record] = counts.get(record, 0) + 1
        
        k_anonymity = min(counts.values())
        return {
            "k_anonymity": int(k_anonymity)
        }

    def _calculate_l_diversity(self, data: np.ndarray) -> Dict[str, int]:
        """
        Calculate l-diversity of the synthetic data
        """
        # Assume last column is sensitive attribute
        sensitive_values = data[:, -1]
        groups = {}
        
        # Group by quasi-identifiers (all columns except last)
        for i, record in enumerate(data[:, :-1]):
            key = tuple(record)
            if key not in groups:
                groups[key] = set()
            groups[key].add(sensitive_values[i])
        
        # Calculate minimum number of distinct sensitive values
        l_diversity = min(len(group) for group in groups.values())
        return {
            "l_diversity": int(l_diversity)
        }

    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        if not self.metrics_history:
            return {}
            
        latest_metrics = self.metrics_history[-1]
        
        report = {
            'privacy_scores': latest_metrics,
            'risk_assessment': {
                'overall_risk': np.mean(list(latest_metrics.values())),
                'high_risk_metrics': [
                    metric for metric, value in latest_metrics.items()
                    if value > 0.7
                ],
                'low_risk_metrics': [
                    metric for metric, value in latest_metrics.items()
                    if value < 0.3
                ]
            },
            'recommendations': self._generate_recommendations(latest_metrics)
        }
        
        return report
        
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate privacy enhancement recommendations based on metrics."""
        recommendations = []
        
        if metrics['identifiability_score'] > 0.7:
            recommendations.append(
                "High identifiability risk detected. Consider increasing privacy budget (ε)."
            )
            
        if metrics['attribute_disclosure_risk'] > 0.6:
            recommendations.append(
                "Significant attribute disclosure risk. Consider implementing additional "
                "noise in sensitive columns."
            )
            
        if metrics['membership_inference_risk'] > 0.65:
            recommendations.append(
                "Elevated membership inference risk. Consider reducing model memorization "
                "through increased regularization."
            )
            
        if metrics['k_anonymity_estimate'] < 0.3:
            recommendations.append(
                "Low k-anonymity detected. Consider grouping or generalizing "
                "quasi-identifier attributes."
            )
            
        return recommendations 