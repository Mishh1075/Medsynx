import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)

class PrivacyEvaluator:
    """Comprehensive privacy evaluation for synthetic data."""
    
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_privacy(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate privacy metrics for synthetic data.
        
        Args:
            real_data: Original DataFrame
            synthetic_data: Generated synthetic DataFrame
            
        Returns:
            Dictionary containing privacy metrics
        """
        try:
            metrics = {
                # Basic Privacy Metrics
                'identifiability_score': self._calculate_identifiability(real_data, synthetic_data),
                'attribute_disclosure_risk': self._calculate_attribute_disclosure(real_data, synthetic_data),
                'membership_inference_risk': self._calculate_membership_inference(real_data, synthetic_data),
                
                # Statistical Privacy Metrics
                'distance_to_closest_record': self._calculate_distance_to_closest(real_data, synthetic_data),
                'duplicate_records': self._calculate_duplicates(synthetic_data),
                
                # Distribution Privacy
                'distribution_similarity': self._calculate_distribution_similarity(real_data, synthetic_data),
                
                # Advanced Metrics
                'k_anonymity_estimate': self._estimate_k_anonymity(synthetic_data),
                'l_diversity_estimate': self._estimate_l_diversity(synthetic_data)
            }
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error in privacy evaluation: {str(e)}")
            raise
            
    def _calculate_identifiability(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate identifiability risk score."""
        try:
            # Normalize data
            real_norm = (real - real.mean()) / real.std()
            syn_norm = (synthetic - synthetic.mean()) / synthetic.std()
            
            # Calculate minimum distances between real and synthetic records
            min_distances = []
            for _, real_row in real_norm.iterrows():
                distances = np.linalg.norm(syn_norm - real_row, axis=1)
                min_distances.append(np.min(distances))
                
            # Convert to risk score (0-1)
            risk_score = 1 / (1 + np.mean(min_distances))
            return float(risk_score)
        except Exception as e:
            logger.warning(f"Error calculating identifiability: {str(e)}")
            return -1.0
            
    def _calculate_attribute_disclosure(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate attribute disclosure risk."""
        try:
            # Calculate correlation matrices
            real_corr = real.corr().fillna(0)
            syn_corr = synthetic.corr().fillna(0)
            
            # Calculate difference in correlations
            correlation_diff = np.abs(real_corr - syn_corr).mean().mean()
            
            # Convert to risk score (0-1)
            risk_score = 1 - np.exp(-correlation_diff)
            return float(risk_score)
        except Exception as e:
            logger.warning(f"Error calculating attribute disclosure: {str(e)}")
            return -1.0
            
    def _calculate_membership_inference(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Estimate membership inference attack risk."""
        try:
            # Prepare data for membership inference
            real_sample = real.sample(n=min(len(real), 1000), random_state=42)
            syn_sample = synthetic.sample(n=min(len(synthetic), 1000), random_state=42)
            
            # Create training data with labels (0 for real, 1 for synthetic)
            X = pd.concat([real_sample, syn_sample])
            y = np.concatenate([np.zeros(len(real_sample)), np.ones(len(syn_sample))])
            
            # Split data and train classifier
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf = RandomForestClassifier(n_estimators=10)
            clf.fit(X_train, y_train)
            
            # Calculate accuracy (risk score)
            y_pred = clf.predict(X_test)
            risk_score = accuracy_score(y_test, y_pred)
            
            return float(risk_score)
        except Exception as e:
            logger.warning(f"Error calculating membership inference: {str(e)}")
            return -1.0
            
    def _calculate_distance_to_closest(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate average distance to closest record."""
        try:
            # Normalize data
            real_norm = (real - real.mean()) / real.std()
            syn_norm = (synthetic - synthetic.mean()) / synthetic.std()
            
            # Calculate distances
            distances = []
            for _, syn_row in syn_norm.iterrows():
                dist = np.linalg.norm(real_norm - syn_row, axis=1)
                distances.append(np.min(dist))
                
            return float(np.mean(distances))
        except Exception as e:
            logger.warning(f"Error calculating distance to closest: {str(e)}")
            return -1.0
            
    def _calculate_duplicates(self, synthetic: pd.DataFrame) -> float:
        """Calculate proportion of duplicate records."""
        try:
            total_records = len(synthetic)
            unique_records = len(synthetic.drop_duplicates())
            return float(1 - (unique_records / total_records))
        except Exception as e:
            logger.warning(f"Error calculating duplicates: {str(e)}")
            return -1.0
            
    def _calculate_distribution_similarity(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate distribution similarity score."""
        try:
            similarities = []
            for column in real.columns:
                if real[column].dtype in ['int64', 'float64']:
                    # For numerical columns, compare distributions
                    real_hist = np.histogram(real[column], bins=20)[0]
                    syn_hist = np.histogram(synthetic[column], bins=20)[0]
                    similarity = 1 - np.mean(np.abs(real_hist/sum(real_hist) - syn_hist/sum(syn_hist)))
                    similarities.append(similarity)
                else:
                    # For categorical columns, compare value frequencies
                    real_freq = real[column].value_counts(normalize=True)
                    syn_freq = synthetic[column].value_counts(normalize=True)
                    common_categories = set(real_freq.index) & set(syn_freq.index)
                    if common_categories:
                        similarity = 1 - np.mean([abs(real_freq.get(cat, 0) - syn_freq.get(cat, 0)) 
                                               for cat in common_categories])
                        similarities.append(similarity)
                        
            return float(np.mean(similarities))
        except Exception as e:
            logger.warning(f"Error calculating distribution similarity: {str(e)}")
            return -1.0
            
    def _estimate_k_anonymity(self, synthetic: pd.DataFrame) -> float:
        """Estimate k-anonymity of synthetic data."""
        try:
            # Count frequency of unique combinations
            frequencies = synthetic.groupby(list(synthetic.columns)).size()
            k = frequencies.min()  # Minimum group size is k
            
            # Normalize to 0-1 range
            k_norm = 1 - (1 / (1 + k))
            return float(k_norm)
        except Exception as e:
            logger.warning(f"Error estimating k-anonymity: {str(e)}")
            return -1.0
            
    def _estimate_l_diversity(self, synthetic: pd.DataFrame) -> float:
        """Estimate l-diversity of synthetic data."""
        try:
            # Consider all columns as quasi-identifiers except the last one
            quasi_identifiers = synthetic.columns[:-1]
            sensitive_attr = synthetic.columns[-1]
            
            # Group by quasi-identifiers and count unique values in sensitive attribute
            l_values = synthetic.groupby(list(quasi_identifiers))[sensitive_attr].nunique()
            l = l_values.min()  # Minimum number of distinct values is l
            
            # Normalize to 0-1 range
            l_norm = 1 - (1 / (1 + l))
            return float(l_norm)
        except Exception as e:
            logger.warning(f"Error estimating l-diversity: {str(e)}")
            return -1.0
            
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