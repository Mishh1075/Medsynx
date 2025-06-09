from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from synthcity.metrics.eval import evaluate_privacy
from synthcity.metrics import eval_detection

class MedicalEvaluator:
    def __init__(self):
        self.label_encoders = {}
        
    def evaluate_medical_data(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_columns: List[str],
        diagnosis_column: str = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of medical data quality
        """
        metrics = {
            "privacy_metrics": self._evaluate_privacy(original_data, synthetic_data),
            "utility_metrics": self._evaluate_utility(original_data, synthetic_data),
            "statistical_metrics": self._evaluate_statistical(original_data, synthetic_data),
        }
        
        if diagnosis_column:
            metrics["diagnosis_metrics"] = self._evaluate_diagnosis(
                original_data,
                synthetic_data,
                diagnosis_column
            )
            
        return metrics
    
    def _evaluate_privacy(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate privacy metrics using synthcity
        """
        # Basic privacy metrics
        privacy_metrics = evaluate_privacy(
            original.values,
            synthetic.values,
            metadata={
                "continuous_columns": original.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": original.select_dtypes(include=['object']).columns.tolist()
            }
        )
        
        # Membership inference attack evaluation
        mia_score = eval_detection.membership_disclosure(
            original.values,
            synthetic.values
        )
        
        return {
            "identifiability_score": float(privacy_metrics["identifiability_score"]),
            "attribute_disclosure_score": float(privacy_metrics["attribute_disclosure"]),
            "membership_inference_auc": float(mia_score),
        }
    
    def _evaluate_utility(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate utility preservation
        """
        # Calculate correlation similarity
        orig_corr = original.corr().values
        syn_corr = synthetic.corr().values
        correlation_distance = np.linalg.norm(orig_corr - syn_corr)
        
        # Calculate basic statistics similarity
        stats_similarity = {}
        for col in original.columns:
            if np.issubdtype(original[col].dtype, np.number):
                orig_stats = original[col].describe()
                syn_stats = synthetic[col].describe()
                stats_similarity[col] = 1 - np.mean([
                    abs(orig_stats[stat] - syn_stats[stat]) / max(abs(orig_stats[stat]), 1e-10)
                    for stat in ['mean', 'std', '25%', '50%', '75%']
                ])
        
        return {
            "correlation_similarity": float(1 - correlation_distance / np.sqrt(len(orig_corr))),
            "statistical_similarity": float(np.mean(list(stats_similarity.values()))),
        }
    
    def _evaluate_statistical(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate statistical properties
        """
        metrics = {}
        
        # KL divergence for continuous variables
        for col in original.select_dtypes(include=[np.number]).columns:
            orig_hist, bins = np.histogram(original[col], bins=50, density=True)
            syn_hist, _ = np.histogram(synthetic[col], bins=bins, density=True)
            
            # Add small constant to avoid division by zero
            eps = 1e-10
            kl_div = np.sum(orig_hist * np.log((orig_hist + eps) / (syn_hist + eps)))
            metrics[f"{col}_kl_divergence"] = float(kl_div)
        
        # Chi-square test for categorical variables
        for col in original.select_dtypes(include=['object']).columns:
            orig_counts = original[col].value_counts()
            syn_counts = synthetic[col].value_counts()
            
            # Align categories
            all_categories = sorted(set(orig_counts.index) | set(syn_counts.index))
            orig_counts = orig_counts.reindex(all_categories).fillna(0)
            syn_counts = syn_counts.reindex(all_categories).fillna(0)
            
            chi2 = np.sum((orig_counts - syn_counts) ** 2 / (orig_counts + eps))
            metrics[f"{col}_chi2"] = float(chi2)
        
        return metrics
    
    def _evaluate_diagnosis(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        diagnosis_column: str
    ) -> Dict[str, float]:
        """
        Evaluate diagnosis-specific metrics
        """
        # Encode diagnosis labels
        if diagnosis_column not in self.label_encoders:
            self.label_encoders[diagnosis_column] = LabelEncoder()
            self.label_encoders[diagnosis_column].fit(
                pd.concat([original[diagnosis_column], synthetic[diagnosis_column]])
            )
        
        orig_labels = self.label_encoders[diagnosis_column].transform(original[diagnosis_column])
        syn_labels = self.label_encoders[diagnosis_column].transform(synthetic[diagnosis_column])
        
        # Calculate metrics
        metrics = {
            "diagnosis_f1_macro": float(f1_score(orig_labels, syn_labels, average='macro')),
            "diagnosis_f1_weighted": float(f1_score(orig_labels, syn_labels, average='weighted')),
            "diagnosis_precision": float(precision_score(orig_labels, syn_labels, average='weighted')),
            "diagnosis_recall": float(recall_score(orig_labels, syn_labels, average='weighted')),
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(orig_labels, syn_labels)
        metrics["diagnosis_confusion_matrix"] = cm.tolist()
        
        # Calculate per-class metrics
        classes = self.label_encoders[diagnosis_column].classes_
        for i, class_name in enumerate(classes):
            metrics[f"diagnosis_f1_{class_name}"] = float(
                f1_score(orig_labels == i, syn_labels == i)
            )
        
        return metrics 