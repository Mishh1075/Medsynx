from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

class MedicalMetrics:
    """Custom metrics for evaluating medical data quality"""
    
    @staticmethod
    def evaluate_disease_distribution(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        disease_columns: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the preservation of disease label distributions
        
        Args:
            original: Original data
            synthetic: Synthetic data
            disease_columns: Columns containing disease labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for col in disease_columns:
            # Chi-square test for distribution similarity
            orig_counts = original[col].value_counts()
            syn_counts = synthetic[col].value_counts()
            
            # Align categories
            all_categories = sorted(set(orig_counts.index) | set(syn_counts.index))
            orig_counts = orig_counts.reindex(all_categories).fillna(0)
            syn_counts = syn_counts.reindex(all_categories).fillna(0)
            
            chi2, p_value = chi2_contingency(
                [orig_counts.values, syn_counts.values]
            )[:2]
            
            metrics[f"{col}_chi2"] = float(chi2)
            metrics[f"{col}_p_value"] = float(p_value)
            
            # Jensen-Shannon divergence
            orig_prob = orig_counts / orig_counts.sum()
            syn_prob = syn_counts / syn_counts.sum()
            
            m = 0.5 * (orig_prob + syn_prob)
            js_div = 0.5 * (
                np.sum(orig_prob * np.log(orig_prob / m)) +
                np.sum(syn_prob * np.log(syn_prob / m))
            )
            
            metrics[f"{col}_js_divergence"] = float(js_div)
        
        return metrics
    
    @staticmethod
    def evaluate_comorbidity_patterns(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        disease_columns: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the preservation of disease comorbidity patterns
        
        Args:
            original: Original data
            synthetic: Synthetic data
            disease_columns: Columns containing disease labels
            
        Returns:
            Dictionary of metrics
        """
        # Calculate comorbidity matrices
        orig_comorb = original[disease_columns].corr()
        syn_comorb = synthetic[disease_columns].corr()
        
        # Frobenius norm of difference
        frob_norm = np.linalg.norm(orig_comorb - syn_comorb)
        
        # Calculate pairwise comorbidity preservation
        pair_metrics = {}
        for i, disease1 in enumerate(disease_columns):
            for j, disease2 in enumerate(disease_columns[i+1:], i+1):
                orig_joint = pd.crosstab(
                    original[disease1],
                    original[disease2],
                    normalize='all'
                )
                syn_joint = pd.crosstab(
                    synthetic[disease1],
                    synthetic[disease2],
                    normalize='all'
                )
                
                # Align categories
                all_rows = sorted(set(orig_joint.index) | set(syn_joint.index))
                all_cols = sorted(set(orig_joint.columns) | set(syn_joint.columns))
                
                orig_joint = orig_joint.reindex(
                    index=all_rows,
                    columns=all_cols,
                    fill_value=0
                )
                syn_joint = syn_joint.reindex(
                    index=all_rows,
                    columns=all_cols,
                    fill_value=0
                )
                
                # Chi-square test
                chi2, p_value = chi2_contingency(
                    [orig_joint.values.flatten(),
                     syn_joint.values.flatten()]
                )[:2]
                
                pair_name = f"{disease1}_{disease2}"
                pair_metrics[f"{pair_name}_chi2"] = float(chi2)
                pair_metrics[f"{pair_name}_p_value"] = float(p_value)
        
        return {
            "comorbidity_matrix_difference": float(frob_norm),
            **pair_metrics
        }
    
    @staticmethod
    def evaluate_temporal_patterns(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        event_date_column: str,
        event_type_column: str
    ) -> Dict[str, float]:
        """
        Evaluate the preservation of temporal patterns in medical events
        
        Args:
            original: Original data
            synthetic: Synthetic data
            event_date_column: Column containing event dates
            event_type_column: Column containing event types
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert dates to datetime
        original[event_date_column] = pd.to_datetime(original[event_date_column])
        synthetic[event_date_column] = pd.to_datetime(synthetic[event_date_column])
        
        # Event frequency over time
        orig_freq = original.groupby(event_type_column).size()
        syn_freq = synthetic.groupby(event_type_column).size()
        
        # Normalize frequencies
        orig_freq = orig_freq / orig_freq.sum()
        syn_freq = syn_freq / syn_freq.sum()
        
        # Calculate KL divergence
        all_events = sorted(set(orig_freq.index) | set(syn_freq.index))
        orig_freq = orig_freq.reindex(all_events).fillna(0)
        syn_freq = syn_freq.reindex(all_events).fillna(0)
        
        kl_div = np.sum(
            orig_freq * np.log(orig_freq / syn_freq)
        )
        metrics["event_frequency_kl_div"] = float(kl_div)
        
        # Event intervals
        for event_type in all_events:
            orig_intervals = np.diff(
                sorted(original[original[event_type_column] == event_type][event_date_column])
            ).astype('timedelta64[D]').astype(float)
            
            syn_intervals = np.diff(
                sorted(synthetic[synthetic[event_type_column] == event_type][event_date_column])
            ).astype('timedelta64[D]').astype(float)
            
            if len(orig_intervals) > 0 and len(syn_intervals) > 0:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = ks_2samp(orig_intervals, syn_intervals)
                
                metrics[f"{event_type}_interval_ks"] = float(ks_stat)
                metrics[f"{event_type}_interval_p"] = float(p_value)
        
        return metrics
    
    @staticmethod
    def evaluate_lab_value_distributions(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        lab_columns: List[str],
        normal_ranges: Optional[Dict[str, tuple]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the preservation of laboratory value distributions
        
        Args:
            original: Original data
            synthetic: Synthetic data
            lab_columns: Columns containing lab values
            normal_ranges: Dictionary mapping lab names to (min, max) normal ranges
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for col in lab_columns:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(
                original[col].dropna(),
                synthetic[col].dropna()
            )
            
            metrics[f"{col}_ks"] = float(ks_stat)
            metrics[f"{col}_p_value"] = float(p_value)
            
            # Compare moments
            for moment in ['mean', 'std', 'skew', 'kurt']:
                orig_stat = getattr(original[col], moment)()
                syn_stat = getattr(synthetic[col], moment)()
                
                metrics[f"{col}_{moment}_diff"] = float(abs(orig_stat - syn_stat))
            
            # Check normal range preservation if provided
            if normal_ranges and col in normal_ranges:
                min_val, max_val = normal_ranges[col]
                
                orig_in_range = (
                    (original[col] >= min_val) &
                    (original[col] <= max_val)
                ).mean()
                
                syn_in_range = (
                    (synthetic[col] >= min_val) &
                    (synthetic[col] <= max_val)
                ).mean()
                
                metrics[f"{col}_normal_range_diff"] = float(
                    abs(orig_in_range - syn_in_range)
                )
        
        return metrics
    
    @staticmethod
    def evaluate_demographic_balance(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        demographic_columns: List[str],
        target_column: str
    ) -> Dict[str, float]:
        """
        Evaluate demographic balance and fairness metrics
        
        Args:
            original: Original data
            synthetic: Synthetic data
            demographic_columns: Columns containing demographic information
            target_column: Target variable (e.g., disease diagnosis)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for col in demographic_columns:
            # Calculate demographic parity
            orig_parity = original.groupby(col)[target_column].mean()
            syn_parity = synthetic.groupby(col)[target_column].mean()
            
            # Align categories
            all_categories = sorted(set(orig_parity.index) | set(syn_parity.index))
            orig_parity = orig_parity.reindex(all_categories).fillna(0)
            syn_parity = syn_parity.reindex(all_categories).fillna(0)
            
            # Maximum parity difference
            max_diff = abs(orig_parity - syn_parity).max()
            metrics[f"{col}_parity_diff"] = float(max_diff)
            
            # Equal opportunity difference
            for category in all_categories:
                orig_tpr = (
                    original[original[col] == category][target_column].mean() /
                    original[target_column].mean()
                )
                syn_tpr = (
                    synthetic[synthetic[col] == category][target_column].mean() /
                    synthetic[target_column].mean()
                )
                
                metrics[f"{col}_{category}_opportunity_diff"] = float(
                    abs(orig_tpr - syn_tpr)
                )
        
        return metrics 