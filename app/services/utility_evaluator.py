from typing import Dict, Any, List
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class UtilityEvaluator:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def evaluate_utility(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive utility evaluation of synthetic data
        """
        metrics = {
            "statistical_similarity": self._evaluate_statistical_similarity(original_data, synthetic_data),
            "correlation_similarity": self._evaluate_correlation_similarity(original_data, synthetic_data),
            "ml_utility": self._evaluate_ml_utility(original_data, synthetic_data),
            "distribution_tests": self._perform_distribution_tests(original_data, synthetic_data)
        }
        return metrics
        
    def _evaluate_statistical_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistical similarity metrics
        """
        metrics = {}
        
        # Basic statistics comparison
        for column in original.columns:
            if np.issubdtype(original[column].dtype, np.number):
                orig_stats = original[column].describe()
                syn_stats = synthetic[column].describe()
                
                metrics[f"{column}_mean_difference"] = abs(orig_stats["mean"] - syn_stats["mean"])
                metrics[f"{column}_std_difference"] = abs(orig_stats["std"] - syn_stats["std"])
                metrics[f"{column}_percentile_difference"] = np.mean([
                    abs(orig_stats[str(p)] - syn_stats[str(p)])
                    for p in ["25%", "50%", "75%"]
                ])
        
        return metrics
        
    def _evaluate_correlation_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Compare correlation matrices between original and synthetic data
        """
        # Calculate correlation matrices
        orig_corr = original.corr().values
        syn_corr = synthetic.corr().values
        
        # Calculate Frobenius norm of difference
        correlation_distance = np.linalg.norm(orig_corr - syn_corr)
        
        # Calculate mutual information scores
        mutual_info = {}
        for col in original.columns:
            if original[col].dtype in [np.int64, np.float64]:
                mutual_info[col] = mutual_info_score(
                    original[col],
                    synthetic[col]
                )
        
        return {
            "correlation_matrix_distance": float(correlation_distance),
            "mutual_information_scores": mutual_info
        }
        
    def _evaluate_ml_utility(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate machine learning utility of synthetic data
        """
        results = {}
        
        # For each numeric column, try to predict it using other columns
        for target_col in original.select_dtypes(include=[np.number]).columns:
            # Prepare feature sets
            feature_cols = [col for col in original.columns if col != target_col]
            
            # Split original data
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                original[feature_cols], original[target_col], test_size=0.2
            )
            
            # Train on original, test on original
            if len(np.unique(y_train_orig)) < 5:  # Classification task
                model_orig = RandomForestClassifier(n_estimators=100)
            else:  # Regression task
                model_orig = RandomForestRegressor(n_estimators=100)
            
            model_orig.fit(X_train_orig, y_train_orig)
            orig_score = model_orig.score(X_test_orig, y_test_orig)
            
            # Train on synthetic, test on original
            if len(np.unique(y_train_orig)) < 5:
                model_syn = RandomForestClassifier(n_estimators=100)
            else:
                model_syn = RandomForestRegressor(n_estimators=100)
            
            model_syn.fit(synthetic[feature_cols], synthetic[target_col])
            syn_score = model_syn.score(X_test_orig, y_test_orig)
            
            results[f"{target_col}_ml_utility"] = syn_score / orig_score if orig_score > 0 else 0
            
        return results
        
    def _perform_distribution_tests(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical distribution tests
        """
        results = {}
        
        for column in original.columns:
            if np.issubdtype(original[column].dtype, np.number):
                # Kolmogorov-Smirnov test
                ks_statistic, ks_pvalue = stats.ks_2samp(
                    original[column].values,
                    synthetic[column].values
                )
                
                # Anderson-Darling test
                ad_statistic = stats.anderson_ksamp(
                    [original[column].values, synthetic[column].values]
                ).statistic
                
                # Chi-square test for binned data
                hist_orig, bins = np.histogram(original[column], bins='auto')
                hist_syn, _ = np.histogram(synthetic[column], bins=bins)
                chi2_statistic, chi2_pvalue = stats.chisquare(hist_syn, hist_orig)
                
                results[column] = {
                    "ks_statistic": float(ks_statistic),
                    "ks_pvalue": float(ks_pvalue),
                    "anderson_darling_statistic": float(ad_statistic),
                    "chi2_statistic": float(chi2_statistic),
                    "chi2_pvalue": float(chi2_pvalue)
                }
                
        return results 