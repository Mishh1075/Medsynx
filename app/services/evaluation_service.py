import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from app.core.config import settings
import logging
import shap
import lime
import lime.lime_tabular
from alibi.explainers import KernelShap
import io
import base64

logger = logging.getLogger(__name__)

class EvaluationService:
    def __init__(self):
        """Initialize the evaluation service."""
        self.metrics_history = []
        
    def evaluate_synthetic_data(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: str = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of synthetic data quality.
        
        Args:
            real_data: Original DataFrame
            synthetic_data: Generated synthetic DataFrame
            target_column: Target column for ML metrics
            
        Returns:
            Dictionary containing evaluation metrics and visualizations
        """
        try:
            # Basic statistical metrics
            statistical_metrics = self._calculate_statistical_metrics(real_data, synthetic_data)
            
            # Privacy metrics
            privacy_metrics = self._calculate_privacy_metrics(real_data, synthetic_data)
            
            # ML utility metrics if target column is provided
            ml_metrics = {}
            if target_column and target_column in real_data.columns:
                ml_metrics = self._calculate_ml_metrics(
                    real_data, synthetic_data, target_column
                )
            
            # Generate visualizations
            visualizations = self._generate_visualizations(real_data, synthetic_data)
            
            # Feature importance and explanations
            feature_insights = self._analyze_feature_importance(
                real_data, synthetic_data, target_column
            )
            
            # Combine all metrics
            evaluation_results = {
                "statistical_metrics": statistical_metrics,
                "privacy_metrics": privacy_metrics,
                "ml_metrics": ml_metrics,
                "visualizations": visualizations,
                "feature_insights": feature_insights,
                "overall_quality_score": self._calculate_overall_score(
                    statistical_metrics,
                    privacy_metrics,
                    ml_metrics
                )
            }
            
            # Store metrics history
            self.metrics_history.append(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in synthetic data evaluation: {str(e)}")
            raise
            
    def _calculate_statistical_metrics(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate statistical similarity metrics."""
        try:
            metrics = {}
            
            # Column-wise statistics
            for column in real.columns:
                if pd.api.types.is_numeric_dtype(real[column]):
                    # KS test for continuous variables
                    from scipy.stats import ks_2samp
                    ks_stat, _ = ks_2samp(real[column], synthetic[column])
                    metrics[f"ks_test_{column}"] = ks_stat
                    
                    # Jensen-Shannon divergence
                    from scipy.spatial.distance import jensenshannon
                    real_hist, _ = np.histogram(real[column], bins=50, density=True)
                    syn_hist, _ = np.histogram(synthetic[column], bins=50, density=True)
                    js_div = jensenshannon(real_hist, syn_hist)
                    metrics[f"js_divergence_{column}"] = js_div
                    
                else:
                    # Chi-square test for categorical variables
                    from scipy.stats import chi2_contingency
                    real_counts = real[column].value_counts()
                    syn_counts = synthetic[column].value_counts()
                    chi2, _ = chi2_contingency(
                        pd.concat([real_counts, syn_counts], axis=1).fillna(0)
                    )[:2]
                    metrics[f"chi2_test_{column}"] = chi2
                    
            # Correlation matrix difference
            real_corr = real.corr()
            syn_corr = synthetic.corr()
            metrics["correlation_difference"] = np.mean(np.abs(real_corr - syn_corr))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating statistical metrics: {str(e)}")
            raise
            
    def _calculate_privacy_metrics(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate privacy preservation metrics."""
        try:
            metrics = {}
            
            # k-anonymity estimation
            metrics["k_anonymity"] = self._estimate_k_anonymity(synthetic)
            
            # l-diversity estimation
            metrics["l_diversity"] = self._estimate_l_diversity(synthetic)
            
            # Membership inference risk
            metrics["membership_inference_risk"] = self._membership_inference_test(
                real, synthetic
            )
            
            # Attribute disclosure risk
            metrics["attribute_disclosure_risk"] = self._attribute_disclosure_test(
                real, synthetic
            )
            
            # Nearest neighbor distance ratio
            metrics["nn_distance_ratio"] = self._calculate_nn_distance_ratio(
                real, synthetic
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            raise
            
    def _calculate_ml_metrics(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_column: str
    ) -> Dict[str, float]:
        """Calculate machine learning utility metrics."""
        try:
            metrics = {}
            
            # Split data
            X_real = real.drop(columns=[target_column])
            y_real = real[target_column]
            X_syn = synthetic.drop(columns=[target_column])
            y_syn = synthetic[target_column]
            
            # Train on real, test on synthetic
            X_train, X_test, y_train, y_test = train_test_split(
                X_real, y_real, test_size=0.2
            )
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            
            # Calculate metrics
            y_pred = clf.predict(X_syn)
            metrics["accuracy"] = accuracy_score(y_syn, y_pred)
            metrics["precision"] = precision_score(
                y_syn, y_pred, average='weighted'
            )
            metrics["recall"] = recall_score(y_syn, y_pred, average='weighted')
            metrics["f1"] = f1_score(y_syn, y_pred, average='weighted')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ML metrics: {str(e)}")
            raise
            
    def _generate_visualizations(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame
    ) -> Dict[str, str]:
        """Generate comparison visualizations."""
        try:
            visualizations = {}
            
            # Distribution plots for numeric columns
            for column in real.select_dtypes(include=[np.number]).columns:
                plt.figure(figsize=(10, 6))
                plt.hist(real[column], alpha=0.5, label='Real')
                plt.hist(synthetic[column], alpha=0.5, label='Synthetic')
                plt.title(f'Distribution Comparison - {column}')
                plt.legend()
                
                # Convert plot to base64 string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                visualizations[f"dist_plot_{column}"] = base64.b64encode(
                    buf.read()
                ).decode()
                plt.close()
                
            # Correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                real.corr() - synthetic.corr(),
                cmap='RdBu',
                center=0
            )
            plt.title('Correlation Difference Heatmap')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            visualizations["correlation_diff"] = base64.b64encode(
                buf.read()
            ).decode()
            plt.close()
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def _analyze_feature_importance(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_column: str = None
    ) -> Dict[str, Any]:
        """Analyze feature importance and generate explanations."""
        try:
            insights = {}
            
            if target_column:
                # SHAP analysis
                X_real = real.drop(columns=[target_column])
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X_real, real[target_column])
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_real)
                
                # Feature importance based on SHAP
                feature_importance = pd.DataFrame({
                    'feature': X_real.columns,
                    'importance': np.abs(shap_values).mean(0)
                })
                insights["shap_importance"] = feature_importance.to_dict()
                
                # LIME explanation for a sample
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_real.values,
                    feature_names=X_real.columns,
                    class_names=[str(c) for c in model.classes_]
                )
                exp = explainer.explain_instance(
                    X_real.iloc[0].values,
                    model.predict_proba
                )
                insights["lime_explanation"] = exp.as_list()
                
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
            
    def _calculate_overall_score(
        self,
        statistical_metrics: Dict[str, float],
        privacy_metrics: Dict[str, float],
        ml_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall synthetic data quality score."""
        try:
            # Weights for different metric categories
            weights = {
                "statistical": 0.3,
                "privacy": 0.4,
                "ml": 0.3
            }
            
            # Calculate category scores
            statistical_score = 1.0 - np.mean([
                v for v in statistical_metrics.values() if isinstance(v, (int, float))
            ])
            
            privacy_score = np.mean([
                v for v in privacy_metrics.values() if isinstance(v, (int, float))
            ])
            
            ml_score = np.mean([
                v for v in ml_metrics.values() if isinstance(v, (int, float))
            ]) if ml_metrics else 0.5
            
            # Calculate weighted average
            overall_score = (
                weights["statistical"] * statistical_score +
                weights["privacy"] * privacy_score +
                weights["ml"] * ml_score
            )
            
            return float(overall_score)
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {str(e)}")
            raise 