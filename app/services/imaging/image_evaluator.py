import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import shap
import lime.lime_image
from alibi.explainers import KernelShap

class ImageEvaluator:
    """Evaluates synthetic medical images with enhanced metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.lime_explainer = lime.lime_image.LimeImageExplainer()
    
    def calculate_metrics(self, real_images: torch.Tensor, 
                         synthetic_images: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        real_np = real_images.cpu().numpy()
        synthetic_np = synthetic_images.cpu().numpy()
        
        metrics = {
            # Image Quality Metrics
            'ssim': self._calculate_ssim(real_np, synthetic_np),
            'psnr': self._calculate_psnr(real_np, synthetic_np),
            'wasserstein': self._calculate_wasserstein(real_np, synthetic_np),
            
            # Statistical Metrics
            'mean_diff': np.abs(np.mean(real_np) - np.mean(synthetic_np)),
            'std_diff': np.abs(np.std(real_np) - np.std(synthetic_np)),
            'histogram_intersection': self._calculate_histogram_intersection(real_np, synthetic_np),
            
            # Privacy Metrics
            'membership_inference_risk': self._membership_inference_risk(real_np, synthetic_np),
            'attribute_disclosure_risk': self._attribute_disclosure_risk(real_np, synthetic_np)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_ssim(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        ssim_values = []
        for i in range(len(real)):
            ssim_val = ssim(real[i], synthetic[i], data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    
    def _calculate_psnr(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        psnr_values = []
        for i in range(len(real)):
            psnr_val = psnr(real[i], synthetic[i], data_range=1.0)
            psnr_values.append(psnr_val)
        return np.mean(psnr_values)
    
    def _calculate_wasserstein(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """Calculate Wasserstein distance between distributions."""
        real_flat = real.reshape(-1)
        synthetic_flat = synthetic.reshape(-1)
        return wasserstein_distance(real_flat, synthetic_flat)
    
    def _calculate_histogram_intersection(self, real: np.ndarray, 
                                       synthetic: np.ndarray, 
                                       bins: int = 50) -> float:
        """Calculate histogram intersection similarity."""
        hist_real, _ = np.histogram(real, bins=bins, density=True)
        hist_syn, _ = np.histogram(synthetic, bins=bins, density=True)
        intersection = np.minimum(hist_real, hist_syn).sum()
        return intersection
    
    def _membership_inference_risk(self, real: np.ndarray, 
                                 synthetic: np.ndarray) -> float:
        """Estimate membership inference attack risk."""
        # Simplified membership inference risk estimation
        distances = np.mean(np.abs(real - synthetic.reshape(synthetic.shape[0], -1)), axis=1)
        threshold = np.percentile(distances, 95)
        risk = np.mean(distances > threshold)
        return float(risk)
    
    def _attribute_disclosure_risk(self, real: np.ndarray, 
                                 synthetic: np.ndarray) -> float:
        """Estimate attribute disclosure risk."""
        # Simplified attribute disclosure risk using correlation
        real_corr = np.corrcoef(real.reshape(real.shape[0], -1).T)
        syn_corr = np.corrcoef(synthetic.reshape(synthetic.shape[0], -1).T)
        risk = np.mean(np.abs(real_corr - syn_corr))
        return float(risk)
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.metrics_history:
            return {}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in self.metrics_history[0].keys():
            values = [m[metric] for m in self.metrics_history]
            avg_metrics[metric] = np.mean(values)
        
        # Generate visualizations
        figs = self._create_visualizations()
        
        report = {
            'average_metrics': avg_metrics,
            'visualizations': figs,
            'privacy_assessment': {
                'membership_inference_risk': avg_metrics['membership_inference_risk'],
                'attribute_disclosure_risk': avg_metrics['attribute_disclosure_risk']
            }
        }
        
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _create_visualizations(self) -> Dict[str, plt.Figure]:
        """Create visualization plots for the report."""
        figs = {}
        
        # Metrics over time
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in ['ssim', 'psnr', 'wasserstein']:
            values = [m[metric] for m in self.metrics_history]
            ax.plot(values, label=metric)
        ax.set_title('Quality Metrics Over Time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.legend()
        figs['metrics_over_time'] = fig
        
        # Privacy risk assessment
        fig, ax = plt.subplots(figsize=(8, 6))
        risks = [
            np.mean([m['membership_inference_risk'] for m in self.metrics_history]),
            np.mean([m['attribute_disclosure_risk'] for m in self.metrics_history])
        ]
        ax.bar(['Membership Inference', 'Attribute Disclosure'], risks)
        ax.set_title('Privacy Risk Assessment')
        ax.set_ylabel('Risk Score')
        figs['privacy_risks'] = fig
        
        return figs
    
    def _save_report(self, report: Dict[str, Any], save_path: str):
        """Save evaluation report to file."""
        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(f"{save_path}/metrics.csv", index=False)
        
        # Save visualizations
        for name, fig in report['visualizations'].items():
            fig.savefig(f"{save_path}/{name}.png")
            plt.close(fig)
        
        # Save summary report
        with open(f"{save_path}/summary.txt", 'w') as f:
            f.write("Synthetic Image Evaluation Report\n")
            f.write("===============================\n\n")
            f.write("Average Metrics:\n")
            for metric, value in report['average_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\nPrivacy Assessment:\n")
            for risk, value in report['privacy_assessment'].items():
                f.write(f"{risk}: {value:.4f}\n")