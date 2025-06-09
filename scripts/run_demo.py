import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.synthetic_generator import SyntheticGenerator
from app.services.privacy_evaluator import PrivacyEvaluator
from app.services.utility_evaluator import UtilityEvaluator
from app.services.medical_evaluator import MedicalEvaluator
from app.services.image_generator import MedicalImageGenerator

def run_tabular_demo():
    """Run tabular data synthesis demo"""
    print("1. Loading sample data...")
    demographics = pd.read_csv('sample_data/demographics.csv')
    conditions = pd.read_csv('sample_data/conditions.csv')
    
    print("\n2. Configuring privacy parameters...")
    privacy_params = {
        'epsilon': 1.0,
        'delta': 1e-5,
        'noise_multiplier': 1.0,
        'num_samples': 1000
    }
    
    print("\n3. Generating synthetic data...")
    generator = SyntheticGenerator(
        epsilon=privacy_params['epsilon'],
        delta=privacy_params['delta'],
        noise_multiplier=privacy_params['noise_multiplier']
    )
    
    synthetic_demographics = generator.generate_synthetic_data(
        demographics,
        num_samples=privacy_params['num_samples']
    )['synthetic_data']
    
    print("\n4. Evaluating results...")
    # Privacy evaluation
    privacy_evaluator = PrivacyEvaluator()
    privacy_metrics = privacy_evaluator.evaluate_privacy(
        demographics.values,
        synthetic_demographics.values
    )
    
    # Utility evaluation
    utility_evaluator = UtilityEvaluator()
    utility_metrics = utility_evaluator.evaluate_utility(
        demographics,
        synthetic_demographics
    )
    
    # Medical evaluation
    medical_evaluator = MedicalEvaluator()
    medical_metrics = medical_evaluator.evaluate_medical_data(
        demographics,
        synthetic_demographics,
        categorical_columns=['RACE', 'ETHNICITY', 'GENDER']
    )
    
    print("\nPrivacy Metrics:")
    for metric, value in privacy_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nUtility Metrics:")
    for metric, value in utility_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n5. Saving results...")
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    synthetic_demographics.to_csv(output_dir / 'synthetic_demographics.csv', index=False)
    
    evaluation_results = {
        'privacy_metrics': privacy_metrics,
        'utility_metrics': utility_metrics,
        'medical_metrics': medical_metrics
    }
    
    import json
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)

def run_imaging_demo():
    """Run medical imaging synthesis demo"""
    print("\n6. Running medical imaging demo...")
    image_generator = MedicalImageGenerator()
    
    # Load sample image
    sample_image = image_generator.load_dicom('sample_data/images/sample.dcm')
    
    # Generate synthetic images
    synthetic_images = image_generator.generate_images(num_images=4)
    
    # Save results
    output_dir = Path('output/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for i, img in enumerate(synthetic_images):
        image_generator.save_dicom(
            img,
            output_dir / f'synthetic_image_{i}.dcm'
        )
    
    print(f"Synthetic images saved to {output_dir}")

def main():
    """Run complete demo workflow"""
    print("Starting MedSynx Demo...")
    print("=" * 50)
    
    # Run tabular data demo
    run_tabular_demo()
    
    # Run imaging demo
    run_imaging_demo()
    
    print("\nDemo completed! Results are saved in the output/ directory.")

if __name__ == "__main__":
    main() 