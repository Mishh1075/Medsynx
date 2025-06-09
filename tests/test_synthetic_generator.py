import pytest
import pandas as pd
import numpy as np
from app.services.synthetic_generator import SyntheticGenerator
from app.services.data_processor import DataProcessor
from app.core.config import settings

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.choice(['<=50K', '>50K'], 100),
        'education': np.random.choice(['HS', 'College', 'Masters'], 100),
        'occupation': np.random.choice(['Tech', 'Sales', 'Other'], 100)
    })

@pytest.fixture
def data_processor():
    """Create data processor instance"""
    return DataProcessor()

@pytest.fixture
def synthetic_generator():
    """Create synthetic generator instance"""
    return SyntheticGenerator(
        epsilon=1.0,
        delta=1e-5,
        model_type='dpgan'
    )

def test_synthetic_generator_initialization():
    """Test synthetic generator initialization"""
    generator = SyntheticGenerator()
    assert generator.epsilon == settings.DEFAULT_EPSILON
    assert generator.delta == settings.DEFAULT_DELTA
    assert generator.model_type == settings.DEFAULT_MODEL

def test_invalid_model_type():
    """Test initialization with invalid model type"""
    with pytest.raises(ValueError):
        SyntheticGenerator(model_type='invalid_model')

def test_synthetic_data_generation(sample_data, data_processor, synthetic_generator):
    """Test synthetic data generation pipeline"""
    # Preprocess data
    loader, info = data_processor.preprocess_data(sample_data)
    
    # Generate synthetic data
    result = synthetic_generator.generate_synthetic_data(loader)
    
    # Check results
    assert 'synthetic_data' in result
    assert 'privacy_metrics' in result
    assert 'utility_metrics' in result
    assert isinstance(result['synthetic_data'], pd.DataFrame)
    assert len(result['synthetic_data']) == len(sample_data)
    assert all(col in result['synthetic_data'].columns for col in sample_data.columns)

def test_privacy_evaluation(sample_data, data_processor, synthetic_generator):
    """Test privacy metrics evaluation"""
    # Preprocess data
    loader, info = data_processor.preprocess_data(sample_data)
    
    # Generate synthetic data
    result = synthetic_generator.generate_synthetic_data(loader)
    
    # Check privacy metrics
    privacy_metrics = result['privacy_metrics']
    assert 'epsilon_score' in privacy_metrics
    assert 'membership_disclosure_score' in privacy_metrics
    
def test_utility_evaluation(sample_data, data_processor, synthetic_generator):
    """Test utility metrics evaluation"""
    # Preprocess data
    loader, info = data_processor.preprocess_data(sample_data)
    
    # Generate synthetic data
    result = synthetic_generator.generate_synthetic_data(loader)
    
    # Check utility metrics
    utility_metrics = result['utility_metrics']
    assert 'statistical_similarity' in utility_metrics
    assert 'feature_correlation' in utility_metrics 