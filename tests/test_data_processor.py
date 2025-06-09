import pytest
import pandas as pd
import numpy as np
from app.services.data_processor import DataProcessor
from synthcity.plugins.core.schema import Schema

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

def test_schema_inference(sample_data, data_processor):
    """Test schema inference"""
    schema = data_processor.infer_schema(sample_data)
    
    assert isinstance(schema, Schema)
    assert 'age' in schema.numerical_columns
    assert 'income' in schema.categorical_columns
    assert 'education' in schema.categorical_columns
    assert 'occupation' in schema.categorical_columns

def test_data_validation(sample_data, data_processor):
    """Test data validation"""
    assert data_processor.validate_data(sample_data) == True
    
    # Test empty data
    with pytest.raises(ValueError):
        data_processor.validate_data(pd.DataFrame())
    
    # Test data with too many missing values
    data_with_missing = sample_data.copy()
    data_with_missing.iloc[:80, :] = np.nan
    with pytest.raises(ValueError):
        data_processor.validate_data(data_with_missing)

def test_data_preprocessing(sample_data, data_processor):
    """Test data preprocessing"""
    # Add some missing values
    data = sample_data.copy()
    data.iloc[0:10, 0] = np.nan  # Add missing values to age
    data.iloc[20:30, 1] = np.nan  # Add missing values to income
    
    # Preprocess data
    loader, info = data_processor.preprocess_data(data)
    
    # Check preprocessing info
    assert 'schema' in info
    assert 'n_samples' in info
    assert info['n_samples'] == len(data)
    
    # Check that missing values are handled
    assert not loader.data.isnull().any().any()
    
    # Check that schema is preserved
    assert 'age' in info['schema']['numerical_columns']
    assert 'income' in info['schema']['categorical_columns']

def test_preprocessing_with_target(sample_data, data_processor):
    """Test preprocessing with target column"""
    loader, info = data_processor.preprocess_data(
        sample_data,
        target_column='income'
    )
    
    assert info['target_column'] == 'income'
    assert loader.target_column == 'income'

def test_preprocessing_with_sensitive_features(sample_data, data_processor):
    """Test preprocessing with sensitive features"""
    sensitive_features = ['education']
    loader, info = data_processor.preprocess_data(
        sample_data,
        sensitive_features=sensitive_features
    )
    
    assert info['sensitive_features'] == sensitive_features
    assert loader.sensitive_features == sensitive_features 