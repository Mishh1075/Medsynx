# MedSynX Quick Start Guide

## Immediate Demo

### 1. Start the Application
```bash
# Using Docker (Recommended)
docker-compose up --build

# Or manually
python run.py
```

### 2. Access the Platform
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

### 3. Quick Demo Steps

#### Step 1: User Registration
1. Navigate to http://localhost:8501
2. Click "Register"
3. Create account with:
   - Email: demo@example.com
   - Password: Demo123!

#### Step 2: Data Upload
1. Go to "Upload Data"
2. Use sample data from `samples/demo_ehr.csv`
3. Review automatic data validation results

#### Step 3: Generate Synthetic Data
1. Select your uploaded dataset
2. Configure privacy parameters:
   - Epsilon: 1.0 (default)
   - Delta: 1e-5 (default)
3. Click "Generate"

#### Step 4: View Results
1. Check evaluation metrics
2. View visualizations
3. Download synthetic data

## Key Features to Demo

### 1. Privacy Controls
- Show privacy parameter configuration
- Demonstrate privacy metrics
- View audit logs

### 2. Data Quality
- Statistical similarity measures
- Distribution comparisons
- ML utility metrics

### 3. Security Features
- Authentication system
- Data encryption
- Access controls

### 4. Monitoring
- Generation progress
- Privacy scores
- System metrics

## Sample Data

We've included sample datasets in `samples/`:
- `demo_ehr.csv`: Synthetic EHR data
- `demo_lab_results.csv`: Lab test results
- `demo_medications.csv`: Medication records

## Common Operations

### Adjust Privacy Settings
```python
# Example privacy configuration
{
    "epsilon": 1.0,        # More private: 0.1, Less private: 10.0
    "delta": 1e-5,        # Recommended range: 1e-7 to 1e-5
    "min_records": 1000   # Minimum records for synthesis
}
```

### View Privacy Metrics
```python
# Example privacy scores
{
    "membership_inference_risk": 0.12,    # Lower is better
    "attribute_disclosure_risk": 0.08,    # Lower is better
    "k_anonymity_score": 0.95,           # Higher is better
    "l_diversity_score": 0.88            # Higher is better
}
```

## Troubleshooting

### Common Issues

1. **Upload Fails**
   - Check file format (CSV/Excel only)
   - Verify file size (< 10MB)
   - Ensure proper column names

2. **Generation Errors**
   - Verify minimum dataset size
   - Check privacy parameters
   - Review data types

3. **Performance Issues**
   - Reduce dataset size
   - Adjust batch size
   - Check system resources

## Next Steps

1. **Customize Parameters**
   - Adjust privacy settings
   - Modify generation options
   - Configure evaluation metrics

2. **Integration**
   - API documentation
   - Authentication tokens
   - Batch processing

3. **Advanced Features**
   - Medical imaging support
   - Custom privacy rules
   - Advanced visualizations

## Support

For immediate assistance:
- Technical Support: support@medsynx.com
- Documentation: http://localhost:8000/docs
- Issues: https://github.com/yourusername/medsynx/issues 