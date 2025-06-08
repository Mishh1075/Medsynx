# MedSynX User Manual

## Introduction

MedSynX is a platform for generating synthetic healthcare data with privacy guarantees. This manual will guide you through using the platform effectively.

## Getting Started

### 1. Account Creation and Login
1. Visit the MedSynX platform
2. Click "Register" in the sidebar
3. Enter your email and password
4. Verify your account
5. Log in with your credentials

### 2. Navigation
The platform has five main sections:
- **My Datasets**: View and manage your uploaded datasets
- **Upload Data**: Upload new healthcare datasets
- **Generate Synthetic Data**: Create synthetic versions of your data
- **Medical Imaging**: Process and generate synthetic medical images
- **Evaluate Results**: Analyze the quality and privacy of synthetic data

## Using MedSynX

### 1. Uploading Data

#### A. Tabular Data
1. Navigate to "Upload Data"
2. Click "Choose a CSV or Excel file"
3. Select your healthcare dataset
4. Review the data summary and validation results
5. Confirm the upload

Supported formats:
- CSV files (.csv)
- Excel files (.xlsx)

#### B. Medical Images
1. Navigate to "Medical Imaging" tab
2. Select "Upload" tab
3. Choose medical image file
4. Review metadata and image information
5. Confirm the upload

Supported formats:
- DICOM files (.dcm)
- NIfTI files (.nii, .nii.gz)
- Standard images (.png, .jpg)

### 2. Generating Synthetic Data

#### A. Tabular Data
1. Go to "Generate Synthetic Data"
2. Select a dataset from your uploads
3. Choose a synthetic data model:
   - DPGAN (Differentially Private GAN)
   - PATEGAN (PATE Framework GAN)
   - Other available models
4. Set privacy parameters:
   - Epsilon (ε): Privacy budget (0.1-10.0)
   - Delta (δ): Privacy relaxation (1e-7 to 1e-3)
5. Click "Generate Synthetic Data"
6. Wait for the generation process to complete

#### B. Medical Images
1. Navigate to "Medical Imaging"
2. Select "Generate" tab
3. Set number of images to generate (1-10)
4. Adjust privacy parameters:
   - Epsilon (ε): Privacy budget
   - Delta (δ): Privacy relaxation
5. Click "Generate"
6. View and download generated images

### 3. Evaluating Results

#### A. Tabular Data
1. Access "Evaluate Results"
2. Select original and synthetic datasets
3. View evaluation metrics:
   - Statistical similarity
   - Privacy guarantees
   - Distribution comparisons

#### B. Medical Images
1. Navigate to "Medical Imaging"
2. Select "Evaluate" tab
3. Upload pairs of real and synthetic images
4. View comprehensive metrics:
   - Image Quality Metrics:
     - Structural Similarity (SSIM)
     - Peak Signal-to-Noise Ratio (PSNR)
     - Wasserstein Distance
   - Privacy Metrics:
     - Membership Inference Risk
     - Attribute Disclosure Risk
5. Examine visual reports and plots

## Privacy Controls

### Understanding Privacy Parameters

1. Epsilon (ε):
   - Lower values = stronger privacy
   - Higher values = better utility
   - Recommended range: 0.1-5.0

2. Delta (δ):
   - Probability of privacy breach
   - Smaller values = stronger privacy
   - Recommended: 1e-5

### Privacy Metrics

1. Differential Privacy:
   - Measures privacy guarantees
   - Shows epsilon achieved

2. Membership Inference:
   - Tests re-identification risk
   - Shows attack success rate

3. Attribute Disclosure:
   - Measures correlation leakage
   - Indicates attribute privacy

## Best Practices

1. Data Preparation:
   - Clean your data before upload
   - Remove sensitive identifiers
   - Ensure consistent formatting

2. Privacy Settings:
   - Start with conservative epsilon values
   - Adjust based on results
   - Monitor privacy metrics

3. Image Processing:
   - Use appropriate file formats
   - Check image metadata
   - Verify image quality

4. Evaluation:
   - Compare distributions
   - Check utility metrics
   - Validate privacy guarantees

## Troubleshooting

### Common Issues

1. Upload Failures:
   - Check file format
   - Verify file size (max 10MB)
   - Ensure clean data

2. Generation Errors:
   - Check input data quality
   - Verify privacy parameters
   - Try different models

3. Evaluation Problems:
   - Refresh the page
   - Check data completeness
   - Verify metrics calculation

4. Image Processing Issues:
   - Verify file format compatibility
   - Check image dimensions
   - Ensure proper metadata

### Getting Help

If you encounter issues:
1. Check the documentation
2. Contact support
3. Report bugs through GitHub

## Security Recommendations

1. Data Protection:
   - Use strong passwords
   - Don't share account credentials
   - Download data securely

2. Privacy Considerations:
   - Use appropriate epsilon values
   - Verify privacy metrics
   - Monitor synthetic data quality

3. Image Security:
   - Remove identifying metadata
   - Use secure transfer protocols
   - Monitor access logs

## Updates and Maintenance

1. Platform Updates:
   - Check for new features
   - Review change logs
   - Update local installations

2. Data Management:
   - Regular cleanup
   - Archive old datasets
   - Monitor storage usage 