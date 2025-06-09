# MedSynx: Privacy-Preserving Medical Data Synthesis Platform

MedSynx is a comprehensive platform for generating synthetic medical data with strong privacy guarantees. It supports both tabular data and medical imaging, with built-in privacy metrics and utility evaluation tools.

## Features

- **Privacy-Preserving Data Generation**
  - Differential Privacy (ε, δ) guarantees
  - Membership Inference Attack resistance
  - k-anonymity and l-diversity metrics

- **Medical Data Support**
  - Tabular data (demographics, conditions, medications)
  - Medical imaging (DICOM/NIFTI support)
  - GAN-based image synthesis

- **Evaluation Tools**
  - Privacy metrics visualization
  - Statistical similarity tests
  - ML utility metrics
  - Medical domain-specific evaluation

## Quick Start

1. **Setup Environment**

```bash
# Clone repository
git clone https://github.com/yourusername/medsynx.git
cd medsynx

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Generate Sample Data**

```bash
# Generate and preprocess Synthea data
python scripts/generate_sample_data.py
```

3. **Run Demo**

```bash
# Run complete demo workflow
python scripts/run_demo.py
```

4. **Start Development Server**

```bash
# Start FastAPI server
uvicorn app.main:app --reload
```

## Documentation

### Privacy Parameters

The platform supports configurable privacy parameters:

- `epsilon (ε)`: Privacy budget (default: 1.0)
- `delta (δ)`: Privacy relaxation parameter (default: 1e-5)
- `noise_multiplier`: Controls noise addition for privacy (default: 1.0)
- `num_samples`: Number of synthetic samples to generate

### Medical Image Generation

The platform includes a GAN-based medical image generator:

```python
from app.services.image_generator import MedicalImageGenerator

# Initialize generator
generator = MedicalImageGenerator()

# Generate synthetic images
images = generator.generate_images(num_images=4)

# Save as DICOM
generator.save_dicom(images[0], 'output/synthetic_image.dcm')
```

### Evaluation Metrics

```python
from app.services.medical_evaluator import MedicalEvaluator

evaluator = MedicalEvaluator()
metrics = evaluator.evaluate_medical_data(
    original_data,
    synthetic_data,
    categorical_columns=['RACE', 'ETHNICITY', 'GENDER']
)
```

## Development

### Project Structure

```
medsynx/
├── app/
│   ├── services/
│   │   ├── synthetic_generator.py
│   │   ├── privacy_evaluator.py
│   │   ├── utility_evaluator.py
│   │   ├── medical_evaluator.py
│   │   └── image_generator.py
│   ├── api/
│   └── main.py
├── scripts/
│   ├── generate_sample_data.py
│   └── run_demo.py
├── notebooks/
│   └── run_demo.ipynb
├── tests/
├── sample_data/
└── docker/
```

### Running Tests

```bash
pytest tests/ --cov=app
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.prod.yml up -d
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Synthea](https://github.com/synthetichealth/synthea) for synthetic patient data generation
- [PyTorch](https://pytorch.org/) for deep learning capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework 