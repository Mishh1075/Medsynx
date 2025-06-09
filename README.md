# MedSynx: Privacy-Preserving Synthetic Medical Data Generator

A scalable platform for generating synthetic medical data using Generative Adversarial Networks (GANs) with Differential Privacy guarantees.

## Features

- 🔒 User Authentication & Authorization
- 📊 Tabular Data Generation using SynthCity
- 🔐 Differential Privacy Integration
- 📈 Utility and Privacy Metrics
- 🌐 Modern React Frontend
- 🚀 FastAPI Backend

## Project Structure

```
Medsynx/
├── app/                    # Backend (FastAPI)
│   ├── api/               # API endpoints
│   ├── core/              # Core configurations
│   ├── db/                # Database models and sessions
│   ├── models/            # SQLAlchemy models
│   └── services/          # Business logic
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   └── services/     # API services
├── data/                  # Data storage
│   ├── original/          # Original uploaded data
│   └── synthetic/         # Generated synthetic data
├── notebooks/             # Jupyter notebooks for testing
├── scripts/               # Utility scripts
├── tests/                 # Test cases
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker compose configuration
└── requirements.txt       # Python dependencies
```

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Medsynx.git
   cd Medsynx
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

4. Start the backend server:
   ```bash
   cd ..
   uvicorn app.main:app --reload
   ```

5. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

6. Visit http://localhost:3000 in your browser

## Environment Variables

Create a `.env` file in the root directory:

```env
SECRET_KEY=your-secret-key
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=synthetic_data
POSTGRES_SERVER=localhost
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Features in Detail

### Data Generation
- Upload tabular data (CSV format)
- Generate synthetic data using SynthCity's DP-GAN implementation
- Download generated synthetic data
- View utility and privacy metrics

### Privacy Guarantees
- Differential Privacy integration through SynthCity
- Configurable privacy parameters (ε, δ)
- Privacy evaluation metrics

### Evaluation Metrics
- Statistical similarity measures
- Machine learning utility metrics
- Privacy attack resistance metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SynthCity library for synthetic data generation
- FastAPI for the backend framework
- React for the frontend framework 