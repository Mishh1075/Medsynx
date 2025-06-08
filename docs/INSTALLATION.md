# Installation Guide

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional)
- Git

## Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medsynx.git
cd medsynx
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
python run.py
```

The application will be available at:
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medsynx.git
cd medsynx
```

2. Build and run with Docker Compose:
```bash
docker-compose up -d --build
```

The application will be available at the same ports as the local installation.

## Configuration

### Environment Variables

- `SECRET_KEY`: Secret key for JWT token generation
- `DATABASE_URL`: SQLite database URL
- `UPLOAD_DIR`: Directory for uploaded files
- `ACCESS_TOKEN_EXPIRE_MINUTES`: JWT token expiration time

### Database Setup

The application uses SQLite by default. The database will be automatically created on first run.

## Troubleshooting

### Common Issues

1. Port conflicts:
   - Ensure ports 8000 and 8501 are available
   - Change ports in docker-compose.yml if needed

2. Permission issues:
   - Ensure write permissions for uploads/ and data/ directories
   - Run with sudo if necessary (Docker only)

3. Dependencies:
   - If you encounter dependency conflicts, try creating a fresh virtual environment
   - Update pip: `pip install --upgrade pip`

### Getting Help

If you encounter any issues:
1. Check the logs: `docker-compose logs` (Docker) or check console output
2. Open an issue on GitHub
3. Contact the development team 