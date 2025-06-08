import subprocess
import sys
import time
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    print("Starting FastAPI backend server...")
    subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--reload"])

def run_frontend():
    """Run the Streamlit frontend"""
    print("Starting Streamlit frontend...")
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])

def setup_environment():
    """Create necessary directories and files"""
    # Create uploads directory
    Path("uploads").mkdir(exist_ok=True)
    
    # Create empty __init__.py files for Python packages
    Path("app/__init__.py").touch()
    Path("app/api/__init__.py").touch()
    Path("app/core/__init__.py").touch()
    Path("app/db/__init__.py").touch()
    Path("app/services/__init__.py").touch()

def main():
    """Main function to run the application"""
    print("Setting up MedSynX environment...")
    setup_environment()
    
    # Start backend
    run_backend()
    
    # Wait for backend to start
    time.sleep(2)
    
    # Start frontend
    run_frontend()
    
    print("\nMedSynX is running!")
    print("Backend API: http://localhost:8000")
    print("Frontend UI: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down MedSynX...")
        sys.exit(0)

if __name__ == "__main__":
    main() 