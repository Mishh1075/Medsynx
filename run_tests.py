import subprocess
import sys
import os
import time

def check_environment():
    """Check if all required components are available."""
    print("Checking environment...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check required packages
    try:
        import pytest
        import fastapi
        import torch
        import monai
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check if test data directory exists
    if not os.path.exists("tests/test_data"):
        os.makedirs("tests/test_data")
        print("Created test data directory.")
    
    return True

def run_unit_tests():
    """Run unit tests."""
    print("\nRunning unit tests...")
    result = subprocess.run(["pytest", "tests/test_image_processing.py", "-v"], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0

def run_api_tests():
    """Run API integration tests."""
    print("\nRunning API tests...")
    result = subprocess.run(["pytest", "tests/test_api_endpoints.py", "-v"], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0

def run_e2e_tests():
    """Run end-to-end tests."""
    print("\nStarting services for E2E tests...")
    
    # Start FastAPI server
    api_process = subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Start Streamlit
    streamlit_process = subprocess.Popen(["streamlit", "run", "frontend/main.py"])
    
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(10)
    
    print("\nRunning E2E tests...")
    result = subprocess.run(["pytest", "tests/test_e2e.py", "-v"], capture_output=True, text=True)
    print(result.stdout)
    
    # Cleanup
    api_process.terminate()
    streamlit_process.terminate()
    
    return result.returncode == 0

def main():
    """Main test runner."""
    if not check_environment():
        sys.exit(1)
    
    # Run all tests
    unit_tests_passed = run_unit_tests()
    api_tests_passed = run_api_tests()
    e2e_tests_passed = run_e2e_tests()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Unit Tests: {'✓' if unit_tests_passed else '✗'}")
    print(f"API Tests: {'✓' if api_tests_passed else '✗'}")
    print(f"E2E Tests: {'✓' if e2e_tests_passed else '✗'}")
    
    if unit_tests_passed and api_tests_passed and e2e_tests_passed:
        print("\nAll tests passed successfully! ✨")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 