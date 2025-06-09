import pytest
from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import numpy as np
from pathlib import Path
import json

client = TestClient(app)

@pytest.fixture
def test_user():
    return {
        "email": "test@example.com",
        "password": "testpassword123"
    }

@pytest.fixture
def auth_headers(test_user):
    # Register user
    client.post("/api/auth/register", json=test_user)
    
    # Login to get token
    response = client.post("/api/auth/login", json=test_user)
    token = response.json()["token"]
    
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.choice(['<=50K', '>50K'], 100),
        'education': np.random.choice(['HS', 'College', 'Masters'], 100),
        'occupation': np.random.choice(['Tech', 'Sales', 'Other'], 100)
    })
    
    # Save to temp file
    temp_file = Path("temp_test_data.csv")
    data.to_csv(temp_file, index=False)
    return temp_file

def test_register_user():
    response = client.post(
        "/api/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "newpassword123"
        }
    )
    assert response.status_code == 201
    assert "id" in response.json()
    assert "email" in response.json()

def test_login_user(test_user):
    response = client.post("/api/auth/login", json=test_user)
    assert response.status_code == 200
    assert "token" in response.json()
    assert "user" in response.json()

def test_upload_data(auth_headers, sample_data):
    with open(sample_data, "rb") as f:
        response = client.post(
            "/api/data/upload",
            headers=auth_headers,
            files={"file": ("test.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    assert "job_id" in response.json()

def test_get_job_status(auth_headers):
    # First upload data to get a job
    with open(sample_data, "rb") as f:
        upload_response = client.post(
            "/api/data/upload",
            headers=auth_headers,
            files={"file": ("test.csv", f, "text/csv")}
        )
    job_id = upload_response.json()["job_id"]
    
    # Check job status
    response = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    assert response.status_code == 200
    assert "status" in response.json()
    assert "progress" in response.json()

def test_generate_synthetic_data(auth_headers, sample_data):
    # Upload data first
    with open(sample_data, "rb") as f:
        upload_response = client.post(
            "/api/data/upload",
            headers=auth_headers,
            files={"file": ("test.csv", f, "text/csv")}
        )
    job_id = upload_response.json()["job_id"]
    
    # Generate synthetic data
    response = client.post(
        f"/api/data/{job_id}/generate",
        headers=auth_headers,
        json={
            "epsilon": 1.0,
            "delta": 1e-5,
            "model_type": "dpgan"
        }
    )
    assert response.status_code == 200
    assert "synthetic_job_id" in response.json()

def test_download_synthetic_data(auth_headers, sample_data):
    # Upload and generate synthetic data first
    with open(sample_data, "rb") as f:
        upload_response = client.post(
            "/api/data/upload",
            headers=auth_headers,
            files={"file": ("test.csv", f, "text/csv")}
        )
    job_id = upload_response.json()["job_id"]
    
    gen_response = client.post(
        f"/api/data/{job_id}/generate",
        headers=auth_headers,
        json={
            "epsilon": 1.0,
            "delta": 1e-5,
            "model_type": "dpgan"
        }
    )
    synthetic_job_id = gen_response.json()["synthetic_job_id"]
    
    # Download synthetic data
    response = client.get(
        f"/api/data/{synthetic_job_id}/download",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert "Content-Type" in response.headers
    assert response.headers["Content-Type"] == "text/csv"

def test_get_privacy_metrics(auth_headers, sample_data):
    # Upload and generate synthetic data first
    with open(sample_data, "rb") as f:
        upload_response = client.post(
            "/api/data/upload",
            headers=auth_headers,
            files={"file": ("test.csv", f, "text/csv")}
        )
    job_id = upload_response.json()["job_id"]
    
    gen_response = client.post(
        f"/api/data/{job_id}/generate",
        headers=auth_headers,
        json={
            "epsilon": 1.0,
            "delta": 1e-5,
            "model_type": "dpgan"
        }
    )
    synthetic_job_id = gen_response.json()["synthetic_job_id"]
    
    # Get privacy metrics
    response = client.get(
        f"/api/metrics/{synthetic_job_id}/privacy",
        headers=auth_headers
    )
    assert response.status_code == 200
    metrics = response.json()
    assert "epsilon_score" in metrics
    assert "membership_disclosure_score" in metrics

def test_get_utility_metrics(auth_headers, sample_data):
    # Upload and generate synthetic data first
    with open(sample_data, "rb") as f:
        upload_response = client.post(
            "/api/data/upload",
            headers=auth_headers,
            files={"file": ("test.csv", f, "text/csv")}
        )
    job_id = upload_response.json()["job_id"]
    
    gen_response = client.post(
        f"/api/data/{job_id}/generate",
        headers=auth_headers,
        json={
            "epsilon": 1.0,
            "delta": 1e-5,
            "model_type": "dpgan"
        }
    )
    synthetic_job_id = gen_response.json()["synthetic_job_id"]
    
    # Get utility metrics
    response = client.get(
        f"/api/metrics/{synthetic_job_id}/utility",
        headers=auth_headers
    )
    assert response.status_code == 200
    metrics = response.json()
    assert "statistical_similarity" in metrics
    assert "feature_correlation" in metrics

def teardown_module(module):
    """Clean up temporary files after tests"""
    temp_file = Path("temp_test_data.csv")
    if temp_file.exists():
        temp_file.unlink() 