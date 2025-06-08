import unittest
from fastapi.testclient import TestClient
from app.main import app
import io
import numpy as np

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_user = {
            "email": "test@example.com",
            "password": "testpassword123"
        }
        
        # Register and login
        self.client.post("/api/auth/register", json=self.test_user)
        response = self.client.post("/api/auth/login", data=self.test_user)
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def test_upload_tabular_data(self):
        # Create test CSV
        csv_content = "id,age,gender\n1,25,M\n2,30,F"
        files = {"file": ("test.csv", csv_content)}
        
        response = self.client.post(
            "/api/upload/tabular",
            files=files,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("dataset_id", response.json())
    
    def test_upload_medical_image(self):
        # Create test image
        image = np.random.rand(256, 256)
        image_bytes = io.BytesIO()
        np.save(image_bytes, image)
        
        files = {"file": ("test.npy", image_bytes.getvalue())}
        response = self.client.post(
            "/api/upload/medical-image",
            files=files,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("metadata", response.json())
    
    def test_generate_synthetic_data(self):
        # Test tabular data generation
        params = {
            "dataset_id": "test_dataset",
            "num_samples": 10,
            "epsilon": 1.0,
            "delta": 1e-5
        }
        response = self.client.post(
            "/api/generate/tabular",
            json=params,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("synthetic_data", response.json())
    
    def test_generate_synthetic_images(self):
        params = {
            "num_images": 1,
            "epsilon": 1.0,
            "delta": 1e-5
        }
        response = self.client.post(
            "/api/generate/medical-image",
            params=params,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("images", response.json())
    
    def test_evaluate_synthetic_data(self):
        # Create test data
        real_data = "id,age\n1,25\n2,30"
        synthetic_data = "id,age\n1,26\n2,31"
        
        files = {
            "real_data": ("real.csv", real_data),
            "synthetic_data": ("synthetic.csv", synthetic_data)
        }
        response = self.client.post(
            "/api/evaluate/tabular",
            files=files,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("metrics", response.json())
    
    def test_evaluate_synthetic_images(self):
        # Create test images
        real_image = np.random.rand(256, 256)
        synthetic_image = np.random.rand(256, 256)
        
        real_bytes = io.BytesIO()
        synthetic_bytes = io.BytesIO()
        np.save(real_bytes, real_image)
        np.save(synthetic_bytes, synthetic_image)
        
        files = {
            "real_images": [("real.npy", real_bytes.getvalue())],
            "synthetic_images": [("synthetic.npy", synthetic_bytes.getvalue())]
        }
        response = self.client.post(
            "/api/evaluate/medical-image",
            files=files,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("metrics", response.json())

if __name__ == '__main__':
    unittest.main() 