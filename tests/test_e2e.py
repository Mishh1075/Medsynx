import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:8501")  # Streamlit port
        self.wait = WebDriverWait(self.driver, 10)
    
    def tearDown(self):
        self.driver.quit()
    
    def test_tabular_data_workflow(self):
        """Test complete tabular data workflow."""
        # Login
        self.login("test@example.com", "testpassword123")
        
        # Upload tabular data
        upload_btn = self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Upload Data')]"))
        )
        upload_btn.click()
        
        file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        file_input.send_keys(os.path.abspath("test_data/sample.csv"))
        
        # Wait for upload confirmation
        self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Upload successful')]"))
        )
        
        # Generate synthetic data
        generate_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Generate')]")
        generate_btn.click()
        
        # Set parameters
        epsilon_input = self.driver.find_element(By.ID, "epsilon")
        epsilon_input.clear()
        epsilon_input.send_keys("1.0")
        
        # Start generation
        start_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Start Generation')]")
        start_btn.click()
        
        # Wait for generation completion
        self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Generation complete')]"))
        )
        
        # Verify results
        download_btn = self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Download')]"))
        )
        self.assertTrue(download_btn.is_enabled())
    
    def test_medical_image_workflow(self):
        """Test complete medical image workflow."""
        # Login
        self.login("test@example.com", "testpassword123")
        
        # Navigate to medical imaging
        nav_btn = self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Medical Imaging')]"))
        )
        nav_btn.click()
        
        # Upload medical image
        file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        file_input.send_keys(os.path.abspath("test_data/sample.dcm"))
        
        # Wait for upload confirmation
        self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Upload successful')]"))
        )
        
        # Generate synthetic images
        num_images_input = self.driver.find_element(By.ID, "num_images")
        num_images_input.clear()
        num_images_input.send_keys("1")
        
        generate_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Generate')]")
        generate_btn.click()
        
        # Wait for generation completion
        self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Generation complete')]"))
        )
        
        # Verify results
        self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "synthetic-image"))
        )
    
    def login(self, email: str, password: str):
        """Helper method to perform login."""
        email_input = self.wait.until(
            EC.presence_of_element_located((By.ID, "email"))
        )
        password_input = self.driver.find_element(By.ID, "password")
        
        email_input.send_keys(email)
        password_input.send_keys(password)
        
        login_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
        login_btn.click()
        
        # Wait for successful login
        self.wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Welcome')]"))
        )

if __name__ == '__main__':
    unittest.main() 