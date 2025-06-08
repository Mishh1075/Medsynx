import unittest
import torch
import numpy as np
from app.services.imaging import ImageProcessor, ImageGenerator, ImageEvaluator

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ImageProcessor()
        self.test_image = np.random.rand(256, 256)
    
    def test_normalize(self):
        normalized = self.processor.normalize(self.test_image)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
        
    def test_resize(self):
        resized = self.processor.resize(self.test_image)
        self.assertEqual(resized.shape, (256, 256))

class TestImageGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ImageGenerator()
    
    def test_generate(self):
        images = self.generator.generate(num_images=1)
        self.assertIsInstance(images, torch.Tensor)
        self.assertEqual(images.shape[0], 1)
    
    def test_privacy(self):
        # Test differential privacy noise addition
        tensor = torch.randn(1, 1, 256, 256)
        noisy = self.generator.add_noise(tensor)
        self.assertFalse(torch.equal(tensor, noisy))

class TestImageEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ImageEvaluator()
        self.real = torch.randn(1, 1, 256, 256)
        self.synthetic = torch.randn(1, 1, 256, 256)
    
    def test_metrics_calculation(self):
        metrics = self.evaluator.calculate_metrics(self.real, self.synthetic)
        required_metrics = [
            'ssim', 'psnr', 'wasserstein', 
            'mean_diff', 'std_diff', 'histogram_intersection',
            'membership_inference_risk', 'attribute_disclosure_risk'
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
    
    def test_report_generation(self):
        metrics = self.evaluator.calculate_metrics(self.real, self.synthetic)
        report = self.evaluator.generate_report()
        self.assertIn('average_metrics', report)
        self.assertIn('privacy_assessment', report)

if __name__ == '__main__':
    unittest.main() 