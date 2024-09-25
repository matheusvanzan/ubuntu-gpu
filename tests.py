import unittest
import torch

class TestCudaAvailability(unittest.TestCase):
    
    def test_gpu_available(self):
        self.assertTrue(torch.cuda.is_available(), "No GPU available.")

    def test_get_device_name(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.assertIsInstance(gpu_name, str, "GPU name should be a string.")
        else:
            self.skipTest("Skipping test as no GPU is available.")
            
if __name__ == '__main__':
    unittest.main()
