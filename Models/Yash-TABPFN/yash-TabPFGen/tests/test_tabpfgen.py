import unittest
import numpy as np
from tabpfgen import TabPFGen
from sklearn.datasets import make_classification, make_regression
import torch

class TestTabPFGen(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cpu")
        # Use smaller values for testing to speed up execution
        self.generator = TabPFGen(
            n_sgld_steps=10,  # Reduced for testing
            sgld_step_size=0.01,
            sgld_noise_scale=0.01,
            device=self.device
        )
        
        # Create synthetic datasets for testing
        self.X_class, self.y_class = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )
        
        self.X_reg, self.y_reg = make_regression(
            n_samples=100,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42
        )

    def test_initialization(self):
        """Test proper initialization of TabPFGen."""
        self.assertEqual(self.generator.n_sgld_steps, 10)
        self.assertEqual(self.generator.sgld_step_size, 0.01)
        self.assertEqual(self.generator.sgld_noise_scale, 0.01)
        self.assertEqual(self.generator.device.type, "cpu")
        self.assertIsNotNone(self.generator.scaler)

    def test_classification_generation(self):
        """Test generation of synthetic classification data."""
        n_samples = 51  # Changed to be divisible by 3 classes
        X_synth, y_synth = self.generator.generate_classification(
            self.X_class,
            self.y_class,
            n_samples,
            balance_classes=True
        )
        
        # Check shapes
        expected_samples = (n_samples // 3) * 3  # Round down to nearest multiple of num_classes
        self.assertEqual(X_synth.shape[0], expected_samples)
        self.assertEqual(X_synth.shape[1], self.X_class.shape[1])
        self.assertEqual(y_synth.shape[0], expected_samples)
        
        # Check data types
        self.assertTrue(isinstance(X_synth, np.ndarray))
        self.assertTrue(isinstance(y_synth, np.ndarray))
        
        # Check if generated classes are valid
        unique_classes = np.unique(self.y_class)
        self.assertTrue(all(cls in unique_classes for cls in np.unique(y_synth)))

    def test_regression_generation(self):
        """Test generation of synthetic regression data."""
        n_samples = 50
        X_synth, y_synth = self.generator.generate_regression(
            self.X_reg,
            self.y_reg,
            n_samples,
            use_quantiles=True
        )
        
        # Check shapes
        self.assertEqual(X_synth.shape[0], n_samples)
        self.assertEqual(X_synth.shape[1], self.X_reg.shape[1])
        self.assertEqual(y_synth.shape[0], n_samples)
        
        # Check data types
        self.assertTrue(isinstance(X_synth, np.ndarray))
        self.assertTrue(isinstance(y_synth, np.ndarray))
        
        # Check if generated values are within reasonable bounds
        y_min, y_max = self.y_reg.min(), self.y_reg.max()
        margin = (y_max - y_min) * 0.2  # Allow 20% margin
        self.assertTrue(np.all(y_synth >= y_min - margin))
        self.assertTrue(np.all(y_synth <= y_max + margin))

    def test_energy_computation(self):
        """Test the energy computation function."""
        x_synth = torch.randn(10, 5, device=self.device)
        y_synth = torch.randint(0, 3, (10,), device=self.device)
        x_train = torch.randn(20, 5, device=self.device)
        y_train = torch.randint(0, 3, (20,), device=self.device)
        
        energy = self.generator._compute_energy(x_synth, y_synth, x_train, y_train)
        
        # Check shape and type
        self.assertEqual(energy.shape, (10,))
        self.assertTrue(torch.is_tensor(energy))
        
        # Check if energy is non-negative
        self.assertTrue(torch.all(energy >= 0))

    def test_sgld_step(self):
        """Test the SGLD step function."""
        x_synth = torch.randn(10, 5, device=self.device)
        y_synth = torch.randint(0, 3, (10,), device=self.device)
        x_train = torch.randn(20, 5, device=self.device)
        y_train = torch.randint(0, 3, (20,), device=self.device)
        
        x_new = self.generator._sgld_step(x_synth, y_synth, x_train, y_train)
        
        # Check shape and type
        self.assertEqual(x_new.shape, x_synth.shape)
        self.assertTrue(torch.is_tensor(x_new))
        
        # Check if values have changed
        self.assertFalse(torch.allclose(x_new, x_synth))

    def test_edge_cases(self):
        """Test edge cases and potential error conditions."""
        # Test with single sample
        X_single = self.X_class[:1]
        y_single = self.y_class[:1]
        
        with self.assertRaises(ValueError):
            # Should raise error for n_samples > number of training samples
            self.generator.generate_classification(X_single, y_single, 2, balance_classes=True)
        
        # Test with zero variance feature
        X_zero_var = np.copy(self.X_class)
        X_zero_var[:, 0] = 1.0
        
        # Should handle zero variance feature without errors
        X_synth, y_synth = self.generator.generate_classification(
            X_zero_var,
            self.y_class,
            10,
            balance_classes=False
        )
        self.assertEqual(X_synth.shape[1], X_zero_var.shape[1])

    def test_reproducibility(self):
        """Test reproducibility with fixed random seed."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        X_synth1, y_synth1 = self.generator.generate_classification(
            self.X_class,
            self.y_class,
            20,
            balance_classes=True
        )
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        X_synth2, y_synth2 = self.generator.generate_classification(
            self.X_class,
            self.y_class,
            20,
            balance_classes=True
        )
        
        # Check if results are identical with same seed
        np.testing.assert_array_almost_equal(X_synth1, X_synth2)
        np.testing.assert_array_equal(y_synth1, y_synth2)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    