import pytest
import numpy as np
import torch


def test_autoencoder_import():
    from src.algorithms.autoencoder import Autoencoder, load_ae
    assert Autoencoder is not None
    assert callable(load_ae)


def test_autoencoder_initialization():
    from src.algorithms.autoencoder import Autoencoder
    
    # Test with small z_size for faster testing
    ae = Autoencoder(z_size=8, learning_rate=1e-4)
    
    assert ae.z_size == 8
    assert ae.learning_rate == 1e-4
    assert ae.device is not None


def test_preprocess_input():
    from src.algorithms.autoencoder import preprocess_input
    
    # Create dummy image (H, W, C) format
    img = np.random.randint(0, 255, size=(96, 96, 3), dtype=np.uint8).astype(np.float32)
    
    processed = preprocess_input(img, mode="rl")
    
    # Should be transposed to (C, H, W) and normalized
    assert processed.shape == (3, 96, 96)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0
