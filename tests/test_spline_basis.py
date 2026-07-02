import numpy as np

from pobsnn.core import BSplineBasis, BSplineLayer


def test_basis_partition_of_unity():
    basis = BSplineBasis(degree=3, num_basis=10)
    x = np.linspace(-1.0, 1.0, 101)
    B = basis(x)
    assert B.shape == (101, 10)
    assert np.allclose(B.sum(axis=-1), 1.0)


def test_layer_forward_shape():
    layer = BSplineLayer(in_features=2, out_features=3, seed=1)
    x = np.zeros((5, 2))
    y = layer(x)
    assert y.shape == (5, 3)


def test_manual_training_reduces_loss():
    x = np.linspace(-1.0, 1.0, 64)[:, None]
    y = np.sin(np.pi * x)
    layer = BSplineLayer(1, 1, num_basis=12, seed=2)
    losses = layer.fit_coefficients_mse(x, y, epochs=100, learning_rate=0.08)
    assert losses[-1] < losses[0]
