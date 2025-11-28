import pytest

from snnmm.datasets.cifar10 import CIFAR10Dataset


def test_cifar10_dataset_shape_and_label():
    try:
        ds = CIFAR10Dataset(root="data", train=True, download=False)
    except Exception:
        pytest.skip("CIFAR-10 data not available in test environment.")
    img, label, name = ds[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= label < 10
    assert isinstance(name, str) and len(name) > 0
