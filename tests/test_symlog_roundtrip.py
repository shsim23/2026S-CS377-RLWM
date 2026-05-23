import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model.utils import symlog, symexp


def test_symexp_inverts_symlog():
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    reconstructed = symexp(symlog(x))
    torch.testing.assert_close(reconstructed, x, atol=1e-4, rtol=1e-4)


def test_symlog_compresses_large_values():
    big = torch.tensor([1000.0])
    small = torch.tensor([1.0])
    assert symlog(big) < big
    assert symlog(small).abs() < small.abs() + 1e-6


def test_symlog_odd_function():
    x = torch.tensor([2.0, 5.0, 10.0])
    torch.testing.assert_close(symlog(-x), -symlog(x))
