"""Tests for qstruct.core."""

from __future__ import annotations

import numpy as np

from qstruct.core import approximate, greet


def test_greet():
    """Verify greet returns expected greeting."""
    assert greet("World") == "Hello, World!"


def test_greet_empty():
    """Verify greet handles empty name."""
    assert greet("") == "Hello, !"


def test_approximate():
    """Verify approximate creates a Chebyshev approximation."""
    f = approximate(np.sin, domain=(-1, 1))
    assert abs(f(0.5) - np.sin(0.5)) < 1e-10
