"""Core qstruct functionality."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from chebpy import chebfun


def greet(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


def approximate(func: Callable[..., Any], domain: Sequence[float] = (-1, 1)) -> Any:
    """Approximate a function using Chebyshev polynomials."""
    return chebfun(func, list(domain))
