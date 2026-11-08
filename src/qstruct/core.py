"""Core qstruct functionality."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from chebpy import chebfun

if TYPE_CHECKING:
    import numpy as np
    from chebpy.chebfun import Chebfun
    from numpy.typing import NDArray


def greet(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


def approximate(
    func: Callable[..., float | NDArray[np.float64]] | str | float | None,
    domain: Sequence[float] = (-1, 1),
) -> Chebfun:
    """Approximate a function using Chebyshev polynomials."""
    return chebfun(func, list(domain))
