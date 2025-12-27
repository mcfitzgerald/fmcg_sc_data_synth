"""Base classes for data writers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseWriter(ABC):
    """Abstract base class for all simulation writers."""

    @abstractmethod
    def write(self, data: Any, destination: str) -> None:
        """Write data to the specified destination."""
        pass
