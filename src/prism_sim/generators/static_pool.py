"""
StaticDataPool - Pre-generated Faker data for O(1) vectorized sampling.

Replaces per-row Faker calls with pre-generated pools that can be sampled
using NumPy vectorization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from faker import Faker

if TYPE_CHECKING:
    from numpy.random import Generator

# Default pool sizes - pre-generate enough to avoid duplicates in most tables
DEFAULT_POOL_SIZES = {
    "names": 5_000,
    "companies": 5_000,
    "cities": 1_000,
    "emails": 5_000,
    "addresses": 2_000,
    "phone_numbers": 2_000,
    "first_names": 3_000,
    "last_names": 3_000,
}


class StaticDataPool:
    """
    Pre-generated pool of Faker data for efficient vectorized sampling.

    All data is generated once at initialization and stored as Python lists
    for memory efficiency. Sampling uses NumPy for O(1) random access.

    Attributes:
        seed: Random seed for reproducibility
        names: Pool of full names
        companies: Pool of company names
        cities: Pool of city names
        emails: Pool of email addresses
        addresses: Pool of street addresses
        phone_numbers: Pool of phone numbers
        first_names: Pool of first names
        last_names: Pool of last names
    """

    def __init__(
        self,
        seed: int = 42,
        pool_sizes: dict[str, int] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the static data pool with pre-generated Faker data.

        Args:
            seed: Random seed for reproducibility
            pool_sizes: Optional dict overriding default pool sizes
            config: Simulation configuration for city data paths
        """
        self.seed = seed
        self._rng: Generator = np.random.default_rng(seed)
        self.config = config or {}

        # Merge custom sizes with defaults
        sizes = {**DEFAULT_POOL_SIZES, **(pool_sizes or {})}

        # Initialize Faker with seed
        self._faker = Faker()
        Faker.seed(seed)

        # Pre-generate all pools
        self.names: list[str] = self._generate_pool(self._faker.name, sizes["names"])
        self.companies: list[str] = self._generate_pool(
            self._faker.company, sizes["companies"]
        )

        # v0.19.12: Load curated GIS cities instead of faking them
        self.cities: list[dict[str, Any]] = self._load_gis_cities()

        self.emails: list[str] = self._generate_pool(self._faker.email, sizes["emails"])
        self.addresses: list[str] = self._generate_pool(
            self._faker.street_address, sizes["addresses"]
        )
        self.phone_numbers: list[str] = self._generate_pool(
            self._faker.phone_number, sizes["phone_numbers"]
        )
        self.first_names: list[str] = self._generate_pool(
            self._faker.first_name, sizes["first_names"]
        )
        self.last_names: list[str] = self._generate_pool(
            self._faker.last_name, sizes["last_names"]
        )

        # Build pool lookup for generic sample() method
        self._pool_arrays: dict[str, list[str]] = {
            "names": self.names,
            "companies": self.companies,
            "emails": self.emails,
            "addresses": self.addresses,
            "phone_numbers": self.phone_numbers,
            "first_names": self.first_names,
            "last_names": self.last_names,
        }

    def _load_gis_cities(self) -> list[dict[str, Any]]:
        """Load curated US cities with lat/lon from config-defined CSV."""
        import csv
        from pathlib import Path

        # Default path if config missing
        geo_params = self.config.get("simulation_parameters", {}).get("geospatial", {})
        csv_path = geo_params.get("city_data_path", "data/static/us_cities.csv")

        cities = []
        path = Path(csv_path)
        if not path.exists():
            # Fallback to faked names if file missing (prevents crash)
            fake_cities = self._generate_pool(self._faker.city, 100)
            return [
                {"city": c, "state": "US", "lat": 0.0, "lon": 0.0}
                for c in fake_cities
            ]

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cities.append({
                    "city": row["city"],
                    "state": row["state"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "region": row.get("region", "Unknown")
                })
        return cities

    def _generate_pool(
        self,
        generator_func: object,  # faker methods are objects
        size: int,
    ) -> list[str]:
        """
        Generate a pool of unique values using a Faker generator function.

        Uses a set to track uniqueness, falling back to duplicates if
        uniqueness cannot be achieved within reasonable iterations.

        Args:
            generator_func: Faker method to call (e.g., faker.name)
            size: Target pool size

        Returns:
            List of generated values
        """
        # Type check hack because Faker methods are dynamic
        gen_func = generator_func

        pool: list[str] = []
        seen: set[str] = set()
        max_attempts = size * 3  # Allow 3x attempts for uniqueness
        attempts = 0  # nosemgrep

        while len(pool) < size and attempts < max_attempts:
            # We assume generator_func is a callable that returns a string
            # In strict typing this is hard to represent without protocols
            value = str(gen_func())  # type: ignore
            if value not in seen:
                seen.add(value)
                pool.append(value)
            attempts += 1

        # If we couldn't get enough unique values, fill with duplicates
        while len(pool) < size:
            pool.append(str(gen_func()))  # type: ignore

        return pool

    def sample_cities(self, n: int, replace: bool = True) -> list[dict[str, Any]]:
        """Sample n city objects (with lat/lon) from the curated GIS pool."""
        if n == 0:
            return []

        pool_size = len(self.cities)
        if not replace and n > pool_size:
            replace = True

        indices = self._rng.choice(pool_size, size=n, replace=replace)
        return [self.cities[i] for i in indices]

    def _sample_from_pool(
        self,
        pool: list[str],
        n: int,
        replace: bool = True,
    ) -> list[str]:
        """
        Sample n items from a pool using vectorized NumPy indexing.

        Args:
            pool: List to sample from
            n: Number of samples
            replace: Whether to sample with replacement (default True)

        Returns:
            List of sampled values
        """
        if n == 0:
            return []

        pool_size = len(pool)

        # For sampling without replacement, n must not exceed pool size
        if not replace and n > pool_size:
            replace = True  # nosemgrep

        indices = self._rng.choice(pool_size, size=n, replace=replace)
        return [pool[i] for i in indices]

    def sample_names(self, n: int, replace: bool = True) -> list[str]:
        """Sample n full names from the pre-generated pool."""
        return self._sample_from_pool(self.names, n, replace)

    def sample_companies(self, n: int, replace: bool = True) -> list[str]:
        """Sample n company names from the pre-generated pool."""
        return self._sample_from_pool(self.companies, n, replace)

    def sample_emails(self, n: int, replace: bool = True) -> list[str]:
        """Sample n email addresses from the pre-generated pool."""
        return self._sample_from_pool(self.emails, n, replace)

    def sample_addresses(self, n: int, replace: bool = True) -> list[str]:
        """Sample n street addresses from the pre-generated pool."""
        return self._sample_from_pool(self.addresses, n, replace)

    def sample_phone_numbers(self, n: int, replace: bool = True) -> list[str]:
        """Sample n phone numbers from the pre-generated pool."""
        return self._sample_from_pool(self.phone_numbers, n, replace)

    def sample_first_names(self, n: int, replace: bool = True) -> list[str]:
        """Sample n first names from the pre-generated pool."""
        return self._sample_from_pool(self.first_names, n, replace)

    def sample_last_names(self, n: int, replace: bool = True) -> list[str]:
        """Sample n last names from the pre-generated pool."""
        return self._sample_from_pool(self.last_names, n, replace)

    def sample(
        self,
        pool_name: str,
        n: int,
        replace: bool = True,
    ) -> list[str]:
        """
        Generic sampling method for any pool by name.

        Args:
            pool_name: Name of pool ("names", "companies", "cities", etc.)
            n: Number of samples
            replace: Whether to sample with replacement

        Returns:
            List of sampled values

        Raises:
            KeyError: If pool_name is not a valid pool
        """
        if pool_name not in self._pool_arrays:
            raise KeyError(
                f"Unknown pool '{pool_name}'. "
                f"Valid pools: {list(self._pool_arrays.keys())}"
            )
        return self._sample_from_pool(self._pool_arrays[pool_name], n, replace)

    def get_pool_sizes(self) -> dict[str, int]:
        """Return the actual sizes of all pools."""
        return {name: len(pool) for name, pool in self._pool_arrays.items()}

    def reset_rng(self, seed: int | None = None) -> None:
        """
        Reset the random number generator.

        Args:
            seed: New seed (uses original seed if None)
        """
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)
