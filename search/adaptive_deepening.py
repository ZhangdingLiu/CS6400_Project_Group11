"""
Adaptive Search Planner

Owner: Yichang Xu
Functionality: Dynamically adjust nprobe and k'
"""

import math
from typing import Optional, Tuple


class AdaptiveSearchPlanner:
    """Adaptive search parameter planner"""

    def __init__(
        self,
        nlist: int,
        nprobe_max: int = 256,
        k_prime_max_factor: int = 10,
        *,
        min_nprobe: int = 8,
        initial_nprobe_factor: float = 1.2,
        growth_factor_nprobe: float = 1.8,
        growth_factor_k_prime: float = 1.5,
        min_selectivity: float = 1e-4,
        initial_k_prime_factor: float = 1.4,
    ):
        """
        Initialize planner.

        Args:
            nlist: Total number of IVF lists
            nprobe_max: Maximum nprobe value
            k_prime_max_factor: Maximum k' as factor of k
            min_nprobe: Lower bound for nprobe
            initial_nprobe_factor: Controls aggressiveness of nprobe_0
            growth_factor_nprobe: Multiplicative growth per deepening step
            growth_factor_k_prime: Multiplicative growth for k'
            min_selectivity: Floor applied to selectivity estimates
            initial_k_prime_factor: Minimum expansion of k over k'
        """
        if nlist <= 0:
            raise ValueError("nlist must be positive.")

        self.nlist = int(nlist)
        self.nprobe_max = max(1, int(nprobe_max))
        self.k_prime_max_factor = max(1, int(k_prime_max_factor))

        self.min_nprobe = max(1, int(min_nprobe))
        self.initial_nprobe_factor = max(0.1, float(initial_nprobe_factor))
        self.growth_factor_nprobe = max(1.0, float(growth_factor_nprobe))
        self.growth_factor_k_prime = max(1.0, float(growth_factor_k_prime))
        self.min_selectivity = max(1e-6, float(min_selectivity))
        self.initial_k_prime_factor = max(1.0, float(initial_k_prime_factor))

        self._active_k_prime_cap: Optional[int] = None

    def _sanitize_selectivity(self, estimated_selectivity: Optional[float]) -> float:
        if estimated_selectivity is None or not math.isfinite(estimated_selectivity):
            return 0.2
        return float(min(1.0, max(self.min_selectivity, estimated_selectivity)))

    def initialize_parameters(self, k: int, estimated_selectivity: float) -> Tuple[int, int]:
        """
        Initialize search parameters.

        Args:
            k: Target number of results
            estimated_selectivity: Estimated filter selectivity

        Returns:
            Tuple[int, int]: (nprobe_0, k_prime_0)
        """
        if k <= 0:
            raise ValueError("k must be positive.")

        s = self._sanitize_selectivity(estimated_selectivity)
        max_nprobe = min(self.nprobe_max, self.nlist)

        rarity = 1.0 - math.sqrt(s)
        nprobe_float = rarity * self.initial_nprobe_factor * self.nlist
        nprobe = int(math.ceil(nprobe_float))
        nprobe = max(self.min_nprobe, min(max_nprobe, nprobe))

        self._active_k_prime_cap = k * self.k_prime_max_factor
        base_multiplier = max(self.initial_k_prime_factor, 1.0 / s)
        k_prime = int(math.ceil(k * base_multiplier))
        k_prime = max(k, min(self._active_k_prime_cap, k_prime))

        return nprobe, k_prime

    def should_deepen(self, n_results: int, k: int) -> bool:
        """
        Determine if search should be deepened.

        Args:
            n_results: Current number of results
            k: Target number of results

        Returns:
            bool: True if should deepen
        """
        if k <= 0:
            raise ValueError("k must be positive.")
        return n_results < k

    def grow_parameters(self, nprobe: int, k_prime: int) -> Tuple[int, int]:
        """
        Grow search parameters.

        Args:
            nprobe: Current nprobe
            k_prime: Current k'

        Returns:
            Tuple[int, int]: (new_nprobe, new_k_prime)
        """
        if nprobe <= 0 or k_prime <= 0:
            raise ValueError("nprobe and k_prime must be positive.")

        max_nprobe = min(self.nprobe_max, self.nlist)
        if nprobe >= max_nprobe:
            new_nprobe = max_nprobe
        else:
            grown = int(math.ceil(nprobe * self.growth_factor_nprobe))
            if grown == nprobe:
                grown += 1
            new_nprobe = min(max_nprobe, max(self.min_nprobe, grown))

        cap = self._active_k_prime_cap
        if cap is None:
            cap = k_prime * self.k_prime_max_factor

        if k_prime >= cap:
            new_k_prime = cap
        else:
            grown = int(math.ceil(k_prime * self.growth_factor_k_prime))
            if grown == k_prime:
                grown += 1
            new_k_prime = min(cap, grown)

        return new_nprobe, new_k_prime
