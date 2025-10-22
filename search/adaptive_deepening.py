"""
Adaptive Search Planner

Owner: Yichang Xu
Functionality: Dynamically adjust nprobe and k'
"""

from typing import Tuple


class AdaptiveSearchPlanner:
    """Adaptive search parameter planner"""

    def __init__(self, nlist: int, nprobe_max: int = 256, k_prime_max_factor: int = 10):
        """
        Initialize planner.

        Args:
            nlist: Total number of IVF lists
            nprobe_max: Maximum nprobe value
            k_prime_max_factor: Maximum k' as factor of k
        """
        self.nlist = nlist
        self.nprobe_max = nprobe_max
        self.k_prime_max_factor = k_prime_max_factor

    def initialize_parameters(self, k: int, estimated_selectivity: float) -> Tuple[int, int]:
        """
        Initialize search parameters.

        Args:
            k: Target number of results
            estimated_selectivity: Estimated filter selectivity

        Returns:
            Tuple[int, int]: (nprobe_0, k_prime_0)
        """
        raise NotImplementedError

    def should_deepen(self, n_results: int, k: int) -> bool:
        """
        Determine if search should be deepened.

        Args:
            n_results: Current number of results
            k: Target number of results

        Returns:
            bool: True if should deepen
        """
        raise NotImplementedError

    def grow_parameters(self, nprobe: int, k_prime: int) -> Tuple[int, int]:
        """
        Grow search parameters.

        Args:
            nprobe: Current nprobe
            k_prime: Current k'

        Returns:
            Tuple[int, int]: (new_nprobe, new_k_prime)
        """
        raise NotImplementedError
