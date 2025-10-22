"""
Metadata Generator

Owner: Yao-Ting Huang
Functionality: Generate synthetic metadata attributes
"""

from typing import List
import numpy as np
import pandas as pd


class MetadataGenerator:
    """Generate synthetic metadata attributes"""

    def generate_category(self, n_samples: int) -> np.ndarray:
        """
        Generate category attribute with Zipf distribution.

        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=int32, range [1, 30]
        """
        raise NotImplementedError

    def generate_importance(self, n_samples: int) -> np.ndarray:
        """
        Generate importance attribute with normal distribution.

        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=int32, range [1, 100]
        """
        raise NotImplementedError

    def generate_year(self, n_samples: int) -> np.ndarray:
        """
        Generate year attribute with mixture distribution.

        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=int32, range [0, 100]
        """
        raise NotImplementedError

    def generate_region(self, n_samples: int) -> np.ndarray:
        """
        Generate region attribute.

        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=object, values from {NA, EU, APAC, LATAM, AFR}
        """
        raise NotImplementedError

    def compute_paragraph_len(self, texts: List[str]) -> np.ndarray:
        """
        Compute token count for each paragraph.

        Args:
            texts: List of paragraph texts

        Returns:
            np.ndarray: Shape (len(texts),), dtype=int32, token counts
        """
        raise NotImplementedError

    def build_metadata_table(self, n_samples: int, texts: List[str]) -> pd.DataFrame:
        """
        Build complete metadata DataFrame.

        Args:
            n_samples: Number of samples
            texts: List of paragraph texts

        Returns:
            pd.DataFrame: N rows × 5 columns [category, importance, year, paragraph_len, region]
        """
        raise NotImplementedError
