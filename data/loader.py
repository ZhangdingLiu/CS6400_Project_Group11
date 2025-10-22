"""
Wikipedia Dataset Loader

Owner: Yao-Ting Huang
Functionality: Load Wikipedia embedding dataset from HuggingFace
"""

from typing import List
import numpy as np


class WikipediaDataLoader:
    """Load Wikipedia embeddings dataset from HuggingFace"""

    def __init__(self, embedding_method: str = 'OpenAI'):
        """
        Initialize the data loader.

        Args:
            embedding_method: 'OpenAI', 'MiniLM', or 'GTE-small'
        """
        self.embedding_method = embedding_method

    def load_embeddings(self) -> np.ndarray:
        """
        Load normalized embeddings from the dataset.

        Returns:
            np.ndarray: Shape (N, d), normalized embeddings
        """
        raise NotImplementedError

    def load_text(self) -> List[str]:
        """
        Load paragraph texts from the dataset.

        Returns:
            List[str]: List of N paragraph texts
        """
        raise NotImplementedError
