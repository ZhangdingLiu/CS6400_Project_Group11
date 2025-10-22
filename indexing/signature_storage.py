"""
Signature Storage

Owner: Zaowei Dai
Functionality: Persist and load metadata signatures
"""

from typing import Dict


class SignatureStorage:
    """Persist and load metadata signatures"""

    def save(self, signatures: Dict, path: str):
        """
        Save signatures to file.

        Args:
            signatures: Signature dictionary
            path: File path
        """
        raise NotImplementedError

    def load(self, path: str) -> Dict:
        """
        Load signatures from file.

        Args:
            path: File path

        Returns:
            Dict: Signature dictionary
        """
        raise NotImplementedError
