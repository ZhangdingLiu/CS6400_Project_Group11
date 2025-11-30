"""
Signature Storage

Owner: Zaowei Dai
Functionality: Persist and load metadata signatures
"""

from typing import Dict, Any
import json
import numpy as np


class SignatureStorage:
    """Persist and load metadata signatures"""

    def save(self, signatures: Dict, path: str):
        """
        Save signatures to file.

        Args:
            signatures: Signature dictionary
            path: File path
        """

        def convert(obj: Any):
            """Convert non-JSON-serializable objects."""
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (set,)):
                return list(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        with open(path, "w", encoding="utf-8") as f:
            json.dump(signatures, f, default=convert, indent=2)

    def load(self, path: str) -> Dict:
        """
        Load signatures from file.

        Args:
            path: File path

        Returns:
            Dict: Signature dictionary
        """
        with open(path, "r", encoding="utf-8") as f:
            signatures = json.load(f)
        return signatures
