"""
Metadata Generator

Owner: Yao-Ting Huang
Functionality: Generate synthetic metadata attributes
"""
from typing import List, Optional, Dict
import numpy as np
import pandas as pd


class MetadataGenerator:
    """Generate synthetic metadata attributes"""
    def __init__(self, seed: Optional[int] = 42) -> None:
        self.rng = np.random.default_rng(seed)
    
    def _clip_int(self, x: np.ndarray, lo: int, hi: int) -> np.ndarray:
        return np.clip(x, lo, hi).astype(np.int32)

    def generate_category(self, n_samples: int) -> np.ndarray:
        """
        Generate category attribute with Zipf distribution with s = 1.2 and truncated to 30
        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=int32, range [1, 30]
        """
        max_cat = 30 
        zipf_s = 1.2
        out = np.empty(n_samples, dtype=np.int32)
        i = 0
        while i < n_samples:
            draws = self.rng.zipf(zipf_s, size=max(1024, n_samples - i))
            keep = draws[draws <= max_cat]
            take = min(len(keep), n_samples - i)
            out[i : i + take] = keep[:take]
            i += take
        return out

    def generate_importance(self, n_samples: int) -> np.ndarray:
        """
        Generate importance attribute with normal distribution.

        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=int32, range [1, 100]
        """
        vals = self.rng.normal(loc=50, scale=15, size=n_samples)
        return self._clip_int(np.round(vals), 1, 100)

    def generate_year(self, n_samples: int) -> np.ndarray:
        """
        Generate year attribute with mixture distribution (30% low, 60% mid, 10% high). 
        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=int32, range [0, 100]
        """
        comps = self.rng.choice([0, 1, 2], size=n_samples, p=[0.3, 0.6, 0.1])
        low = (100 * self.rng.beta(a=2.0, b=8.0, size=n_samples)).astype(np.float32)
        mid = self.rng.integers(low=0, high=101, size=n_samples).astype(np.float32)
        high = (100 * self.rng.beta(a=8.0, b=2.0, size=n_samples)).astype(np.float32)
        vals = np.where(comps == 0, low, np.where(comps == 1, mid, high))
        return self._clip_int(np.round(vals), 0, 100)

    def generate_region(self, n_samples: int) -> np.ndarray:
        """
        Generate region attribute.

        Args:
            n_samples: Number of samples

        Returns:
            np.ndarray: Shape (n_samples,), dtype=object, values from {NA, EU, APAC, LATAM, AFR}
        """
        regions = np.array(["NA", "EU", "APAC", "LATAM", "AFR"], dtype=object)
        probs = np.array([0.4, 0.25, 0.2, 0.1, 0.05], dtype=np.float64)
        return self.rng.choice(regions, size=n_samples, p=probs)

    def compute_paragraph_len(self, texts: List[str]) -> np.ndarray:
        """
        Compute token count for each paragraph.

        Args:
            texts: List of paragraph texts

        Returns:
            np.ndarray: Shape (len(texts),), dtype=int32, token counts
        """
        return np.array([len(t.split()) for t in texts], dtype=np.int32)

    def build_metadata_table(self, n_samples: int, texts: List[str]) -> pd.DataFrame:
        """
        Build complete metadata DataFrame.

        Args:
            n_samples: Number of samples
            texts: List of paragraph texts

        Returns:
            pd.DataFrame: N rows × 5 columns [category, importance, year, paragraph_len, region]
        """
        if len(texts) != n_samples:
            raise ValueError(f"n_samples={n_samples} but len(texts)={len(texts)}")
        
        category = self.generate_category(n_samples)
        importance = self.generate_importance(n_samples)
        year = self.generate_year(n_samples)
        paragraph_len = self.compute_paragraph_len(texts)
        region = self.generate_region(n_samples)

        df = pd.DataFrame(
            {
                "category": category,
                "importance": importance,
                "year": year,
                "paragraph_len": paragraph_len,
                "region": region,
            }
        )
        df["category"] = df["category"].astype("int32")
        df["importance"] = df["importance"].astype("int32")
        df["year"] = df["year"].astype("int32")
        df["paragraph_len"] = df["paragraph_len"].astype("int32")
        df["region"] = df["region"].astype("string")

        return df