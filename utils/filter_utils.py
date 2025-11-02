"""Filter utilities for metadata filtering"""

import numpy as np
import pandas as pd
from typing import Dict


def apply_filter(metadata_df: pd.DataFrame, filter_dict: Dict) -> np.ndarray:
    """
    Apply filter conditions to metadata DataFrame

    Args:
        metadata_df: Metadata DataFrame
        filter_dict: Filter conditions
            Example: {
                'category': {'op': 'IN', 'values': [1, 5, 10]},
                'year': {'op': 'RANGE', 'min': 20, 'max': 80}
            }

    Returns:
        np.ndarray: Boolean mask or indices of rows that pass the filter
    """
    if not filter_dict:
        return np.arange(len(metadata_df))

    mask = np.ones(len(metadata_df), dtype=bool)

    for field, condition in filter_dict.items():
        op = condition['op']

        if op == 'IN':
            values = condition['values']
            mask &= metadata_df[field].isin(values).values

        elif op == 'RANGE':
            min_val = condition['min']
            max_val = condition['max']
            mask &= (metadata_df[field] >= min_val).values & \
                   (metadata_df[field] <= max_val).values

        elif op == 'EQ':
            value = condition['value']
            mask &= (metadata_df[field] == value).values

        elif op == 'GT':
            value = condition['value']
            mask &= (metadata_df[field] > value).values

        elif op == 'GTE':
            value = condition['value']
            mask &= (metadata_df[field] >= value).values

        elif op == 'LT':
            value = condition['value']
            mask &= (metadata_df[field] < value).values

        elif op == 'LTE':
            value = condition['value']
            mask &= (metadata_df[field] <= value).values

    return np.where(mask)[0]
