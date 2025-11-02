"""Unit tests for baselines and oracle"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.test_data_helpers import (
    generate_test_embeddings,
    generate_test_metadata,
    MockIVFPQIndex
)
from utils.filter_utils import apply_filter
from evaluation.oracle import ExactSearchOracle
from baselines.prefilter_bruteforce import PreFilterBruteForce
from baselines.postfilter_ann import PostFilterANN


def test_oracle_basic():
    embeddings = generate_test_embeddings(n=100, d=128)
    metadata = generate_test_metadata(n=100)
    oracle = ExactSearchOracle(embeddings, metadata)

    query = embeddings[0]
    filter_dict = {'category': {'op': 'IN', 'values': [1, 5, 10]}}

    distances, ids = oracle.search(query, filter_dict, k=10)

    assert len(distances) <= 10
    assert len(ids) <= 10
    assert len(distances) == len(ids)


def test_oracle_empty_filter():
    embeddings = generate_test_embeddings(n=100, d=128)
    metadata = generate_test_metadata(n=100)
    oracle = ExactSearchOracle(embeddings, metadata)

    query = embeddings[0]
    filter_dict = {}

    distances, ids = oracle.search(query, filter_dict, k=10)

    assert len(distances) == 10
    assert len(ids) == 10


def test_prefilter_correctness():
    embeddings = generate_test_embeddings(n=100, d=128)
    metadata = generate_test_metadata(n=100)

    oracle = ExactSearchOracle(embeddings, metadata)
    prefilter = PreFilterBruteForce(embeddings, metadata)

    query = embeddings[0]
    filter_dict = {'category': {'op': 'IN', 'values': [1, 5, 10]}}

    oracle_dist, oracle_ids = oracle.search(query, filter_dict, k=10)
    pre_dist, pre_ids = prefilter.search(query, filter_dict, k=10)

    assert len(oracle_ids) == len(pre_ids)
    np.testing.assert_array_equal(oracle_ids, pre_ids)


def test_postfilter_with_mock():
    embeddings = generate_test_embeddings(n=100, d=128)
    metadata = generate_test_metadata(n=100)

    mock_index = MockIVFPQIndex(embeddings)
    postfilter = PostFilterANN(mock_index, metadata)

    query = embeddings[0]
    filter_dict = {'category': {'op': 'IN', 'values': [1, 5, 10]}}

    distances, ids = postfilter.search(query, filter_dict, k=10)

    assert len(distances) <= 10
    assert len(ids) <= 10


def test_filter_in_operator():
    metadata = generate_test_metadata(n=100)
    filter_dict = {'category': {'op': 'IN', 'values': [1, 5, 10]}}

    indices = apply_filter(metadata, filter_dict)

    for idx in indices:
        assert metadata.iloc[idx]['category'] in [1, 5, 10]


def test_filter_range_operator():
    metadata = generate_test_metadata(n=100)
    filter_dict = {'year': {'op': 'RANGE', 'min': 20, 'max': 80}}

    indices = apply_filter(metadata, filter_dict)

    for idx in indices:
        year = metadata.iloc[idx]['year']
        assert 20 <= year <= 80


def test_combined_filters():
    metadata = generate_test_metadata(n=100)
    filter_dict = {
        'category': {'op': 'IN', 'values': [1, 2, 3]},
        'year': {'op': 'RANGE', 'min': 30, 'max': 70}
    }

    indices = apply_filter(metadata, filter_dict)

    for idx in indices:
        assert metadata.iloc[idx]['category'] in [1, 2, 3]
        year = metadata.iloc[idx]['year']
        assert 30 <= year <= 70


def test_no_matching_results():
    embeddings = generate_test_embeddings(n=100, d=128)
    metadata = generate_test_metadata(n=100)
    oracle = ExactSearchOracle(embeddings, metadata)

    query = embeddings[0]
    filter_dict = {'category': {'op': 'IN', 'values': [999]}}

    distances, ids = oracle.search(query, filter_dict, k=10)

    assert len(distances) == 0
    assert len(ids) == 0
