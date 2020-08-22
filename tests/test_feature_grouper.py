"""pytest unit tests for the feature_grouper module"""
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
from feature_grouper import __version__, feature_grouper


def get_correlated_features(covariance_matrix, num_samples):
    """
    Generate num_samples given a square covariance matrix
    """
    # Ensure the covariance matrix is square
    shape = covariance_matrix.shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    c = cholesky(covariance_matrix, lower=True)
    xr = np.random.RandomState(11).normal(size=(shape[0], num_samples))
    X = np.dot(c, xr)
    return X


def test_version():
    assert __version__ == "0.1.0"


def test_find_clusters():
    """Test that the function finds expected clusters given example data"""
    # Features 1 and 2 have correlation of 0.6, both are negatively
    # correlated with Feature 0
    cov = np.array([[3.4, -2.75, -2.0], [-2.75, 5.5, 1.5], [-2.0, 1.5, 1.25]])
    features = get_correlated_features(cov, 30)
    corr = np.corrcoef(features)
    clusters = feature_grouper.find_clusters(corr, 0.1)
    assert np.array_equal(clusters, np.array([1, 0, 0]))
    clusters = feature_grouper.find_clusters(corr, 0.5)
    assert np.array_equal(clusters, np.array([1, 0, 0]))
    clusters = feature_grouper.find_clusters(corr, 0.7)
    assert np.array_equal(clusters, np.array([2, 0, 1]))


def test_make_loading_matrix():
    """Test that the expected loading matrix is given from the example data"""
    cov = np.array([[3.4, -2.75, -2.0], [-2.75, 5.5, 1.5], [-2.0, 1.5, 1.25]])
    features = get_correlated_features(cov, 30)
    corr = np.corrcoef(features)
    threshold = 0.5
    clusters = feature_grouper.find_clusters(corr, threshold)
    load_matrix = feature_grouper.make_load_matrix(clusters, threshold)
    expected = np.array([[0.0, 1.0], [0.70710678, 0.0], [0.70710678, 0.0]])
    assert np.allclose(load_matrix, expected, rtol=0.01)
