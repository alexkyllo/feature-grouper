"""pytest unit tests for the feature_grouper module"""
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
import feature_grouper


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
    assert feature_grouper.__version__ == "0.1.0"


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
    load_matrix = feature_grouper.make_loading_matrix(clusters, threshold)
    expected = np.array([[0.0, 1.0], [0.70710678, 0.0], [0.70710678, 0.0]])
    assert np.allclose(load_matrix, expected, rtol=0.01)


def test_fit_X():
    """Test that FeatureGrouper.fit() results in the correct loading matrix."""
    cov = np.array([[3.4, -2.75, -2.0], [-2.75, 5.5, 1.5], [-2.0, 1.5, 1.25]])
    features = get_correlated_features(cov, 30).T
    threshold = 0.5
    fg = feature_grouper.FeatureGrouper(0.5)
    fg.fit(features)
    expected = np.array([[0.0, 1.0], [0.70710678, 0.0], [0.70710678, 0.0]])
    assert np.allclose(fg.loading_matrix_, expected, rtol=0.01)


def test_fit_transform_X():
    cov = np.array([[3.4, -2.75, -2.0], [-2.75, 5.5, 1.5], [-2.0, 1.5, 1.25]])
    features = get_correlated_features(cov, 10).T
    threshold = 0.5
    fg = feature_grouper.FeatureGrouper(0.5)
    transformed = fg.fit_transform(features)
    expected = np.array(
        [
            [-4.07622073, 3.22583515],
            [-0.1235076, -0.52749254],
            [1.86870475, -0.89349396],
            [5.81390654, -4.89247768],
            [-1.47844261, -0.0152761],
            [-0.78533797, -0.58937111],
            [2.02293676, -0.98949565],
            [1.24869368, 0.58157378],
            [-0.838455, 0.77637916],
            [0.99094234, -1.96487481],
        ]
    )
    assert np.allclose(expected, transformed, rtol=0.01)


def test_pandas():
    """Test that the transformations work with a pandas.DataFrame input"""
    # TODO: write a test for pandas.DataFrame input
