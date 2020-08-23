"""
A set of functions and an sklearn transformer class for finding clusters of correlated
features and grouping them together into feature groups.
"""
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def find_clusters(corr, threshold=0.5):
    """
    Find clusters of correlated features from a correlation matrix
    using hierarchical clustering.

    :param corr: A n x n symmetrical correlation coefficient matrix
    :param threshold: The minimum correlation similarity threshold to group
           descendants of a cluster node into the same flat cluster.
    """
    dissimilarity = 1.0 - corr
    # Ensure that the diagonal is 0 after floating point calculation
    diss_thresh = 1.0 - threshold
    hierarchy = linkage(
        squareform(dissimilarity, checks=False), method="single"
    )
    # Make labels 0-indexed
    labels = fcluster(hierarchy, diss_thresh, criterion="distance") - 1
    return labels


def make_loading_matrix(labels, threshold=0.5):
    """
    Generate a loading matrix from the feature cluster labels, given
    a minimum correlation similarity threshold.

    Apply the loading matrix to the original data with ``np.matmul``.

    :Example:

    >>> import numpy as np
    >>> import feature_grouper
    >>> threshold = 0.5
    >>> corr = np.corrcoef(X.T)
    >>> clusters = feature_grouper.find_clusters(corr, threshold)
    >>> loading_matrix = feature_grouper.make_loading_matrix(clusters, threshold)
    >>> X_transformed = np.matmul(X, loading_matrix)

    :param labels: A numpy 1d array containing the cluster number label
           for each column in the original dataset.
    :param threshold: The minimum correlation similarity threshold that was
           used to cluster the features.
    """
    label_counts = Counter(labels)
    loading_matrix = np.zeros((len(labels), len(label_counts)))
    for feature, label in zip(range(len(labels)), labels):
        if label_counts[label] > 1:
            loading_matrix[feature, label] = 1.0 / (
                np.sqrt(threshold) * float(label_counts[label])
            )
        else:
            loading_matrix[feature, label] = 1.0

    return loading_matrix


class FeatureGrouper(BaseEstimator, TransformerMixin):
    """
    Hierarchical clustering-based dimensionality reduction.

    Calculates correlation matrix of all features in X, applies hierarchical
    clustering to create flat clusters of highly correlated features,
    then generates and applies a loading matrix that evenly weights the input
    features within each cluster.

    Input features should be normalized (i.e. z-scores).

    :param threshold: The minimum correlation similarity threshold to group
           descendants of a cluster node into the same flat cluster.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.loading_matrix_ = None

    def fit(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X."""
        corr = np.corrcoef(X.T)
        labels = find_clusters(corr, self.threshold)
        self.loading_matrix_ = make_loading_matrix(labels, self.threshold)
        return self

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X."""
        check_is_fitted(self)
        return np.matmul(X, self.loading_matrix_)
