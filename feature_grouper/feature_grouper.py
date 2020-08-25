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


def cluster(X, threshold=0.5):
    """
    Find clusters of correlated features from a correlation matrix
    using hierarchical clustering.

    :param X: array-like, shape (n_samples, n_features)
               New data, where n_samples is the number of samples
               and n_features is the number of features.
    :param threshold: float The minimum correlation similarity threshold to group
           descendants of a cluster node into the same flat cluster.
    """
    dissim = 1.0 - np.corrcoef(X.T)
    hierarchy = linkage(squareform(dissim, checks=False), method="single")
    labels = fcluster(hierarchy, 1.0 - threshold, criterion="distance") - 1
    return labels


def make_loadings(labels, threshold=0.5):
    """
    Generate a loading matrix from the feature cluster labels, given
    a minimum correlation similarity threshold.

    Apply the loading matrix to the original data with ``np.matmul``
    or the ``@`` operator.

    :Example:

    >>> import numpy as np
    >>> import feature_grouper
    >>> threshold = 0.5
    >>> clusters = feature_grouper.cluster(X, threshold)
    >>> loading_matrix = feature_grouper.make_loading_matrix(clusters, threshold)
    >>> X_transformed = X @ loading_matrix

    :param labels: array-like, shape (n,)
           A numpy 1d array containing the cluster number label
           for each column in the original dataset.
    :param threshold: float The minimum correlation similarity threshold that was
           used to cluster the features.
    """
    label_counts = Counter(labels)
    loadings = np.zeros((len(label_counts), len(labels)))
    for feature, label in zip(range(len(labels)), labels):
        if label_counts[label] > 1:
            loadings[label, feature] = 1.0 / (
                np.sqrt(threshold) * float(label_counts[label])
            )
        else:
            loadings[label, feature] = 1.0

    return loadings


class FeatureGrouper(BaseEstimator, TransformerMixin):
    """
    Hierarchical clustering-based dimensionality reduction.

    Calculates correlation matrix of all features in X, applies hierarchical
    clustering to create flat clusters of highly correlated features,
    then generates and applies a loading matrix that evenly weights the input
    features within each cluster.

    Input features should be normalized (i.e. z-scores).

    :param threshold: float The minimum correlation similarity threshold to group
           descendants of a cluster node into the same flat cluster.
    :param copy: bool If False, data passed to transform are overwritten.
    :ivar components\_: array, shape (n_components, n_features)
          The loading matrix obtained from clustering and weighting
          correlated features.
    :ivar n_components\_: int The number of components that were estimated
          from the data.
    """

    def __init__(self, threshold=0.5, copy=True):
        self.threshold = threshold
        self.components_ = None
        self.copy = copy
        self.n_components_ = 0

    def fit(self, X, y=None):
        """
        Fit the model with X.

        :param X: array-like, shape (n_samples, n_features)
               New data, where n_samples is the number of samples
               and n_features is the number of features.
        """
        corr = np.corrcoef(X.T)
        labels = cluster(corr, self.threshold)
        self.components_ = make_loadings(labels, self.threshold)
        self.n_components_ = self.components_.shape[0]
        return self

    def transform(self, X):
        """
        Apply dimensionality reduction on X.

        :param X: array-like, shape (n_samples, n_features)
               New data, where n_samples is the number of samples
               and n_features is the number of features.
        """
        check_is_fitted(self)
        if not self.copy:
            X = X @ self.components_.T
            return X

        return X @ self.components_.T

    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.

        :param X: array-like, shape (n_samples, n_components)
               New data, where n_samples is the number of samples
               and n_components is the number of components.
        """
        return X @ self.components_
