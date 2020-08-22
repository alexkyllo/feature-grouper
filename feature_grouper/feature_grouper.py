"""
A set of functions and an sklearn transformer class for finding clusters of correlated
features and grouping them together into feature groups.
"""
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


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
    hierarchy = linkage(squareform(dissimilarity, checks=False), method="single")
    # Make labels 0-indexed
    labels = fcluster(hierarchy, diss_thresh, criterion="distance") - 1
    return labels


def make_load_matrix(labels, threshold):
    """
    """
    label_counts = Counter(labels)
    load_matrix = np.zeros((len(labels), len(label_counts)))
    for feature, label in zip(range(len(labels)), labels):
        if label_counts[label] > 1:
            load_matrix[feature, label] = 1.0 / (
                np.sqrt(threshold) * float(label_counts[label])
            )
        else:
            load_matrix[feature, label] = 1.0

    return load_matrix
