feature-grouper
===============

.. image:: https://badge.fury.io/py/feature-grouper.svg
    :target: https://badge.fury.io/py/feature-grouper
.. image:: https://readthedocs.org/projects/feature-grouper/badge/?version=latest
    :target: https://feature-grouper.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Simple, hierarchical clustering-based dimensionality reduction.

Calculates a correlation matrix of all features in X, applies hierarchical
clustering to create flat clusters of highly correlated features,
then generates and applies a loading matrix that evenly weights the input
features within each cluster.

Use as a simplified alternative to dimensionality reduction methods like PCA
when you need highly interpretable loadings.
