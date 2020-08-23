feature-grouper
===============

Hierarchical clustering-based dimensionality reduction.

Calculates correlation matrix of all features in X, applies hierarchical
clustering to create flat clusters of highly correlated features,
then generates and applies a loading matrix that evenly weights the input
features within each cluster.
