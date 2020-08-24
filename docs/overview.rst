Overview
========

The ``feature-grouper`` package provides functions and a scikit-learn
transformer class for applying a simple yet effective form of
dimensionality reduction based on hierarchical clustering of correlated
features.

Example usage:

.. code:: python

   import numpy as np
   import pandas as pd
   from sklearn import datasets
   import feature_grouper

   iris = datasets.load_iris()
   iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
   print(iris_df.head())
   """
      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
   0                5.1               3.5                1.4               0.2
   1                4.9               3.0                1.4               0.2
   2                4.7               3.2                1.3               0.2
   3                4.6               3.1                1.5               0.2
   4                5.0               3.6                1.4               0.2
   """

   threshold = 0.5 # correlation coefficient threshold for clustering
   fg = feature_grouper.FeatureGrouper(threshold)
   iris_trans = fg.fit_transform(iris_df)

   loadings = pd.DataFrame(fg.components_, columns=iris.feature_names)
   print(loadings)
   """
      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
   0           0.471405               0.0           0.471405          0.471405
   1           0.000000               1.0           0.000000          0.000000
   """

   # column 0 is now a linear combination of correlated features
   # sepal length, petal length, and petal width.
   print(iris_trans.head())
   """
             0    1
   0  3.158410  3.5
   1  3.064129  3.0
   2  2.922708  3.2
   3  2.969848  3.1
   4  3.111270  3.6
   """

