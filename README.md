
[![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)

`rari` is a Python implementation of Pinto et. al's ranked adjusted Rand index (RARI) from
[Ranked Adjusted Rand: integrating distance and partition information in a measure of clustering agreement](https://doi.org/10.1186/1471-2105-8-44).
The ranked adjusted Rand index is an extension of the [adjusted Rand index](https://en.wikipedia.org/wiki/Rand_index)
that measures the agreement between two independent clustering solutions while incorporating distances
between instances/clusters from each solution. An index of 1 indicates perfect agreement between solutions while an
index close to 0 indicates random labeling and distances.

The benefit of RARI is in penalizing the adjusted Rand index when a given pair of instances is close together in cluster
solution 'A' and far apart in cluster solution 'B'.

## Lightning Example
* Below is a comparison of the agreement between hierarchical and k-means clustering solutions on the iris data set. The
same distance matrix is used to calculate pairwise distances between each iris instance, but this is not a requirement.

``` python
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from rari import rari

X = load_iris().data

model_1 = AgglomerativeClustering(n_clusters=3, linkage='ward')
x = model_1.fit_predict(X)

model_2 = KMeans(n_clusters=3)
y = model_2.fit_predict(X)

dist_x = pairwise_distances(X, metric='euclidean')
dist_y = pairwise_distances(X, metric='euclidean')

rari(x, y, dist_x, dist_y)
```
Out[1]: **.975**

## Install

* Development

``` python
pip install git+https://github.com/nredell/rari
```

## Examples

### Example 1: ARI vs. RARI, Few Clusters, High Agreement

* Results/interpretation TBD

``` python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from rari import rari

X, y = make_blobs(n_samples=[50, 50, 50], n_features=2, cluster_std=1.0, center_box=(-5.0, 5.0), shuffle=True, random_state=224)
data = pd.DataFrame(np.hstack([X, y[:, np.newaxis]]), columns=["X1", "X2", "Cluster"])

model_1 = AgglomerativeClustering(n_clusters=3, linkage='ward')
x = model_1.fit_predict(X)

model_2 = KMeans(n_clusters=3)
y = model_2.fit_predict(X)

dist_x = pairwise_distances(X, metric='euclidean')
dist_y = pairwise_distances(X, metric='euclidean')
```

![](./tools/example_1_plot_1.png) ![](./tools/example_1_plot_2.png) ![](./tools/example_1_plot_3.png)

``` python
adjusted_rand_score(x, y)
rari(x, y, dist_x, dist_y)
```
**ARI:** .83
**RARI:** .89
