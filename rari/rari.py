"""Main module."""

import numpy as np
from scipy.stats import rankdata
from pandas import Categorical, crosstab

def rari(x, y, dist_x, dist_y):

    """ Computes the ranked adjusted Rand index for measuring clustering agreement.

        An implementation of Pinto et. al's ranked adjusted Rand index which is an extension of the adjusted Rand index
        that measures the agreement between two independent clustering solutions while incorporating distances
        between instances/clusters in each solution. An index of 1 indicates perfect agreement between solutions while
        and index close to 0 indicates random labeling and distances.

        Pinto, F. R., Carriço, J. A., Ramirez, M., & Almeida, J. S. (2007). Ranked Adjusted Rand: integrating distance
            and partition information in a measure of clustering agreement. BMC bioinformatics, 8(1), 44.
            https://doi.org/10.1186/1471-2105-8-44

        Parameters
        ----------
        x : np.array
            A numeric array of cluster membership values with 1 value per instance. Cluster values should start at either
            0 or 1. The number of clusters can differ between 'x' and 'y'; however, len(x) must equal len(y).

        y : np.array
            A numeric array of cluster membership values with 1 value per instance. Cluster values should start at either
            0 or 1. The number of clusters can differ between 'x' and 'y'; however, len(x) must equal len(y).

        dist_x : A 2-D np.array
            A len(x) by len(x) matrix giving the distances between instances in 'x'. The distance metric used to
            create 'dist_x' can be different from that used to create 'dist_y'.

        dist_y : A 2-D np.array
            A len(y) by len(y) matrix giving the distances between instances in 'y'. The distance metric used to
            create 'dist_y' can be different from that used to create 'dist_x'.

        Returns
        ----------
        rari : float
        """

    n_instances = x.size

    clusters_x = np.sort(np.unique(x))
    clusters_y = np.sort(np.unique(y))

    # Cluster membership should be a numeric vector starting at 0 or 1.
    if 0 not in clusters_x:
        clusters_x = clusters_x - 1

    if 0 not in clusters_y:
        clusters_y = clusters_y - 1

    n_clusters_x = len(clusters_x)
    n_clusters_y = len(clusters_y)
    # -------------------------------------------------------------------------------
    # Determine which instances in x and y belong to which cluster.
    clust_instances_x = []
    for i in clusters_x:
        clust_instances_x.append(np.arange(n_instances)[i == x])

    clust_instances_y = []
    for i in clusters_y:
        clust_instances_y.append(np.arange(n_instances)[i == y])
    # -------------------------------------------------------------------------------
    # Separately for each cluster solution 'x' and 'y', for each instance in the input
    # distance matrix, compute the average distance between the select instance and
    # all instances in each cluster. The result is an n_instance by n_cluster matrix.
    # TODO: Add additional methods--beyond mean()--for computing pairwise distances.
    clust_dist_x = []
    for cluster_instances in clust_instances_x:
        for instance in range(n_instances):
            clust_dist_x.append(np.nanmean(dist_x[instance, cluster_instances]))
    clust_dist_x = np.array(clust_dist_x).reshape(n_instances, -1, order='F')

    clust_dist_y = []
    for cluster_instances in clust_instances_y:
        for instance in range(n_instances):
            clust_dist_y.append(np.nanmean(dist_y[instance, cluster_instances]))
    clust_dist_y = np.array(clust_dist_y).reshape(n_instances, -1, order='F')

    clust_dist_x = np.hstack([clust_dist_x, x[:, np.newaxis]])
    clust_dist_y = np.hstack([clust_dist_y, y[:, np.newaxis]])
    # -------------------------------------------------------------------------------
    # Eqn 4., Table 3. rmm is the rank mismatch matrix indexed from 0 to number of clusters -1.

    # The algorithm proceeds in two steps:

    # First, compute the n_clusters by n_clusters matrix of distances between clusters for each
    # cluster solution x and y. The distance matrix may be asymmetric. As a result, the matrix
    # should be interpreted as "the cluster in column j is [this distance] from the cluster in
    # row i."
    clust_dist_matrix_x = []
    for column in range(n_clusters_x):
        for cluster_instances in clust_instances_x:
            clust_dist_matrix_x.append(np.nanmean(clust_dist_x[cluster_instances, column]))
    clust_dist_matrix_x = np.array(clust_dist_matrix_x).reshape(n_clusters_x, n_clusters_x, order='C')

    clust_dist_matrix_y = []
    for column in range(n_clusters_y):
        for cluster_instances in clust_instances_y:
            clust_dist_matrix_y.append(np.nanmean(clust_dist_y[cluster_instances, column]))
    clust_dist_matrix_y = np.array(clust_dist_matrix_y).reshape(n_clusters_y, n_clusters_y, order='C')

    # As the inter-cluster distances will be rank ordered, all instances in a cluster have a 0 inter-cluster distance.
    np.fill_diagonal(clust_dist_matrix_x, 0)
    np.fill_diagonal(clust_dist_matrix_y, 0)

    # Rank the distances between clusters. This matrix may be asymmetric. Subtracting 1 for easier indexing.
    for i in range(n_clusters_x):
        clust_dist_matrix_x[:, i] = rankdata(clust_dist_matrix_x[:, i], method='dense') - 1

    for i in range(n_clusters_y):
        clust_dist_matrix_y[:, i] = rankdata(clust_dist_matrix_y[:, i], method='dense') - 1

    # Second, compute instance-level pairwise differences in cluster distance ranks and
    # set the resulting n_instance by n_instance diagonal to NA to remove the bias in calculating
    # an instances distance from itself.
    instance_dist_rank_matrix_x = []
    for i in range(n_instances):
        for j in range(n_instances):
            if i != j:  # Heaviside function.
                instance_dist_rank_matrix_x.append(clust_dist_matrix_x[x[i], x[j]])
            else:
                instance_dist_rank_matrix_x.append(np.nan)

    instance_dist_rank_matrix_y = []
    for i in range(n_instances):
        for j in range(n_instances):
            if i != j:  # Heaviside function.
                instance_dist_rank_matrix_y.append(clust_dist_matrix_y[y[i], y[j]])
            else:
                instance_dist_rank_matrix_y.append(np.nan)

    rmm = crosstab(Categorical(instance_dist_rank_matrix_x), Categorical(instance_dist_rank_matrix_y)).values
    # -------------------------------------------------------------------------------
    # Eqn 5. mdd is the mean diagonal deviance. Note: the Rand index = 1 - mdd.
    mdd_numerator = []
    for i in range(rmm.shape[0]):
        for j in range(rmm.shape[1]):
            mdd_numerator.append(rmm[i, j] * abs(i / rmm.shape[0] - j / rmm.shape[1]))
    mdd = sum(mdd_numerator) / (n_instances ** 2 - n_instances)
    # -------------------------------------------------------------------------------
    # Eqn 6. rmmi is the expected value of the rank mismatch matrix under independence.
    rmmi = np.zeros((rmm.shape[0], rmm.shape[1]))
    for i in range(rmm.shape[0]):
        for j in range(rmm.shape[1]):
            rmmi[i, j] = np.sum(rmm[i, :]) * (np.sum(rmm[:, j]) / np.sum(rmm))
    # -------------------------------------------------------------------------------
    # Eqn 5. mddi is the mean diagonal deviance under independence.
    mddi_numerator = []
    for i in range(rmm.shape[0]):
        for j in range(rmm.shape[1]):
            mddi_numerator.append(rmmi[i, j] * abs(i / rmmi.shape[0] - j / rmmi.shape[1]))
    mddi = sum(mddi_numerator) / (n_instances ** 2 - n_instances)
    # -------------------------------------------------------------------------------
    # Eqn. 7.
    rari = (mddi - mdd) / mddi

    return rari
