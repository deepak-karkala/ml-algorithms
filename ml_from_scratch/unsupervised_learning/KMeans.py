import numpy as np
import random

class KMeans():
    """K-means clustering
    """

    def __init__(self, k=2, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        """Compute cluster centroids for given data
        """
        # Initialise cluster centroids to random k data points
        self.centroids = self._init_random_centroids(X)

        # Run for max_iterations
        for _ in range(self.max_iterations):
            # Assign each data point to a cluster based on closest centroid
            cluster_labels = self._assign_clusters(X, self.centroids)
            prev_centroids = self.centroids
            # Calculate new cluster centroids
            self.centroids = self._calculate_centroids(X, cluster_labels)
            # Stopping critera: If centroids haven't changed
            diff = self.centroids - prev_centroids
            if not diff.any():
                break
        

    def predict(self, X):
        """Assign cluster labels to given data
        """
        return self._assign_clusters(X, self.centroids)
    

    def _init_random_centroids(self, X):
        """Initialise cluster centroids to random k data points"""
        num_samples, num_features = X.shape
        self.centroids = np.zeros((self.k, num_features))
        self.centroids = X[random.sample(range(0, num_samples-1), self.k)]
        return self.centroids

    def _assign_clusters(self, X, centroids):
        """Assign each data point to a cluster based on closest centroid"""
        cluster_labels = np.zeros((X.shape[0], 1)).astype(int)
        for sample_i, sample in enumerate(X):
            euclidean_distance = np.sqrt(np.sum((X[sample_i] - centroids)**2, axis=1))
            cluster_labels[sample_i] = np.argmin(euclidean_distance)
        return cluster_labels

    def _calculate_centroids(self, X, cluster_labels):
        centroids = np.zeros((self.k, X.shape[1]))
        # Calculate new cluster centroids
        for i in range(self.k):
            # Get indices of data points belonging to ith cluster
            # cluster_i_idx = np.array(random.sample(range(0, X.shape[0]-1), int(X.shape[0]/self.k)))
            cluster_i_idx = np.array(cluster_labels==i)
            cluster_i_idx = cluster_i_idx[:,0]
            # Compute mean of such data points
            centroids[i] = X[cluster_i_idx, :].mean(axis=0)
        return centroids