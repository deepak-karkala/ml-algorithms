import numpy as np

class PCA():
    """Principal component analysis
    """

    def __init__(self, num_components):
        self.num_components = num_components

    def _calculate_covariance_matrix(self, x, y=None):
        """Calculates covariance matrix XtX"""
        if y is None:
            y = x
        num_samples = x.shape[0]
        covariance_matrix = (1/(num_samples-1)) * (x - x.mean(axis=0)).T @ (y - y.mean(axis=0))
        return np.array(covariance_matrix, dtype=float)

    def transform(self, x):
        """ Projects the given data onto principal components
        """

        # Calculate covariance matrix
        covariance_matrix = self._calculate_covariance_matrix(x)

        # Get eigenvectors, eigenvalues using Eigen decomposition or SVD
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # First num_component eigenvectors (sorted in decreasing
        # # order of eigenvalues) are the principal components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:self.num_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :self.num_components]

        # Project data onto principal components
        x_transformed = x.dot(eigenvectors)
        return x_transformed
