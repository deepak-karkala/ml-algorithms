import numpy as np
import math

def _calculate_covariance_matrix(x, y=None):
        """Calculates covariance matrix XtX"""
        if y is None:
            y = x
        num_samples = x.shape[0]
        covariance_matrix = (1/(num_samples-1)) * (x - x.mean(axis=0)).T @ (y - y.mean(axis=0))
        return np.array(covariance_matrix, dtype=float)


class GaussianMixtureModel():
    """Probabilitstic clustering for Gaussian data
    Computes mean, covariance for clusters and assigns soft guesses
    (weights) for each sample.
    Uses Expectation-Maximisation algorithm to determine parameters
    Parameters
    ----------
    k: int
        Number of clusters
    max_iterations: int
        Max number of iterations for EM convergence
    tolerance: float
        Stop algorithm if change in parameters < tolerance
    """

    def __init__(self, num_clusters=2, max_iterations=500, tolerance=1e-8):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.parameters = []
        self.cluster_weights = None
        self.cluster_assignments = None
        #To check for convergence (keep appending over multiple iterations)
        self.cluster_weights_max = []

    def fit(self, x):
        """Fit parameters (mean, covariance, cluster weights)
        """
        # Initialise Gaussian distributions
        self._init_gaussians(x)

        # Run EM algorithm till convergence
        for _ in range(self.max_iterations):
            # Expectation step
            self._expectation(x)
            # Maximisation step
            self._maximisation(x)

            # Break if convergence
            if self._converged():
                break

    def predict(self, x):
        """Assign cluster assignments
        (after having fit the parameters)
        """
        self._expectation(x)
        return self.cluster_assignments
    
    def _init_gaussians(self, x):
        # Initialise Gaussian distributions randomly
        # Mean - random sample from X
        # Covariance - common for all classes
        num_samples = np.shape(x)[0]
        self.priors = (1/self.num_clusters) * np.ones(self.num_clusters)
        for i in range(self.num_clusters):
            params = {}
            params["mean"] = x[np.random.choice(range(num_samples))]
            params["covar"] = _calculate_covariance_matrix(x)
            self.parameters.append(params)
    
    def _expectation(self, x):
        """ E step of EM algorithm
        Calculate each sample's soft weights to different clusters
        """
        # Weight = posterior = likelihood * prior for each cluster
        posteriors = self._get_likelihoods(x) * self.priors
        sum_posteriors = np.expand_dims(np.sum(posteriors, axis=1), axis=1)
        self.cluster_weights = (posteriors/sum_posteriors)
        # Cluster assignments based on class with highest weight
        self.cluster_assignments = np.argmax(self.cluster_weights, axis=1)
        # Store weights across iterations to check for convergence
        self.cluster_weights_max.append(np.max(self.cluster_weights, axis=1))
    
    def _get_likelihoods(self, x):
        """ Get likelihood of data given parameters for each class"""
        num_samples = np.shape(x)[0]
        likelihoods = np.zeros((num_samples, self.num_clusters))
        for i in range(self.num_clusters):
            likelihoods[:, i] = self._multivariate_gaussian_likelihood(x, self.parameters[i])
        return likelihoods

    def _multivariate_gaussian_likelihood(self, x, params):
        """ Get prob density of Multivariate Gaussian distribution"""
        num_samples, num_features = np.shape(x)
        mean = params["mean"]
        covar = params["covar"]
        det = np.linalg.det(covar)
        mvn_likelihoods = np.zeros(num_samples)
        for i, sample in enumerate(x):
            coeff = 1 / ( (math.pow(2 * math.pi, num_features/2)) * (math.sqrt(det)) )
            exponent = np.exp(-0.5 * (sample-mean).T @ np.linalg.pinv(covar) @ (sample-mean) )
            mvn_likelihoods[i] = coeff * exponent
        return mvn_likelihoods

    def _maximisation(self, x):
        """ M-step of EM algorithm
        Determine, priors, means and covar of each class given soft weights of samples
        """
        for i in range(self.num_clusters):
            w = np.expand_dims(self.cluster_weights[:,i], axis=1)
            mean = (x * w).sum(axis=0) / w.sum()
            covar = (x - mean).T @ ((x - mean) * w) / w.sum()
            self.parameters[i]["mean"] = mean
            self.parameters[i]["covar"] = covar
        
        num_samples = np.shape(x)[0]
        self.priors = np.sum(self.cluster_weights, axis=0) / num_samples
    
    def _converged(self):
        """Check for convergence
        (If change in cluster weights is less than tol over 2 iterations)
        """
        if len(self.cluster_weights_max) < 2:
            return False
        diff = np.linalg.norm(self.cluster_weights_max[-1] - self.cluster_weights_max[-2])
        return diff <= self.tolerance
