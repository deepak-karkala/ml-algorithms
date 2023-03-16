import numpy as np
import cvxopt

# Define kernels
def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2)**2
        return np.exp(-gamma * distance)


class SupportVectorMachine(object):
    """Support vector classifier
    Parameters
    ----------
    C: float
        Penalty term
    kernel: function
        Kernel function (linear, gaussian, polynomial)
    gamma: float
    power: int
    coef: float
        Bias term in polynomial function
    """
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.power = power
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vectors_labels = None
        self.intercept = None
    
    def fit(self, x, y):
        """Find SVM parameters
        (Lagrange multipliers, corresponding support vectors)
        """
        n_samples, n_features = np.shape(x)

        if not self.gamma:
            self.gamma = 1/n_features
        
        # Instantiate kernel
        self.kernel = self.kernel(
            power = self.power,
            coef = self.coef,
            gamma = self.gamma
        )

        # Find kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(x[i], x[j])
        
        # Solve the quadratic optimization problem using cvxopt package
        # [TODO: Understand how to formulate QP optimisation]
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        lagr_multipliers = np.ravel(minimization['x'])

        # Select non-zero lagr_multipliers (alpha)
        idx = lagr_multipliers > 1e-7
        self.lagr_multipliers = lagr_multipliers[idx]
        # Select corresponding support vectors and y labels
        self.support_vectors = x[idx]
        self.support_vectors_labels = y[idx]

        # Calculate intercept (TODO: How?)
        self.intercept = self.support_vectors_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vectors_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])
    
    def predict(self, x):
        """ Get SVM prediction
        (Weighted (lagr_multipliers) sum of kernel outputs with support vectors)
        """
        y_pred = []
        for sample in x:
            pred = 0
            # Iterate over support vectors
            for i in range(len(self.lagr_multipliers)):
                pred += self.lagr_multipliers[i] * self.support_vectors_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            pred += self.intercept
            y_pred.append(np.sign(pred))
        return np.array(y_pred)
