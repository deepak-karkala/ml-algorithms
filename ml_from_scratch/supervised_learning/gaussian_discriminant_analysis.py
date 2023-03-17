import numpy as np

def calculate_covariance_matrix(x, y=None):
        """Calculates covariance matrix XtX"""
        if y is None:
            y = x
        num_samples = x.shape[0]
        covariance_matrix = (1/(num_samples-1)) * (x - x.mean(axis=0)).T @ (y - y.mean(axis=0))
        return np.array(covariance_matrix, dtype=float)


class GDA(object):
    """Gaussian Discriminant Analysis
    Super class of LDA and QDA
    """
    def __init__(self):
        self.parameters = []
        return
    
    def fit(self, x, y):
        """Fit parameters: prior probabilities, class wise mean and
        variance (common for LDA, class wise for QDA) 
        Assumes samples in each class are Gaussian distributed
        """
        self.classes = np.unique(y)
        #self.parameters = np.empty((1, len(self.classes)))
        self.parameters = []
        # Find mean, covariance matrix for each class
        for i, c in enumerate(self.classes):
            x_class = x[np.where(y==c)]
            """
            params = []
            params["mean"] = x_class.mean(axis=0)
            params["covariance"] = calculate_covariance_matrix(x_class)
            params["prior"] = len(x_class) / len(x)
            """
            params = {  "mean": x_class.mean(axis=0),
                        "covariance": calculate_covariance_matrix(x_class),
                        "prior": len(x_class) / len(x)}
            self.parameters.append(params)
    
    def predict(self, x):
        """Predict class output
        Assign sample to the class with highest posterior probability
        """
        y_pred = []
        for sample in x:
            # Get discriminant function value for each class
            discriminants = self.discriminant_function(sample, self.parameters)
            # Assign to class with highest discriminant function
            y_pred.append(self.classes[np.argmax(discriminants)])
        return y_pred


class LDA(GDA):
    """Linear Discriminant analysis
    Class specific mean and common covariance matrix
    Linear boundary b/w classes
    """
    def _lda_discriminant(self, x, params):
        common_covariance = 0
        for i, params in enumerate(self.parameters):
            # Common covariance across classes (sum all)
            common_covariance += params["covariance"]

        #discriminants = np.zeros((1, len(self.classes)))
        discriminants = []

        for i, params in enumerate(self.parameters):
            # Mean and prior for each class
            mean = np.expand_dims(params["mean"], axis=1)
            prior = params["prior"]
            # Inverse of common covariance matrix
            sigma_inv = np.linalg.pinv(common_covariance)
            # LDA discriminant function
            discriminant = (x.T @ sigma_inv @ mean) - 0.5 * (mean.T @ sigma_inv @ mean) + np.log(prior+1e-7)
            discriminants.append(discriminant)
        return np.array(discriminants)


    def predict(self, x):
        self.discriminant_function = self._lda_discriminant
        return super(LDA, self).predict(x)
    
    def _calculate_scatter_matrices(self, x, y):
        """Calculate scatter matrices
        """
        n_samples, n_features = np.shape(x)
        
        # Within class scatter
        sw = np.empty((n_features, n_features))
        for label in self.classes:
            x_class = x[y == label]
            sw += (len(x_class)-1) * calculate_covariance_matrix(x_class)

        # Across class scatter
        sb = np.empty((n_features, n_features))
        mean_overall = np.mean(x, axis=0)
        for label in self.classes:
            x_class = x[y == label]
            mean_class = np.mean(x_class, axis=0)
            sb += len(x_class) * (mean_class - mean_overall).dot((mean_class - mean_overall).T)
        return sw, sb

    def transform(self, x, y, n_components):
        """Project to lower dimensions (onto LDA decision boundary)
        """
        sw, sb = self._calculate_scatter_matrices(x, y)
        # [TODO: Understand this operation]
        a = np.linalg.inv(sw).dot(sb)

        # Get eigenvectors, eigenvalues of this feature space
        eigenvalues, eigenvectors = np.linalg.eigh(a)

        # Get first n_components eigenvectors in sorted order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        # Transform data by projecting onto eigenvectors
        x_transform = x.dot(eigenvectors)
        return x_transform
    
    def plot_in_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation."""
        X_transformed = self.transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        if title: plt.title(title)
        plt.show()


class QDA(GDA):
    """Quadratic Discriminant analysis
    Class specific mean and class specific covariance matrix
    Quadratic boundary b/w classes
    """
    def _qda_discriminant(self, x, params):
        #discriminants = np.empty((1, len(self.classes)))
        discriminants = []
        for i, params in enumerate(self.parameters):
            # Mean and prior for each class
            mean = np.expand_dims(params["mean"], axis=1)
            prior = params["prior"]
            covariance = params["covariance"]
            # Inverse of common covariance matrix
            sigma_inv = np.linalg.pinv(covariance)
            # LDA discriminant function
            discriminant = -0.5 * (x.T @ sigma_inv @ x)  + (x.T @ sigma_inv @ mean) -\
                0.5 * (mean.T @ sigma_inv @ mean) - 0.5 * (np.log(np.linalg.det(sigma_inv)+1e-7)) +\
                np.log(prior+1e-7)
            discriminants.append(discriminant)
        return np.array(discriminants)


    def predict(self, x):
        self.discriminant_function = self._qda_discriminant
        return super(QDA, self).predict(x)
