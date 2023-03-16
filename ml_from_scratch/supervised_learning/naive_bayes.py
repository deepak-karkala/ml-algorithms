import numpy as np
import math

class NaiveBayes(object):
    """Naive Bayes Classifier (Gaussian)
    """

    def fit(self, x, y):
        # Fit parameters (Since continuous variables,
        # find mean, variance of each feature for each class)

        self.classes = np.unique(y)
        n_samples, n_features = np.shape(x)
        self.y = y
        self.parameters = []

        # Find parameters for each class        
        for i, c in enumerate(self.classes):
            # Filter datapoints belonging to ith class
            x_class = x[y == c]
            self.parameters.append([])
            # Iterate through each feature and find mean,variance
            for feature_i in range(n_features):
                parameters = {"mean":x_class[:, feature_i].mean(),  "var": x_class[:, feature_i].var()}
                self.parameters[i].append(parameters)
    
    def _calculate_prior(self, c):
        """Ratio of samples belonging to this class"""
        return len(self.y[self.y==c]) / len(self.y)

    def _calculate_likelihood(self, mean, var, x):
        """Return probability of lying on Gaussian defined by mean and var
        """
        eps = 1e-4
        coef = 1 / math.sqrt(2 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2)) / (2 * var + eps))
        return coef * exponent

    def _classify(self, x):
        """Assign sample to class with highest posterior probability
        P(Y|X) = P(X|Y) * P(Y) / P(X)
        P(X|Y): Likelihood of seeing data X given class Y
        P(Y): Prior (Ratio of samples belonging to this class)
        P(Y|X): Posterior likelihood: Probability of class Y given X
        Naive Bayes Assumption
        P(X1, X2, X3|Y) = P(X1|Y) * P(X2|Y) * P(X3|Y)
        """
        posteriors = []
        for i, c in enumerate(self.classes):
            # Compute prior for each class
            posterior = self._calculate_prior(c)

            # Compute likelihood of feature given each class
            # Get posterior = likelihood * prior for each class
            for params, feature_value in zip(self.parameters[i], x):
                posterior *= self._calculate_likelihood(params["mean"], params["var"], feature_value)
            posteriors.append(posterior)

        # Assign sample to the class with highest posterior probability 
        return self.classes[np.argmax(posteriors)]

    def predict(self, x):
        """Get class output
        """
        y_pred = [self._classify(sample) for sample in x]
        return np.array(y_pred)
