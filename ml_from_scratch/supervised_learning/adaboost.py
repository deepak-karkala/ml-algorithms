import numpy as np
import math
import sys
import matplotlib.pyplot as plt

class DecisionStump():
    """One level decision tree
    """
    def __init__(self):
        # Flag to indicate whether sample is classified as
        # 1 or -1 based on threshold comparison
        self.polarity = 1
        # Feature index used to split
        self.feature_index = None
        # Threshold value used to split
        self.threshold = None
        # Alpha - indicative of classifier's accuracy
        self.alpha = None


class Adaboost():
    """Adaboost classifier
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
    
    def fit(self, x, y):
        """Fit a set of weak learners
        Learn which feature,threshold to set and corresponding alpha
        (classifier accuracy) which will be used to weight in prediction
        """
        n_samples, n_features = x.shape
        self.clfs = []
        w = np.full(n_samples, (1/n_samples))

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            # Find the best feature to split on
            for feature_i in range(n_features):
                feature_values = np.expand_dims(x[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Find the best threshold to split on
                for threshold in unique_values:
                    # Init predictions to all ones
                    pred = np.ones(np.shape(y))
                    # Set polarity to 1
                    polarity = 1
                    # Set -1 when feature values are less than threshold
                    negative_idx = x[:, feature_i] < threshold
                    pred[negative_idx] = -1

                    # Calculate error (sum of weighted samples)
                    error = sum(w[y != pred])

                    # If error is > 0.5, flip predictions
                    if error > 0.5:
                        polarity = -1
                        error = 1 - error

                    # Record this feature, threshold if error is better
                    # than previous best
                    if min_error > error:
                        min_error = error
                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_index = feature_i
            
            # Calculate alpha for the best classifier (based on min_error)
            clf.alpha = 0.5 * math.log((1 - min_error)/(min_error + 1e-10))

            # Get predictions for this best classifier
            pred = np.ones(np.shape(y))
            negative_idx = (clf.polarity * x[:, clf.feature_index] < clf.polarity * clf.threshold)
            pred[negative_idx] = -1

            # Weight samples based on misclassifications 
            # The ones which are misclassified will get larger weights
            w *= np.exp(-clf.alpha * y * pred)
            # Normalise weights
            w /= sum(w)

            # Append this classifier
            self.clfs.append(clf)
    
    def predict(self, x):
        """ Get predictions for adaboost model
        Weighted (alpha) sum of each weak learner's output
        """
        n_samples = np.shape(x)[0]
        y_pred = np.zeros((n_samples,1))

        for clf in self.clfs:
            pred = np.ones((n_samples, 1))
            negative_idx = (clf.polarity * x[:, clf.feature_index] < clf.polarity * clf.threshold)
            pred[negative_idx] = -1
            # Weighted (alpha) sum of predictions from weak learners
            y_pred += clf.alpha * pred
        y_pred = np.sign(y_pred).flatten()
        return y_pred
