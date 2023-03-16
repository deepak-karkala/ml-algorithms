import numpy as np

def euclidean_distance(x, y):
    """Compute euclidean distance"""
    return np.linalg.norm(x-y, 2)

class KNN(object):
    """ Super class of KNN classifier and KNN Regressor
    Parameters
    ----------
    k: int
        Number of nearest neighbors to be used
    """
    def __init__(self, k=5):
        self.k = k
    
    def predict(self, x_test, x_train, y_train):
        """ Get output of KNN"""
        y_pred = np.empty(x_test.shape[0])
        for i, sample in enumerate(x_test):
            # Find distance to all training points and find the k-nearest ones
            k_nearest_idx = np.argsort([euclidean_distance(sample, training_data) for training_data in x_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[j] for j in k_nearest_idx])
            # Get y labels of k nearest points
            y_pred[i] = self._get_aggregated_output(k_nearest_neighbors)
        return y_pred


class KNNClassifier(KNN):
    """ KNN classifier
    (Defines the majority vote function)
    """
    def _majority_vote(self, neighbor_labels):
        counts = np.bincount(neighbor_labels.astype("int"))
        return counts.argmax()

    def predict(self, x_test, x_train, y_train):
        self._get_aggregated_output = self._majority_vote
        return super(KNNClassifier, self).predict(x_test, x_train, y_train)


class KNNRegressor(KNN):
    """KNN Regressor
    Defines the mean of nearest neighbors function
    """
    def _mean(self, neighbor_values):
        return np.mean(neighbor_labels, axis=0)

    def predict(self, x_test, x_train, y_train):
        self._get_aggregated_output = self._mean
        return super(KNNRegressor, self).predict(x_test, x_train, y_train)
