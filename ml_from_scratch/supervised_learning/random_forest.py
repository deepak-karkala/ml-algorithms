import numpy as np
import math
import sys
import matplotlib.pyplot as plt

class RandomForest():
    """Random Forest classifier
    Parameters
    ----------
    n_estimators: int
        Number of trees
    max_features: int
        Max number of features to be used while splitting
    min_samples_split: int
        Min number of samples needed before splitting
    max_depth: int
        Max depth of tree
    min_gain: float
        Min info gain in order to make a split
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        # Init trees for the forest
        self.trees = []
        for i in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split = self.min_samples_split,
                    min_impurity = self.min_gain,
                    max_depth = self.max_depth
                )
            )
    
    def fit(self, x, y):
        """Fit trees of the forest using random sampling of 
        both dataset (bagging) and random subset of features
        """
        n_features = np.shape(x)[1]

        # Set max_features = sqrt(n_features) if not set
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Get random samples of data with replacement (bagging - bootstrap aggregating)
        samples = get_random_subsets(x, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            xbag, ybag = samples[i]

            # Get random subset of max_features number of features
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            self.trees[i].feature_indices = idx
            xbag_subset = xbag[:, idx]

            # Fit tree
            self.trees[i].fit(xbag_subset, ybag)
    
    def predict(self, x):
        """Predict classifier output using majority voting
        """
        y_preds = np.empty((x.shape[0], self.n_estimators))

        # Get predictions from each tree in the forest
        for i, tree in enumerate(self.trees):
            x_subset = x[:, tree.feature_indices]
            y_preds[:, i] = tree.predict(x_subset)
        
        y_pred = []
        # Majority voting
        for pred in y_preds:
            y_pred.append(np.bincount(pred.astype('int')).argmax())
        return np.asarray(y_pred)
