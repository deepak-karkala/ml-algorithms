import numpy as np
import math
import sys
import matplotlib.pyplot as plt

class DecisionNode():
    """Class that represents decision node or leaf in the tree
    Parameters
    ----------
    feature_i: int
        Feature index which will be used to split the data into two branches
    threshold: float
        Value used to split into two branches
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Left branch node (will have samples for which feature at i >= threshold)
    false_branch: DecisionNode
        Right branch node (will have samples for which feature at i < threshold)
    """
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    """Class to represent a decision tree
    Super class of RegressionTree() and ClassificationTree()
    Parameters
    ----------
    min_samples_split: int
        Min number of samples needed to make a split
    min_impurity: float
        Min impurity required to make a split
    max_depth: int
        Max depth of tree
    loss: function
        Loss function used to compute impurity for gradient boosted trees
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None # Root node of the tree
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        # Attributes set by child classes
        # Function to calculate values at leaf node
        # (Regression: mean, Classification: majority)
        self._leaf_value_calculation = None
        # Metric used to calculate impurity 
        # (Regression: Variance reduction, classification: Info gain(entropy))
        self._impurity_calculation = None
        # Loss function, used only for gradient boosted trees
        self.loss = None 
        
        # Additional attributes
        # If y is one-hot encoded (multi-dim) or not
        self.one_dim = None
    
    def fit(self, x, y):
        """Determine the tree structure
        """
        self.one_dim = len(np.shape(y))==1
        self.root = self._build_tree(x, y)
    
    def predict_value(self, x, tree=None):
        """Predict the value/class for a single data point (Sample)
        Parameters
        ----------
        tree: Recursive DecisionNode structure
            has attribute value at leaf node
        """
        # Recurse through the tree and find which region each sample in x
        # belongs to and then assign the corresponding leaf value
        # (Regression: mean, Classification: Majority vote)

        if tree is None:
            tree = self.root
        
        # Base case: we have reached leaf level
        if tree.value is not None:
            return tree.value
        
        # Decide which branch to take, keep recursing
        # taking left or right branch based on feature
        # value and threshold
        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            branch = tree.true_branch if feature_value >= tree.threshold else tree.false_branch
        else:
            branch = tree.true_branch if feature_value == tree.threshold else tree.false_branch

        # Recurse through the tree until leaf node is found
        return self.predict_value(x, branch)
    
    def predict(self, x):
        """Compute tree output for datapoints
        """
        return [self.predict_value(sample) for sample in x]

    def _build_tree(self, x, y, current_depth=0):
        """Build the recursive tree structure based on feature, threshold
        resulting in most impurity reduction at each level (greedy)
        and compute values at each leaf node
        """
        n_samples, n_features = np.shape(x)

        largest_impurity = 0
        best_criteria = None
        best_sets = None

        # If y is one-dim, convert to column vector
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        
        # Concatenate x and y
        xy = np.concatenate((x,y), axis=1)

        # Continue building tree at this split only if
        #   number of samples > min_samples_split 
        #   current_depth <= max_depth
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:

            # Iterate through all features to find the feature with best split
            for feature_i in range(n_features):

                feature_values = np.expand_dims(x[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterature through all unique values of a given feature
                for threshold in unique_values:

                    # Calculate impurity (var-red or info-gain for this split)
                    xy1, xy2 = divide_on_feature(xy, feature_i, threshold)

                    # Ensure that both branches has at least one sample
                    if len(xy1)>0 and len(xy2)>0:
                        
                        # Compute impurity for the split
                        y1 = xy1[:, n_features:]
                        y2 = xy2[:, n_features:]

                        impurity = self._impurity_calculation(y, y1, y2)
                        # If current impurity better than best, save this
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i":feature_i, "threshold":threshold};
                            best_sets = {
                                "leftx" : xy1[:, :n_features],
                                "lefty" : xy1[:, n_features:],
                                "rightx" : xy2[:, :n_features],
                                "righty" : xy2[:, n_features:]
                            }
            
        # After iterating through all features, threshold values,
        # and finding the feature,threshold with best split (largest impurity reduction),
        # continue to build tree recursively (left (True) and right (false) branches)
        # if the largest_impurity is more than min_impurity parameter set
        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftx"], best_sets["lefty"], current_depth+1)
            false_branch = self._build_tree(best_sets["rightx"], best_sets["righty"], current_depth+1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                                true_branch=true_branch, false_branch=false_branch)
        
        # If largest_impurity is less than min_impurity needed to split further
        # then we have reached the leaf node, compute leaf value
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)


class RegressionTree(DecisionTree):
    """Class for Regression Tree
    Inherits from Decision tree
    """
    def _variance_reduction(self, y, y1, y2):
        total_variance = calculate_variance(y)
        left_branch_variance = calculate_variance(y1)
        right_branch_variance = calculate_variance(y2)

        ratio = len(y1)/len(y)
        variance_reduction = total_variance - (ratio*left_branch_variance + (1-ratio)*right_branch_variance)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        return np.mean(y, axis=0)

    # functions for impurity calculation and leaf value
    # calculation need to be assigned before invoking the 
    # super's fit function
    def fit(self, x, y):
        self._impurity_calculation = self._variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(x,y)


class ClassificationTree(DecisionTree):
    """Class for classification Tree
    """

    # Define the leaf calculation and impurity reduction
    # functions for classification
    def _info_gain(self, y, y1, y2):
        # Compute entropies of entire data, left and right splits
        total_entropy = calculate_entropy(y)
        left_branch_entropy = calculate_entropy(y1)
        right_branch_entropy = calculate_entropy(y2)

        ratio = len(y1)/len(y)
        info_gain = total_entropy - (ratio*left_branch_entropy + (1-ratio)*right_branch_entropy)
        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        #most_common = np.argmax([len(y[y==label]) for label in np.unique(y)])
        return most_common
        

    def fit(self, x, y):
        self._impurity_calculation = self._info_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(x,y)