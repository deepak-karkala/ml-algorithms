import numpy as np

class Loss(object):
    def __init__(self): pass

    def loss(self, y, y_pred):
        raise NotImplemented()
    
    def gradient(self, y, y_pred):
        raise NotImplemented()
    
    def acc(self, y, y_pred):
        return 0

class CrossEntropyLoss(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y/p + (1 - y)/(1 - p)
    
    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))
