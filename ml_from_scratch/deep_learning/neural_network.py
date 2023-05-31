import numpy as np
import progressbar

def batch_iterator(X, y=None, batch_size=64):
    """
    Yields batch of data
    """
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        start, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[start:end], y[start:end]
        else:
            yield X[start:end]

class NeuralNetwork():
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_function = loss
        self.layers = []
        self.errors = {"training":[], "validation":[]}

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If the network has a layer, use it to set input shape of 
        #   new layer being added
        if self.layers:
            layer.set_input_shape(shape = self.layers[-1].output_shape())
        
        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        
        # Add current layer to network
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        return loss, acc
    
    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        # Forward pass
        y_pred = self._forward_pass(X, training=True)
        # Compute loss function and accuracy on train data
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Gradient wrt input of loss function
        grad_loss = self.loss_function.gradient(y, y_pred)
        # Backward pass (back propagate gradients through entire network)
        self._backward_pass(grad_loss)
        return loss, acc

    def fit(self, X, y, n_epochs, batch_size, X_val=None, y_val=None):
        """ Trains the model for a fixed number of epochs """
        # Run for multiple epochs
        #for _ in progressbar.ProgressBar(range(n_epochs)):
        for _ in range(n_epochs):
            # For each epoch, run over all minibatches
            batch_loss = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_loss.append(loss)
            self.errors["training"].append(np.mean(batch_loss))

            # At the end of each epoch, get loss,acc on train and val data
            if X_val is not None and y_val is not None:
                val_loss, _ = self.test_on_batch(X_val, y_val)
                self.errors["validation"].append(val_loss)
        
        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        return layer_output
    
    def _backward_pass(self, grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
        return grad
    
    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward_pass(X, training=False)
