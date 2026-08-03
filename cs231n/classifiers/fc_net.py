from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################

        self._rng = np.random.default_rng()
        self.params['W1'] = self._rng.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(shape=(hidden_dim,))
        self.params['W2'] = self._rng.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(shape=(num_classes,))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c. (done)

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # unroll x
        _x = X.reshape((X.shape[0], X[0].size))
        scores = np.dot(np.maximum(np.dot(_x, self.params['W1']) + self.params['b1'], 0), self.params['W2']) + self.params['b2']

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, df = softmax_loss(scores, y)

        # regularize loss (sum of squared components)
        _l2_reg = np.sum(self.params['W2'] ** 2) + np.sum(self.params['W1'] ** 2)
        loss = loss + (0.5 * self.reg * _l2_reg)

        # grad of loss wrt x (score input), W1, W2, b1, b2
        positive_scores = np.zeros_like(scores)
        positive_scores[scores > 0] = 1 # set 1 to positions for positive scores

        # print(_x.T.shape, df.shape, self.params['W2'].shape)
        layer1_score = _x @ self.params['W1'] + self.params['b1']
        l1_positive = np.zeros_like(layer1_score)
        l1_positive[layer1_score > 0] = 1

        grads["W2"] = np.maximum(layer1_score.T, 0) @ df + self.params['W2'] * self.reg
        grads["b2"] = np.sum(df, axis=0)

        grads["W1"] = _x.T @ ((df @ self.params['W2'].T) * l1_positive) + self.params['W1'] * self.reg
        grads['b1'] = np.sum((df @ self.params['W2'].T) * l1_positive, axis=0)

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        # weight 0: input dim * hidden[0]        
        _rng = np.random.default_rng()
        _weights = []
        _biases = []
        _weights.append(_rng.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dims[0])))
        _biases.append(np.zeros((hidden_dims[0])))
        _gammas = []
        _betas = []

        if self.normalization == 'batchnorm':
           _gammas.append(np.ones((hidden_dims[0])))
           _betas.append(np.zeros((hidden_dims[0])))
        
        # weight 1 onward: hidden[i] * hidden[i+1]
        for _ih in range(len(hidden_dims) - 1):
           _weights.append(_rng.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[_ih], hidden_dims[_ih + 1])))
           _biases.append(np.zeros((hidden_dims[1])))

           if self.normalization == 'batchnorm':
               _gammas.append(np.ones((hidden_dims[_ih+1]))) # for input (N, D), scale has size D, init to ones
               _betas.append(np.zeros((hidden_dims[_ih+1]))) # same for shift - but init to 0


        # last weight: hidden[-1] * class_num
        _weights.append(_rng.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[-1], num_classes)))
        _biases.append(np.zeros((num_classes)))

        if self.normalization == 'batchnorm':
           _gammas.append(np.ones((num_classes)))
           _betas.append(np.zeros((num_classes)))

        for _iw, _w in enumerate(_weights, 1):
           self.params[f'W{_iw}'] = _w
           self.params[f'b{_iw}'] = _biases[_iw-1]

           if self.normalization == 'batchnorm':
                self.params[f'gamma{_iw}'] = _gammas[_iw-1]
                self.params[f'beta{_iw}'] = _betas[_iw-1]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        _affine_caches = []
        _relu_caches = []

        _x = X
        for _li in range(self.num_layers - 1):
            f, cache = affine_forward(_x, self.params[f'W{_li+1}'], self.params[f'b{_li+1}'])
            _affine_caches.append(cache)

            a, cache = relu_forward(f)
            _relu_caches.append(cache)
            _x = a

        f, cache = affine_forward(_x, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        _affine_caches.append(cache)
        scores = f

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # store loss and grads
        
        loss, df = softmax_loss(scores, y)

        # calculate sum of L2 regularizations for each weight
        _l2_norm = 0
        for i in range(self.num_layers):
           _l2_norm += np.sum( self.params[f'W{i+1}'] ** 2)

        loss = loss + (0.5 * _l2_norm * self.reg)

        dout = df

        # last layer: affine only
        dout, dw, db = affine_backward(dout, _affine_caches.pop())
        grads[f'W{self.num_layers}'] = dw + self.params[f'W{self.num_layers}'] * self.reg
        grads[f'b{self.num_layers}'] = db

        # backprop from layerX to layer1
        for _li in range(self.num_layers-1, 0, -1):
            # run ReLU for every layer except for last
            dout = relu_backward(dout, _relu_caches.pop())                

            # back to affine layer
            dout, dw, db = affine_backward(dout, _affine_caches.pop())
            
            # d of loss w.r.t w, b 
            # with regularization

            grads[f'W{_li}'] = dw + self.params[f'W{_li}'] * self.reg
            grads[f'b{_li}'] = db


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True