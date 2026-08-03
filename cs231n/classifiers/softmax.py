from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # GRADIENT
        _tc = np.zeros(num_classes)
        _tc[y[i]] = 1

        dW += X[i].reshape([X[i].shape[0], 1]).dot((p - _tc).reshape([num_classes, 1]).T)


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    
    dW /= num_train

    # REGULARIZED GRAD
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    score = X.dot(W)

    score -= np.max(score, axis=1).reshape([score.shape[0], 1])
    p = np.exp(score)
    p /= np.sum(p, axis=1).reshape([p.shape[0], 1])

    logp = -np.log(p)


    # get class score for true class of each image
    _tc = logp[range(y.shape[0]), y]

    # sum and return

    loss = np.sum(_tc)

    loss /= X.shape[0] # get mean loss

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    _is_tc = np.zeros([y.shape[0], W.shape[1]]) # (N, C)
    _is_tc[range(y.shape[0]), y] = 1
    dW = X.T.dot(p - _is_tc)

    dW /= X.shape[0]
    dW += reg * 2 * W

    return loss, dW
