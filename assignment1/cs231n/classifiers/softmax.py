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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  F = X.dot(W)
  for i, (f, y) in enumerate(zip(F, y)):
    f -= np.max(f)
    loss += -f[y] + np.log(np.sum(np.exp(f)))
    probs = np.ones(dW.shape) * (np.exp(f) / np.sum(np.exp(f)))
    probs[:, y] -= 1
    dW += (X[i].reshape(X[i].shape[0], 1)) * np.ones(dW.shape) * probs

  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  F = X.dot(W)
  F -= np.max(F, axis = 1).reshape(F.shape[0], -1)
  denom = np.sum(np.exp(F), axis=1)  # D, 1
  loss = np.sum(-np.log( np.exp(F[np.arange(F.shape[0]), y]) / denom))

  prob = np.exp(F) / denom.reshape(denom.shape[0], -1)  # D, C
  prob[np.arange(0, prob.shape[0]), y] -= 1
  dW = X.T.dot(prob)

  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

