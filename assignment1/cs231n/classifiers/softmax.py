import numpy as np
from random import shuffle

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
  for i in range(X.shape[0]):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    denom = 0
    for s in scores:
        denom += np.exp(s)
    for j in range(W.shape[1]):
        dW[:, j] += 1.0 / denom * np.exp(scores[j]) * X[i]
        if j == y[i]:
            dW[:, j] -= X[i]
    loss += -np.log(np.exp(correct_class_score) / denom)
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW /= X.shape[0]
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
  scores = X.dot(W)
  scores -= np.max(scores)
    
  correct = scores[np.arange(scores.shape[0]),y]
  
  L = np.log(np.sum(np.exp(scores), axis=1))
  L -= correct
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis=1) # sum over columns
  sum_exp_scores = 1.0 / (sum_exp_scores)

  dW = exp_scores.T * sum_exp_scores
  dW = np.dot(X.T,dW.T)
  correct_mat = np.zeros(scores.shape)
  correct_mat[np.arange(scores.shape[0]),y] = 1
  dW -= np.dot(X.T,correct_mat)

  # average and regularize
  dW /= X.shape[0]
  dW += reg * W
  loss = np.sum(L) / X.shape[0]
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

