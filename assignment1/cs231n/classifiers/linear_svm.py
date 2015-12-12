import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[y[i],:] -= X[:,i].T
        dW[j,:] += X[:,i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[0]
  num_train = X.shape[1]

  # need to use an advanced index so we get copies of the data rather than a view (as in slicing)
  examples_idx = np.arange(num_train)



  scores = W.dot(X)
  correct_scores = scores[y, examples_idx]
  margin = scores - correct_scores + 1
  margin[y, examples_idx] = 0 # set the loss for correct classes to 0

  loss_matrix = np.zeros((num_classes, num_train))
  loss_matrix = np.maximum(loss_matrix, margin) # element-wise maximum
  loss = np.sum(loss_matrix) / num_train


  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = np.zeros(margin.shape)
  mask[margin > 0] = 1

  # sum over all j != y_i
  incorrect_classes = np.sum(mask, axis=0)
  mask[y, examples_idx] = -1 * incorrect_classes[examples_idx]
  dW = np.dot(mask, X.T) / num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
