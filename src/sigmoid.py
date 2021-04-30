from numpy import exp


def sigmoid(z):
  """
  The sigmoid function
  """
  return 1.0/(1.0+exp(-z))
