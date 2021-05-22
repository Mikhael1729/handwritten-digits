import numpy as np
import random


def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
  return sigmoid(z) * (1 - sigmoid(z))


class Network:
  def __init__(self, dims):
    self.dimensions = dims
    self.layers_size = len(dims)
    self.biases = [np.random.randn(nl, 1) for nl in dims[1:]]
    self.weights = [np.random.randn(nl_b, nl)
                    for nl, nl_b in zip(dims[:-1], dims[1:])]

  def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)

    return a

  def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """
    training_data: (x, y) samples
    epochs: quantity of iterations
    """
    if test_data:
      n_test = len(test_data)
      n = len(training_data)

    for iteration in range(epochs):
      random.suffle(training_data)
      mini_batches = [training_data[m: m + mini_batch_size]
                      for m in range(0, n, mini_batch_size)]

      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)

    if test_data:
      print(f"Epoch {iteration}: {self.evaluate(test_data)} / {n_test}")
    else:
      print("Epoch {iteration} complete ")

  def update_mini_batch(self, mini_batch, eta):
    """
    mini_batch a list of tuples (x, y)
    eta learning rate
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    # Applying backpropagation.
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    self.weights = [w - (eta / len(mini_batch)) * nw for w,
                    nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta / len(mini_batch)) * nb for b,
                   nb in zip(self.biases, nabla_b)]

  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    activation = x
    activations = [x]
    zs = []

    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)

    delta = self.cost_derivative(
        activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

    return (nabla_b, nabla_w)

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedofrward(x)), y)
                    for (x, y) in test_data]

    return sum(int(x == y) for (x, y) in test_results)

  def cost_derivative(self, output_activations, y):
    return (output_activations - y)
