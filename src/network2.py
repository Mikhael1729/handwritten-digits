import numpy as np


class NeuralNetwork:
  def __init__(self):
    # Seeding for random number generation.
    np.random.seed(1)

    # 3 x 1 matrix of weights with values from -1 to 1 and mean of 0.
    self.weights = 2 * np.random.random((3, 1)) - 1

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def train(self, training_inputs, training_outputs, iterations_number):
    for iteration in range(iterations_number):
      output = self.fire(training_inputs)
      error = training_outputs - output
      adjustments = np.dot(training_inputs.T, error *
                           self.sigmoid_derivative(output))

      if iteration == 0:
        print("OUTPUT: ", output)
        print("ERROR: ", error)
        print("ADJUSTMENTS: ", adjustments)

      self.weights += adjustments

      if iteration == 0:
        print("SELF.WEIGHTS: ", self.weights)

  def fire(self, inputs):
    inputs = inputs.astype(float)
    output = self.sigmoid(np.dot(inputs, self.weights))

    return output

if __name__ == "__main__":
  # Compute training input and output data.
  training_inputs = np.array([[0, 0, 1],
                              [1, 1, 1],
                              [1, 0, 1],
                              [0, 1, 1]])
  training_outputs = np.array([[0, 1, 1, 0]]).T

  # Create and train the network.
  network = NeuralNetwork()
  network.train(training_inputs, training_outputs, 15000)

  # Test the network
  test_data = [1, 0, 0]
  result = network.fire(np.array(test_data))

  print(result)
