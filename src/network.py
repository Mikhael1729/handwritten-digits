from sigmoid import sigmoid

import numpy as np


class Network:
  def __init__(self, layers_dimensions):
    """
    Creates the network by initializing each layer and their coorresponding connections.

    `layer_dimensions`: list
      description:
        an array where each position represents a layer and each one holds the quantity of
        neurons that layer has.
      example: [2, 3, 1]
        This generates a network with this shape
            * 
          * * *
          * * 
    """
    self.layers_dimensions = layers_dimensions
    self.layers_quantity = len(layers_dimensions)
    self.biases = [np.random.randn(y, 1) for y in layers_dimensions[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(
        layers_dimensions[:-1], layers_dimensions[1:])]

    print("BIASES")
    print(self.biases)
    print("WEIGHTS")
    print(self.weights)

network = Network([2, 3, 1])
