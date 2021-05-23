# Handwritten Digits

A simple ANN that recognizes handwritten digits on a photo of $78 \times 78$ pixels.

## Architecture

This is the structure of the ANN:

- Input layer has 784 neurons. Each neuron has a value from 0 to 255 representing the intensity of each pixel of the given image.

- Hidden layers (uses $ReLU$). Has only one with 10 neurons.

- Output layer (uses $softmax$). Has 10 neurons, each one represents a probability distribution from 0 to 1 indicating which number is the most probably true answer for the given image. The output neuron with the greatest value is the number the network thinks matches the given image.

Here is a graphical representation of the network.

<p align="center">
  <img src="./images/representation.jpg" width="100%">
</p>

Here are the equations used for a single iteration of Gradient Descent:

<p align="center">
  <img src="./images/one_iteration_gd.jpg" width="400">
  <img align="top" src="./images/dimensions.jpg" width="300">
</p>

