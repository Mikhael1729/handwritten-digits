import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Load and suffle data.
data = np.array(pd.read_csv('../data/train.csv')) # (m, 784 + 1)

# m, samples, n values on each sample m.
m, n = data.shape

# Shuffle the data.
np.random.shuffle(data)

# Dev data.
data_dev = data[0:1000].T # (784 + 1, m)
Y_dev = data_dev[0]  # True labels.
X_dev = data_dev[1:n] / 255  # Pixels for each label normalized (0 to 1)

# Train data.
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255
_, m_train = X_train.shape


def initialize():
  n0, n1, n2 = (784, 10, 10)

  W1 = np.random.rand(n1, n0) - 0.5
  b1 = np.random.rand(n1, 1) - 0.5

  W2 = np.random.rand(n2, n1) - 0.5
  b2 = np.random.rand(n2, 1) - 0.5

  return W1, b1, W2, b2


def relu(Z):
  return np.maximum(Z, 0)


def relu_derivative(Z):
  return Z > 0


def softmax(Z):
  A = np.exp(Z) / sum(np.exp(Z))
  return A

def forward(w1, b1, w2, b2, X, for_prediction=False):
  Z1 = np.dot(w1, X) + b1
  A1 = relu(Z1)

  Z2 = np.dot(w2, A1) + b2
  A2 = softmax(Z2)

  if for_prediction == False:
    return Z1, A1, Z2, A2
  else:
    return A2


def one_hot(Y):
  # Create a (m, nL) array of 0s to store each one hot value.
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))

  # Place a one in each position indicated in Y (true labels)
  one_hot_Y[np.arange(Y.size), Y] = 1

  # Use a (nL, m) array instead. Each column is the one hot encoded labels
  one_hot_Y = one_hot_Y.T

  return one_hot_Y


def backward(Z1, A1, Z2, A2, W1, W2, X, Y):
  one_hot_Y = one_hot(Y)

  dZ2 = A2 - one_hot_Y
  dW2 = (1 / m) * np.dot(dZ2, A1.T)
  db2 = (1 / m) * np.sum(dZ2)

  dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
  dW1 = (1 / m) * np.dot(dZ1, X.T)
  db1 = (1 / m) * np.sum(dZ1)

  return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1

  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * db2

  return W1, b1, W2, b2


def get_predictions(A2):
  # For each sample, get the index where the greatest value (from 0 to 9)
  # predictions shape is (1, m)
  predictions = np.argmax(A2, axis=0)
  return predictions


def get_accuracy(predictions, Y):
  # Divide how many of the predictions are correct by the total examples.
  accuracy = np.sum(predictions == Y) / Y.size
  return accuracy

def gradient_descent(X, Y, alpha, iterations):
  W1, b1, W2, b2 = initialize()

  for i in range(iterations):
    # Forward propagation.
    Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)

    # Backward propagation.
    dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)

    # Update parameters
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

    # Make updates.
    if i % 10 == 0:
      predictions = get_predictions(A2)
      accuracy = get_accuracy(predictions, Y)
      print(f"Iteration {i}: {accuracy}")
      print(predictions)

  return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
  A2 = forward(W1, b1, W2, b2, X, for_prediction=True)
  predictions = get_predictions(A2)

  return predictions

def test_prediction(index, W1, b1, W2, b2):
  current_image = X_train[:, index, None]
  prediction = predict(X_train[:, index, None], W1, b1, W2, b2)
  label = Y_train[index]
  print("Prediction: ", prediction)
  print("Label: ", label)

  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation='nearest')
  plt.show()

"""
Train the network
"""
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

"""
Make predictions on the test set
"""
for i in range(0, 4):
  test_prediction(i, W1, b1, W2, b2)

"""
Accuracy on the dev set
"""

dev_predictions = predict(X_dev, W1, b1, W2, b2)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)

print(f"Accuracy on the dev set: {dev_accuracy}")
print("Dev predictions:")
print(dev_predictions)
