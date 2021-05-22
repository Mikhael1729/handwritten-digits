import mnist_loader
from network import Network

print("ANN for recognizing hand written digits")

"""
- 60,000 training images (original)
  - 50,000: For training
  - 10,000: For image validation set (adjust hyperparameters)
- 10,000 test images (original)
"""

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
network = Network([784, 30, 10])
network.sgd(
    training_data=training_data,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)


