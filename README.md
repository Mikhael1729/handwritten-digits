# Handwritten Digits

Trying to implement an AI that recognizes handwritten digits.

## Datasets

The specifications of the training and test datasets:

| Type          | Quantity      |
|---------------|---------------|
| Tranining set | 60,000 images |
| Test set.     | 10,000 images |

## Concepts

- *Gradient descendent* (equation 1.6) algorithm. Allows to find the weights and biases which make the cost as small as possible.

## Process overview

The goal in training the neural network is to find weights and biases which minimize the quadratic cost function $C(w, b)$. In other words, this is a minimization problem and a way to solve it is by using a gradient descendent.

0. Use a smooth cost function.
1. Focus on minimizing the quadratic cost. Apparently, the gradient descendant allows you to see small changes to act. This is useful to then follow the next step.
2. Examine the classification accuaracy.
3. Make some modifications to the cost function. It will allow you to get a totally different set of minimizing weights and biases.

## Network overview

- The *input* value $X$ is a vector of $28 \times 28$. Each entry in the vector represents the grey value for a single pixel in the image.
- The *output* is denoted by $y = y(x)$ where y is a 10-dimensional vector.

## References

- [The MNIST Database](http://yann.lecun.com/exdb/mnist/)

