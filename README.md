# Neural Network From Scratch

A neural network built entirely from scratch in Python to recognize handwritten digits from the MNIST dataset.

The goal of this project was to understand how neural networks actually work by implementing the mathematics and training process myself, without using machine-learning frameworks such as TensorFlow or PyTorch.

## Overview

This project implements a neural network that takes a 28×28 grayscale handwritten digit image as input and predicts which digit it is (0–9).

Each MNIST image contains:

- 28 × 28 pixels
- 784 input values
- Pixel values normalized from 0–255 to 0–1

The network architecture:

```text
784 input pixels
       ↓
64 hidden neurons
       ↓
ReLU activation
       ↓
10 output neurons
       ↓
Softmax
       ↓
Predicted digit (0–90
```

### Why I Built This:

I wanted to understand neural networks from the inside to better optimise/train my future neural networks.

I implemented components such as:
- Forward propagation
- ReLU activation
- Softmax
- Cross-entropy loss
- Backpropagation
- Gradient descent
- Mini-batch training
- Weight initialization
- Model prediction
- Saving trained weights
- Loading MNIST directly from the IDX binary files

The project started with much smaller networks such as linear regression and gradually developed into the MNIST classifier.

### How the Neural Network Works
1. Input

Each handwritten digit is a 28×28 image.

The image is flattened containing 784 pixel values:

[0.0, 0.0, 0.12, 0.54, ..., 0.0]

The original pixel values range from 0 to 255, so I normalize them:

normalized_pixel = pixel / 255

This gives values between 0 and 1.

2. Hidden Layer

Each hidden neuron calculates a weighted sum of the inputs:

z = x₁w₁ + x₂w₂ + ... + x₇₈₄w₇₈₄ + b

The result is passed through the ReLU activation function:

ReLU(z) = max(0, z)

This allows the network to learn nonlinear patterns.

3. Output Layer

The hidden-layer outputs are connected to 10 output neurons.

Each output neuron corresponds to one digit:

0 1 2 3 4 5 6 7 8 9

The output neurons first produce raw scores.

These scores are passed through the softmax function to convert them into probabilities.

For example:
```
0 → 0.001
1 → 0.002
2 → 0.003
3 → 0.004
4 → 0.010
5 → 0.940
6 → 0.005
7 → 0.002
8 → 0.020
9 → 0.013
```
The predicted digit is the class with the highest probability.

#### Training

The network learns by comparing its prediction with the correct label.

I use cross-entropy loss:

Loss = -log(probability of the correct class)

If the network assigns a high probability to the correct digit, the loss is small.

If it assigns a low probability to the correct digit, the loss is large.

#### Backpropagation

After calculating the loss, the network calculates gradients for its weights and biases.

The gradients tell the network how each parameter contributed to the error.

The parameters are then updated using gradient descent:

parameter = parameter - learning_rate × gradient

I implemented these gradient calculations manually rather than using an automatic differentiation library.

#### Mini-Batch Training

Instead of calculating an update using the entire dataset at once, the network uses mini-batches.

For example:

Training data
     ↓
[batch 1]
[batch 2]
[batch 3]
...
     ↓
Update weights after each batch

The current experiments use a batch size of 10.

The training data is shuffled between iterations so that the network does not always see the examples in the same order.

#### MNIST Data

The MNIST dataset is stored in IDX binary files.

Rather than using a library to load the dataset, I wrote my own loader for the binary format.

The image file contains:

Magic number
Number of images
Number of rows
Number of columns
Pixel data

The label file contains:

Magic number
Number of labels
Label data

The loader converts these binary files into Python lists that can be passed directly to the neural network.

### Current Results
Training examples: 1,000
Hidden neurons:    64
Batch size:        10
Learning rate:     0.1
Iterations:        15

Training loss: 0.0157
Accuracy:      91.72%

### What I Learned
- Multiple neurons combine to form a layer
- Softmax converts output scores into probabilities
- Cross-entropy measures classification error
- Backpropagation applies the chain rule to calculate gradients
- Gradient descent updates model parameters
- Mini-batches affect training
- Learning rate affects how quickly the model learns
- Model weights can be saved and reused for inference

### Limitations

This project is intentionally implemented using basic Python data structures and loops instead of optimized numerical libraries.

Because of this, training is significantly slower than implementations using NumPy, PyTorch, or TensorFlow.

The current experiments also use a relatively small subset of MNIST compared with the full dataset.

These limitations are intentional because the main goal of the project is to understand the underlying mechanics of neural networks.

### Future Improvements

Possible improvements include:
+ Train on more of the MNIST dataset
+ Experiment with different hidden-layer sizes
+ Add multiple hidden layers
+ Improve training speed
+ Add a graphical interface for drawing digits
+ Compare the implementation against a NumPy-based version

### Current Resaults

```text
| Training Size | Batch Size | LR | Iterations | Accuracy |
|---------------|------------|----|------------|----------|
| 1,000         | 10         | .1 | 10         | 91.12%   |
| 1,000         | 10         | .1 | 15         | 91.72%   |
| 1,000         | 10         | .1 | 20         | 91.26%   |
| ...           | ...        | ...| ...        | ...      |
```
