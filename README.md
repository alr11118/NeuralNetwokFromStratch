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
## Installation

Install digitReader directly from PyPI: https://pypi.org/project/digitReader/0.1.0/#description 

```bash
python3 -m pip install digitReader
```
To Run:
```bash
classifyDigit --image image.jpeg
```
Supported image files: .jpg, .jpeg, .png

## Why I Built This:

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

## How the Neural Network Works
1. **Input**
<p>Each handwritten digit is a 28×28 image.<br>
The image is flattened containing 784 pixel values:<br>
[0.0, 0.0, 0.12, 0.54, ..., 0.0]<br>

The original pixel values range from 0 to 255, so I normalize them:<br>
normalized_pixel = pixel / 255<br>
This gives values between 0 and 1</p>

2. **Hidden Layer**
<p>Each hidden neuron calculates a weighted sum of the inputs:<br>
z = x₁w₁ + x₂w₂ + ... + x₇₈₄w₇₈₄ + b<br>

The result is passed through the ReLU activation function:<br>
ReLU(z) = max(0, z)<br>

This allows the network to learn nonlinear patterns.</p>

3. **Output Layer**
<p>The hidden-layer outputs are connected to 10 output neurons.<br>
Each output neuron corresponds to one digit:<br>
0 1 2 3 4 5 6 7 8 9<br>

The output neurons first produce raw scores.<br>
These scores are passed through the softmax function to convert them into probabilities.</p>

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

4. **Training**
<p>The network learns by comparing its prediction with the correct label.<br>
I use cross-entropy loss:<br>
Loss = -log(probability of the correct class)<br>

If the network assigns a high probability to the correct digit, the loss is small.<br>
If it assigns a low probability to the correct digit, the loss is large.</p>

5. **Backpropagation**
<p>After calculating the loss, the network calculates gradients for its weights and biases.<br>
The gradients tell the network how each parameter contributed to the error.<br>
The parameters are then updated using gradient descent:<br>
parameter = parameter - learning_rate × gradient<br>

I implemented these gradient calculations manually rather than using an automatic differentiation library.</p>

6. **Mini-Batch Training**
<p>Instead of calculating an update using the entire dataset at once, the network uses mini-batches.<br>

For example:<br>

Training data
     ↓
[batch 1]
[batch 2]
[batch 3]
...
     ↓
Update weights after each batch<br>

The current experiments use a batch size of 10.<br>

The training data is shuffled between iterations so that the network does not always see the examples in the same order.</p>

7. **MNIST Data**
<p>The MNIST dataset is stored in IDX binary files.<br>
Rather than using a library to load the dataset, I wrote my own loader for the binary format.<br>
The image file contains:<br>
- Magic number
- Number of images
- Number of rows
- Number of columns
- Pixel data

The label file contains:<br>
- Magic number
- Number of labels
- Label data
The loader converts these binary files into Python lists that can be passed directly to the neural network.</p>

## Current Results
Final loss: 0.007298405493316789\
Training time: 344.3988567920169\
Accuracy 1.0\
Training Size: 5000\
Batch Size: 10\
Learning Rate: 0.1\
Iterations: 15\

## What I Learned
- Multiple neurons combine to form a layer
- Softmax converts output scores into probabilities
- Cross-entropy measures classification error
- Backpropagation applies the chain rule to calculate gradients
- Gradient descent updates model parameters
- Mini-batches affect training
- Learning rate affects how quickly the model learns
- Model weights can be saved and reused for inference

## Limitations

This project is intentionally implemented using basic Python data structures and loops instead of optimized numerical libraries.

Because of this, training is significantly slower than implementations using NumPy, PyTorch, or TensorFlow.

The current experiments also use a relatively small subset of MNIST compared with the full dataset.

These limitations are intentional because the main goal of the project is to understand the underlying mechanics of neural networks.

## Future Improvements

Possible improvements include:
+ Train on more of the MNIST dataset
+ Experiment with different hidden-layer sizes
+ Add multiple hidden layers
+ Improve training speed
+ Add a graphical interface for drawing digits
+ Compare the implementation against a NumPy-based version

## License
MIT License