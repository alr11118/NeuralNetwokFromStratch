import math
import random

# ACTIVATION FUNCTIONS
def relu(x):
    return max(0, x)

def softmax(outputs):
    total = 0
    probabilities = []

    for output in outputs:
        total += math.exp(output)

    for output in outputs:
        probabilities.append(math.exp(output) / total)

    return probabilities

# FORWARD PASS
def forwardPass(input, hiddenWeights, hiddenBiases, outputWeights, outputBiases):
    hiddenZ = []
    hiddenOutputs = []
    for i in range(len(hiddenWeights)):
        z = 0
        for j in range(len(input)):
            z += hiddenWeights[i][j] * input[j]
        z += hiddenBiases[i]
        hiddenZ.append(z)
        hiddenOutputs.append(relu(z))

    rawOutputs = []
    for i in range(len(outputWeights)):
        output = 0
        for j in range(len(hiddenOutputs)):
            output += hiddenOutputs[j] * outputWeights[i][j]
        output += outputBiases[i]
        rawOutputs.append(output)

    probabilities = softmax(rawOutputs)

    return probabilities, hiddenOutputs, hiddenZ

# LOSS
def lossFunction(probabilities, target):
    probabilityOfCorrectClass = probabilities[target]
    return -math.log(probabilityOfCorrectClass)

# BACKWARD PASS
def backwardPass(input, target, probabilities,
                 hiddenOutputs, hiddenZ,
                 hiddenWeights, hiddenBiases,
                 outputWeights, outputBiases, 
                 d_rawOutputs):
    # Set Gradients to 0
    # Note: d means how much the loss changes with respect to the next thing
    d_outputWeights = [
        [0] * len(hiddenOutputs)
        for _ in range(len(outputWeights))
    ]
    d_outputBiases = [0] * len(outputBiases)
    d_hiddenWeights = [
        [0] * len(input)
        for _ in range(len(hiddenWeights))
    ]
    d_hiddenBiases = [0] * len(hiddenBiases)

    loss = lossFunction(probabilities, target)    

    # OUPUT LAYER
    # Calculate output-layer gradients
    for i in range(len(d_rawOutputs)):
        for j in range(len(hiddenOutputs)):
            d_outputWeights[i][j] += d_rawOutputs[i] * hiddenOutputs[j]

    for i in range(len(d_rawOutputs)):
        d_outputBiases[i] += d_rawOutputs[i]

    # HIDDEN LAYER
    # Calculate hidden-layer gradients
    for i in range(len(hiddenWeights)):
        dah = 0
        for p in range(len(d_rawOutputs)):
            dah += d_rawOutputs[p] * outputWeights[p][i]
        d_hiddenOutput = dah * (1 if hiddenZ[i] > 0 else 0) # Check if relu was activated
        for j in range(len(input)):
            d_hiddenWeights[i][j] += d_hiddenOutput * input[j]
        d_hiddenBiases[i] += d_hiddenOutput

    return (
        loss,
        d_outputWeights,
        d_outputBiases,
        d_hiddenWeights,
        d_hiddenBiases
    )

# TRAINING
def train(x, y,
          hiddenWeights, hiddenBiases,
          outputWeights, outputBiases,
          lr, maxIterations):
    loss = float("inf")
    iterations = 0
    while loss > 0.0000001 and iterations < maxIterations:
        iterations += 1

        # Reseet Gradients
        d_outputWeights = [
            [0] * len(hiddenWeights)
            for _ in range(len(outputWeights))
        ]
        d_outputBiases = [0] * len(outputBiases)
        d_hiddenWeights = [
            [0] * len(x[0])
            for _ in range(len(hiddenWeights))
        ]
        d_hiddenBiases = [0] * len(hiddenBiases)
        loss = 0

        # Process each training example
        for x_index in range(len(x)):

            # Forward pass
            probabilities, hiddenOutputs, hiddenZ = forwardPass(
                x[x_index],
                hiddenWeights,
                hiddenBiases,
                outputWeights,
                outputBiases
            )
            d_rawOutputs = [0] * len(probabilities)

            for i in range(len(probabilities)):
                d_rawOutputs[i] = probabilities[i] - (1 if i == y[x_index] else 0)

            # Backward pass
            (
                exampleLoss,
                example_d_outputWeights,
                example_d_outputBiases,
                example_d_hiddenWeights,
                example_d_hiddenBiases
            ) = backwardPass(
                x[x_index],
                y[x_index],
                probabilities,
                hiddenOutputs,
                hiddenZ,
                hiddenWeights,
                hiddenBiases,
                outputWeights,
                outputBiases, 
                d_rawOutputs
            )

            # Update Gradients and Loss
            loss += exampleLoss

            # Update Gradients
            for i in range(len(outputWeights)):
                for j in range(len(outputWeights[i])):
                    d_outputWeights[i][j] += example_d_outputWeights[i][j]

            for i in range(len(outputBiases)):
                d_outputBiases[i] += example_d_outputBiases[i]

            for i in range(len(hiddenWeights)):
                for j in range(len(x[x_index])):
                    d_hiddenWeights[i][j] += example_d_hiddenWeights[i][j]

            for i in range(len(hiddenBiases)):
                d_hiddenBiases[i] += example_d_hiddenBiases[i]

        # Average loss and gradients
        loss /= len(x)

        for i in range(len(outputWeights)):
            for j in range(len(outputWeights[i])):
                d_outputWeights[i][j] /= len(x)

        for i in range(len(outputBiases)):
            d_outputBiases[i] /= len(x)

        for i in range(len(hiddenWeights)):
            for j in range(len(hiddenWeights[i])):
                d_hiddenWeights[i][j] /= len(x)

        for i in range(len(hiddenBiases)):
            d_hiddenBiases[i] /= len(x)

        # Update Weights and Biases
        for i in range(len(outputWeights)):
            for j in range(len(outputWeights[i])):
                outputWeights[i][j] -= lr * d_outputWeights[i][j]

        for i in range(len(outputBiases)):
            outputBiases[i] -= lr * d_outputBiases[i]

        for i in range(len(hiddenWeights)):
            for j in range(len(hiddenWeights[i])):
                hiddenWeights[i][j] -= lr * d_hiddenWeights[i][j]

        for i in range(len(hiddenBiases)):
            hiddenBiases[i] -= lr * d_hiddenBiases[i]

    return (
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases,
        loss,
        iterations
    )

def predict(input, 
            hiddenWeights,
            hiddenBiases,
            outputWeights,
            outputBiases):
    probabilities, _, _ = forwardPass(
        input,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases
    )
    return probabilities

def initializeNetwork(inputSize, hiddenSize, outputSize):
    # Example: initializeNetwork(784, 64, 10)
    # 64 Hidden Neurans with 784 inputs each
    hiddenWeights = [
        [random.uniform(-0.1, 0.1) for _ in range(inputSize)]
        for _ in range(hiddenSize)
    ]
    hiddenBiases = [0] * hiddenSize
    # 10 Output neurons for 10 classses
    outputWeights = [
        [random.uniform(-0.1, 0.1) for _ in range(hiddenSize)]
        for _ in range(outputSize)
    ] 
    outputBiases = [0] * outputSize
    return (hiddenWeights, hiddenBiases, 
            outputWeights, outputBiases)

# MAIN / DATA
(hiddenWeights, 
 hiddenBiases, 
 outputWeights, 
 outputBiases) = initializeNetwork(784, 64, 10)

# Training data
x = [
    [2, 5],   # class 0
    [3, 1],   # class 1
    [4, 7],   # class 2
    [1, 2]    # class 1
]

y = [
    0,
    1,
    2,
    1
]


# TRAIN
hiddenWeights, hiddenBiases, outputWeights, outputBiases, loss, iterations = train(
    x, y,
    hiddenWeights, hiddenBiases,
    outputWeights, outputBiases,
    lr=0.01,
    maxIterations=10000
)

print("Final loss:", loss)
print("Iterations:", iterations)

print("\nHidden weights:", hiddenWeights)
print("Hidden biases:", hiddenBiases)
print("Output weights:", outputWeights)
print("Output biases:", outputBiases)


# TEST TRAINING EXAMPLES
print("\nPredictions:")

for i in range(len(x)):
    probabilities = predict(
        x[i],
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases
    )

    predictedClass = probabilities.index(max(probabilities))

    print(
        "Input:", x[i],
        "Target:", y[i],
        "Probabilities:", probabilities,
        "Predicted class:", predictedClass
    )


# TEST NEW INPUT
probabilities = predict(
    [2, 3],
    hiddenWeights,
    hiddenBiases,
    outputWeights,
    outputBiases
)

predictedClass = probabilities.index(max(probabilities))

print("\nFor [2, 3]:")
print("Probabilities:", probabilities)
print("Predicted class:", predictedClass)
