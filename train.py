import math
import random
import MINST_loader

# ACTIVATION FUNCTIONS
def relu(x):
    return max(0, x)

def softmax(outputs):
    maxOutput = max(outputs)
    expOutputs = []
    for output in outputs:
        expOutputs.append(math.exp(output - maxOutput))
    total = sum(expOutputs)
    probabilities = []
    for expOutput in expOutputs:
        probabilities.append(expOutput / total)
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
    while loss > 0.000001 and iterations < maxIterations:
        iterations += 1
        loss = 0

        # Prepare small bathes
        batchSize = 20
        for batchIndex in range(len(x) // batchSize):
            start = batchIndex * batchSize
            end = start + batchSize

            x_batch = x[start:end]
            y_batch = y[start:end]

            # Reseet Gradients before each batch
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
            batchLoss = 0
            # Process each training example
            for x_index in range(len(x_batch)):

                # Forward pass
                probabilities, hiddenOutputs, hiddenZ = forwardPass(
                    x_batch[x_index],
                    hiddenWeights,
                    hiddenBiases,
                    outputWeights,
                    outputBiases
                )
                d_rawOutputs = [0] * len(probabilities)

                for i in range(len(probabilities)):
                    d_rawOutputs[i] = probabilities[i] - (1 if i == y_batch[x_index] else 0)

                # Backward pass
                (
                    exampleLoss,
                    example_d_outputWeights,
                    example_d_outputBiases,
                    example_d_hiddenWeights,
                    example_d_hiddenBiases
                ) = backwardPass(
                    x_batch[x_index],
                    y_batch[x_index],
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
                batchLoss += exampleLoss

                # Update Gradients
                for i in range(len(outputWeights)):
                    for j in range(len(outputWeights[i])):
                        d_outputWeights[i][j] += example_d_outputWeights[i][j]

                for i in range(len(outputBiases)):
                    d_outputBiases[i] += example_d_outputBiases[i]

                for i in range(len(hiddenWeights)):
                    for j in range(len(x_batch[x_index])):
                        d_hiddenWeights[i][j] += example_d_hiddenWeights[i][j]

                for i in range(len(hiddenBiases)):
                    d_hiddenBiases[i] += example_d_hiddenBiases[i]

            # Average loss and gradients
            batchLoss /= len(x_batch)
            loss += batchLoss

            for i in range(len(outputWeights)):
                for j in range(len(outputWeights[i])):
                    d_outputWeights[i][j] /= len(x_batch)

            for i in range(len(outputBiases)):
                d_outputBiases[i] /= len(x_batch)

            for i in range(len(hiddenWeights)):
                for j in range(len(hiddenWeights[i])):
                    d_hiddenWeights[i][j] /= len(x_batch)

            for i in range(len(hiddenBiases)):
                d_hiddenBiases[i] /= len(x_batch)

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
    loss /= (len(x)/batchSize)

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
    # Using Kaiming Initialization for optimization
    scale = math.sqrt(2 / inputSize)
    hiddenWeights = [
        [random.gauss(0, scale) for _ in range(inputSize)]
        for _ in range(hiddenSize)
    ]
    hiddenBiases = [0] * hiddenSize

    # 10 Output neurons for 10 classses
    scale = math.sqrt(2 / hiddenSize)
    outputWeights = [
        [random.gauss(0, scale) for _ in range(hiddenSize)]
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
numData = 500
x = MINST_loader.loadImages("MINST/train-images.idx3-ubyte", numData)
y = MINST_loader.loadLabels("MINST/train-labels.idx1-ubyte", numData)

# TRAIN
hiddenWeights, hiddenBiases, outputWeights, outputBiases, loss, iterations = train(
    x, y,
    hiddenWeights, hiddenBiases,
    outputWeights, outputBiases,
    lr=0.01,
    maxIterations=20
)

print("\nTraining complete!")
print("Final loss:", loss)
print("Iterations:", iterations)

# TEST
probabilities = predict(
        x[0],
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases
    )
predictedClass = probabilities.index(max(probabilities))

print(
    #"Input:", x[0],
    "Target:", y[0],
    "Probabilities:", probabilities,
    "Predicted class:", predictedClass
)

"""
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
"""

