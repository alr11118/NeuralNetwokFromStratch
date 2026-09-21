import math
import random
import MINST_loader
import src.test as test
import json
import time

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
          lr, maxIterations,
          batchSize):

    loss = float("inf")
    iterations = 0

    while loss > 0.0000001 and iterations < maxIterations:
        iterations += 1

        # Shuffle training data each iteration/epoch
        indices = list(range(len(x)))
        random.shuffle(indices)

        totalLoss = 0

        # Process mini-batches
        for batchStart in range(0, len(x), batchSize):

            batchIndices = indices[
                batchStart:batchStart + batchSize
            ]

            # Reset gradients for this mini-batch
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

            # Process examples in this mini-batch
            for x_index in batchIndices:

                # Forward pass
                probabilities, hiddenOutputs, hiddenZ = forwardPass(
                    x[x_index],
                    hiddenWeights,
                    hiddenBiases,
                    outputWeights,
                    outputBiases
                )

                # Softmax + cross-entropy derivative
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

                batchLoss += exampleLoss

                # Add example gradients to batch gradients
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

            # Number of examples actually in this batch
            currentBatchSize = len(batchIndices)

            # Average gradients over the mini-batch
            for i in range(len(outputWeights)):
                for j in range(len(outputWeights[i])):
                    d_outputWeights[i][j] /= currentBatchSize

            for i in range(len(outputBiases)):
                d_outputBiases[i] /= currentBatchSize

            for i in range(len(hiddenWeights)):
                for j in range(len(hiddenWeights[i])):
                    d_hiddenWeights[i][j] /= currentBatchSize

            for i in range(len(hiddenBiases)):
                d_hiddenBiases[i] /= currentBatchSize

            # Update weights immediately after each mini-batch
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

            # Track loss
            totalLoss += batchLoss

        # Average loss across the entire dataset
        loss = totalLoss / len(x)

        print("Iteration:", iterations, "Loss:", loss)

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
    random.seed(42)
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

def saveWeights(filePath, hiddenWeights,hiddenBiases,
                outputWeights,outputBiases):
    data = {
        "hiddenWeights": hiddenWeights,
        "hiddenBiases": hiddenBiases,
        "outputWeights": outputWeights,
        "outputBiases": outputBiases
    }

    with open(filePath, "w") as file:
        json.dump(data, file, indent=4)

def main():
    (hiddenWeights, 
    hiddenBiases, 
    outputWeights, 
    outputBiases) = initializeNetwork(784, 64, 10)

    # Training data
    numData = 5000
    x = MINST_loader.loadImages("MINST/train-images.idx3-ubyte", numData)
    y = MINST_loader.loadLabels("MINST/train-labels.idx1-ubyte", numData)

    # TRAIN
    lr = 0.10
    maxIterations = 15
    batchSize = 10
    start_time = time.perf_counter()
    hiddenWeights, hiddenBiases, outputWeights, outputBiases, loss, iterations = train(
        x, y,
        hiddenWeights, hiddenBiases,
        outputWeights, outputBiases,
        lr,
        maxIterations,
        batchSize
        )
    end_time = time.perf_counter()
    total_time = end_time - start_time
    accuracy = test.test(5000, hiddenWeights, hiddenBiases, outputWeights, outputBiases, False)

    print("\n####################")
    print("RESAULTS")
    print("Final loss:", loss)
    print("Training time:", total_time)
    print("Accuracy", accuracy)

    print("PARAMETERS")
    print("Training Size:", numData)
    print("Batch Size:", batchSize)
    print("Learning Rate:", lr)
    print("Iterations:", iterations)
    print("####################")

    saveWeights("model/weights.json", hiddenWeights, hiddenBiases, outputWeights, outputBiases)
    print("Weights saved")

if(__name__ == "__main__"):
    main()
