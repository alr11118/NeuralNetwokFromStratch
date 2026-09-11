# ACTIVATION FUNCTIONS
def relu(x):
    return max(0, x)

# FORWARD PASS
def forwardPass(input, hiddenWeights, hiddenBiases, outputWeights, outputBias):
    hiddenZ = []
    hiddenOutputs = []
    for i in range(len(hiddenWeights)):
        z = 0
        for j in range(len(input)):
            z += hiddenWeights[i][j] * input[j]
        z += hiddenBiases[i]
        hiddenZ.append(z)
        hiddenOutputs.append(relu(z))

    prediction = 0
    for i in range(len(hiddenOutputs)):
        prediction += hiddenOutputs[i] * outputWeights[i]
    prediction += outputBias
    return prediction, hiddenOutputs, hiddenZ

# LOSS
def lossFunction(error):
    return error ** 2

# BACKWARD PASS
def backwardPass(input, target, prediction,
                 hiddenOutputs, hiddenZ,
                 hiddenWeights, hiddenBiases,
                 outputWeights, outputBias):
    # Gradients start at zero
    d_outputWeights = [0] * len(outputWeights)
    d_outputBias = 0

    d_hiddenWeights = [
        [0] * len(input)
        for _ in range(len(hiddenWeights))
    ]
    d_hiddenBiases = [0] * len(hiddenBiases)

    # Calculate prediction error
    error = prediction - target
    loss = lossFunction(error)
    # Note: d means how much the loss changes with respect to the next thing
    
    # Calculate how the loss changes with prediction
    d_prediction = 2 * error

    # OUPUT LAYER
    # Calculate output-layer gradients
    for i in range(len(hiddenOutputs)):
        d_outputWeights[i] += d_prediction * hiddenOutputs[i]
    d_outputBias += d_prediction

    # HIDDEN LAYER
    # Calculate hidden-layer gradients
    for i in range(len(hiddenWeights)):
        dah = d_prediction * outputWeights[i]
        d_hiddenOutput = dah * (1 if hiddenZ[i] > 0 else 0)
        for j in range(len(input)):
            d_hiddenWeights[i][j] += d_hiddenOutput * input[j]
        d_hiddenBiases[i] += d_hiddenOutput

    return (
        loss,
        d_outputWeights,
        d_outputBias,
        d_hiddenWeights,
        d_hiddenBiases
    )

# TRAINING
def train(x, y,
          hiddenWeights, hiddenBiases,
          outputWeights, outputBias,
          lr, maxIterations):
    loss = float("inf")
    iterations = 0
    while loss > 0.0000001 and iterations < maxIterations:
        iterations += 1

        # Reseet Gradients
        d_outputWeights = [0] * len(outputWeights)
        d_outputBias = 0
        d_hiddenWeights = [
            [0] * len(x[0])
            for _ in range(len(hiddenWeights))
        ]
        d_hiddenBiases = [0] * len(hiddenBiases)
        loss = 0

        # Process each training example
        for x_index in range(len(x)):

            # Forward pass
            prediction, hiddenOutputs, hiddenZ = forwardPass(
                x[x_index],
                hiddenWeights,
                hiddenBiases,
                outputWeights,
                outputBias
            )

            # Backward pass
            (
                exampleLoss,
                example_d_outputWeights,
                example_d_outputBias,
                example_d_hiddenWeights,
                example_d_hiddenBiases
            ) = backwardPass(
                x[x_index],
                y[x_index],
                prediction,
                hiddenOutputs,
                hiddenZ,
                hiddenWeights,
                hiddenBiases,
                outputWeights,
                outputBias
            )

            loss += exampleLoss

            # Update Gradients

            for i in range(len(outputWeights)):
                d_outputWeights[i] += example_d_outputWeights[i]

            d_outputBias += example_d_outputBias

            for i in range(len(hiddenWeights)):
                for j in range(len(x[x_index])):
                    d_hiddenWeights[i][j] += example_d_hiddenWeights[i][j]

            for i in range(len(hiddenBiases)):
                d_hiddenBiases[i] += example_d_hiddenBiases[i]
        # Average loss and gradient s
        loss /= len(x)

        for i in range(len(outputWeights)):
            d_outputWeights[i] /= len(x)

        d_outputBias /= len(x)

        for i in range(len(hiddenWeights)):
            for j in range(len(hiddenWeights[i])):
                d_hiddenWeights[i][j] /= len(x)

        for i in range(len(hiddenBiases)):
            d_hiddenBiases[i] /= len(x)

        # Update Weights and Biases
        for i in range(len(outputWeights)):
            outputWeights[i] -= lr * d_outputWeights[i]

        outputBias -= lr * d_outputBias

        for i in range(len(hiddenWeights)):
            for j in range(len(hiddenWeights[i])):
                hiddenWeights[i][j] -= lr * d_hiddenWeights[i][j]

        for i in range(len(hiddenBiases)):
            hiddenBiases[i] -= lr * d_hiddenBiases[i]

    return (
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBias,
        loss,
        iterations
    )

def predict(input, 
            hiddenWeights,
            hiddenBiases,
            outputWeights,
            outputBias):
    prediction, _, _ = forwardPass(
        input,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBias
    )
    return prediction

# MAIN / DATA
hiddenWeights = [
    [0.1, -0.1],
    [0.1,  0.1],
    [-0.1, 0.1]
]
hiddenBiases = [0.1, 0.1, 0.1]
outputWeights = [0.1, 0.1, 0.1]
outputBias = 0

x = [
    [2, 5],
    [3, 1],
    [4, 7],
    [1, 2]
]
y = [
    12,
    7,
    18,
    5
]

# TRAIN
hiddenWeights, hiddenBiases, outputWeights, outputBias, loss, iterations = train(
    x, y,
    hiddenWeights, hiddenBiases,
    outputWeights, outputBias,
    lr=0.01,
    maxIterations=10000
)

print("Final loss:", loss)
print("Iterations:", iterations)

print("\nHidden weights:", hiddenWeights)
print("Hidden biases:", hiddenBiases)
print("Output weights:", outputWeights)
print("Output bias:", outputBias)

print("\nHidden activations:")

for inputs in x:
    prediction, hiddenOutputs, hiddenZ = forwardPass(
        inputs,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBias
    )

    print(
        "Input:", inputs,
        "Z:", hiddenZ,
        "Hidden:", hiddenOutputs,
        "Prediction:", prediction
    )

# TEST
for i in range(len(x)):
    prediction = predict(
        x[i],
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBias
    )

    print("Input:", x[i],
          "Target:", y[i],
          "Prediction:", prediction)

print("for [2, 3]: " + str(predict(
        [2, 3],
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBias
    )))
