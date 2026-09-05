import sys

def forwardPass(input, hiddenWeights, hiddenBiases, outputWeights, outputBias):
    hiddenZ = []
    hiddenOutputs = []
    for i in range(len(hiddenWeights)):
        z = hiddenWeights[i] * input + hiddenBiases[i]
        hiddenZ.append(z)
        hiddenOutputs.append(max(0, z))

    prediction = 0
    for i in range(len(hiddenOutputs)):
        prediction += hiddenOutputs[i] * outputWeights[i]
    prediction += outputBias
    return prediction, hiddenOutputs, hiddenZ

#x = 2
#y = 12
x = [-2, -1, 0, 1, 2]
y = [-32, -20, -8, 4, 16]
lr = 0.01 # learning rate

loss = sys.maxsize
iterations = 0

# Hidden Neurons
hiddenWeights = [2, -1, 3]
hiddenBiases = [0, 1, -2]

#OutputNeuron 
outputWeights = [2, -3, 0.5]
outputBias = 1

while loss > 0.0000001 and iterations < 10000:
    # Set variables
    iterations += 1
    loss = 0
    d_outputWeights = [0] * len(outputWeights)
    d_outputBias = 0
    d_hiddenWeights = [0] * len(hiddenWeights)
    d_hiddenBiases = [0] * len(hiddenBiases)
    for x_index in range(len(x)):
        # Calculate prediction and error
        prediction, hiddenOutputs, hiddenZ = forwardPass(x[x_index], hiddenWeights, hiddenBiases, outputWeights, outputBias)
        error = prediction - y[x_index]
        loss += error ** 2

        # Adjust weights 
        # Note: d means how much the loss changes with respoect to the thing
        d_prediction = 2 * error
        #d_outputWeights = [] #dwo = d_prediction * ah
        for i in range(len(hiddenOutputs)):
            d_outputWeights[i] += d_prediction * hiddenOutputs[i]
        d_outputBias += d_prediction

        #d_hiddenWeights = []
        #d_hiddenBiases = []
        for i in range(len(outputWeights)):
            dah = d_prediction * outputWeights[i] # dah = d_prediction * wo
            d_hiddenOutput = dah * (1 if hiddenZ[i] > 0 else 0) # dzh = dah * (1 if zh > 0 else 0)
            d_hiddenWeight = d_hiddenOutput * x[x_index] #  dwh = dzh * x
            d_hiddenWeights[i] += d_hiddenWeight
            d_hiddenBiases[i] += d_hiddenOutput # dbh = dzh

    # Average
    loss /= len(x)

    for i in range(len(outputWeights)):
        d_outputWeights[i] /= len(x)

    d_outputBias /= len(x)

    for i in range(len(hiddenWeights)):
        d_hiddenWeights[i] /= len(x)
        d_hiddenBiases[i] /= len(x)

    # Update
    for i in range(len(outputWeights)): # wo -= lr * dwo
        outputWeights[i] -= lr * d_outputWeights[i]
    outputBias -= lr * d_outputBias
    for i in range(len(hiddenWeights)): # wh -= lr * dwh
        hiddenWeights[i] -= lr * d_hiddenWeights[i]
    for i in range(len(hiddenBiases)): # bh -= lr * dbh
        hiddenBiases[i] -= lr * d_hiddenBiases[i]

    if iterations % 2 == 0:
        print(
            "hiddenWeights:", hiddenWeights,"\n", 
            "hiddenBiases:", hiddenBiases,"\n",
            "outputWeights:", outputWeights,"\n",
            "outputBias:", outputBias,"\n",
            "loss:", loss
        )
# Print final values
for x_value in x:
    prediction, _, _ = forwardPass(
        x_value,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBias
    )
    print(x_value, "->", round(prediction, 3))
print("Iterations done:", iterations)