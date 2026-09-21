import MINST_loader
import json
import src.train as train

def loadWeights(filePath):
    with open(filePath, "r") as file:
        data = json.load(file)
    
    hiddenWeights = data["hiddenWeights"]
    hiddenBiases = data["hiddenBiases"]
    outputWeights = data["outputWeights"]
    outputBiases = data["outputBiases"]
    
    return hiddenWeights, hiddenBiases, outputWeights, outputBiases

def test_withLoadedData(inputs, outputs,
         hiddenWeights, hiddenBiases,
         outputWeights, outputBiases):
    correct = 0
    for i in range(len(inputs)):
        probabilities = train.predict(
            inputs[i],
            hiddenWeights,
            hiddenBiases,
            outputWeights,
            outputBiases
        )
        predictedClass = probabilities.index(max(probabilities))
        if predictedClass == outputs[i]:
            correct += 1

    accuracy = correct/len(inputs)
    return accuracy

def test(testSize, 
         hiddenWeights, hiddenBiases, 
         outputWeights, outputBiases, useTestset = True):
    
    x_test = MINST_loader.loadImages(
        'MINST/t10k-images.idx3-ubyte' if useTestset else 'MINST/train-images.idx3-ubyte',
        testSize
    )
    y_test = MINST_loader.loadLabels(
        'MINST/t10k-labels.idx1-ubyte' if useTestset else 'MINST/train-labels.idx1-ubyte',
        testSize
    )
    correct = 0
    for i in range(len(x_test)):
        probabilities = train.predict(
            x_test[i],
            hiddenWeights,
            hiddenBiases,
            outputWeights,
            outputBiases
        )
        predictedClass = probabilities.index(max(probabilities))
        if predictedClass == y_test[i]:
            correct += 1

    accuracy = correct/len(x_test)
    return accuracy

def main():
    (hiddenWeights, hiddenBiases, outputWeights, outputBiases) = loadWeights("model/weights.json")
    x_test = MINST_loader.loadImages(
        "MINST/t10k-images.idx3-ubyte",
        5000
    )
    y_test = MINST_loader.loadLabels(
        "MINST/t10k-labels.idx1-ubyte",
        5000
    )
    accuracy = test_withLoadedData(x_test, y_test, hiddenWeights, hiddenBiases, outputWeights, outputBiases)
    print(f"Accuracy: {accuracy} ({accuracy * 100:.2f}%)")

if(__name__ == "__main__"):
    main()