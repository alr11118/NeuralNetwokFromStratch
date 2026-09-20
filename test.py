import MINST_loader
import json
import train

def loadWeights(filePath):
    with open(filePath, "r") as file:
        data = json.load(file)
    
    hiddenWeights = data["hiddenWeights"]
    hiddenBiases = data["hiddenBiases"]
    outputWeights = data["outputWeights"]
    outputBiases = data["outputBiases"]
    
    return hiddenWeights, hiddenBiases, outputWeights, outputBiases

def test(inputs, outputs,
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

    return (f"Accuracy: {correct}/{len(inputs)} ({correct / len(inputs) * 100:.2f}%)")

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
    print(test(x_test, y_test, hiddenWeights, hiddenBiases, outputWeights, outputBiases))

if(__name__ == "__main__"):
    main()