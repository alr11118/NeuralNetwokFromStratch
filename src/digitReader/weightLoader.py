import json
def loadWeights(filePath):
    with open(filePath, "r") as file:
        data = json.load(file)
    
    hiddenWeights = data["hiddenWeights"]
    hiddenBiases = data["hiddenBiases"]
    outputWeights = data["outputWeights"]
    outputBiases = data["outputBiases"]
    
    return hiddenWeights, hiddenBiases, outputWeights, outputBiases