import json
from importlib.resources import files

def loadWeights():
    weightsPath = files("digitReader").joinpath("weights.json")

    with weightsPath.open("r", encoding="utf-8") as file:
        data = json.load(file)

    hiddenWeights = data["hiddenWeights"]
    hiddenBiases = data["hiddenBiases"]
    outputWeights = data["outputWeights"]
    outputBiases = data["outputBiases"]

    return hiddenWeights, hiddenBiases, outputWeights, outputBiases