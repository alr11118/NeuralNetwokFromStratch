from . import train
from . import weightLoader

def predict(input):
    
    (hiddenWeights, hiddenBiases, outputWeights,outputBiases) = weightLoader.loadWeights("weights.json")
    probabilities, _, _ = train.forwardPass(
        input,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases
    )
    predictedClass = probabilities.index(max(probabilities))
    return predictedClass