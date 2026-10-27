#from . import train
import train
#from . import weightLoader
import weightLoader

def predict(input):
    
    (hiddenWeights, hiddenBiases, outputWeights,outputBiases) = weightLoader.loadWeights()
    probabilities, _, _ = train.forwardPass(
        input,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases
    )
    predictedClass = probabilities.index(max(probabilities))
    return predictedClass