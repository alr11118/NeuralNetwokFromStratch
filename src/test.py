from . import MINST_loader
from . import model 

def test_withLoadedData(inputs, outputs):
    correct = 0
    for i in range(len(inputs)):
        predictedClass = model.predict(inputs[i])
        if predictedClass == outputs[i]:
            correct += 1

    accuracy = correct/len(inputs)
    return accuracy

def test(testSize, useTestset = True):
    
    x_test = MINST_loader.loadImages(
        'data/t10k-images.idx3-ubyte' if useTestset else 'data/train-images.idx3-ubyte',
        testSize
    )
    y_test = MINST_loader.loadLabels(
        'data/t10k-labels.idx1-ubyte' if useTestset else 'data/train-labels.idx1-ubyte',
        testSize
    )
    correct = 0
    for i in range(len(x_test)):
        predictedClass = model.predict(x_test[i])
        if predictedClass == y_test[i]:
            correct += 1

    accuracy = correct/len(x_test)
    return accuracy

def main():
    x_test = MINST_loader.loadImages(
        "data/t10k-images.idx3-ubyte",
        5000
    )
    y_test = MINST_loader.loadLabels(
        "data/t10k-labels.idx1-ubyte",
        5000
    )
    accuracy = test_withLoadedData(x_test, y_test)
    print(f"Accuracy: {accuracy} ({accuracy * 100:.2f}%)")

if(__name__ == "__main__"):
    main()