def loadImages(filename):
    with open(filename, "rb") as file:
        data = file.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    numImages = int.from_bytes(data[4:8], byteorder="big")
    numRows = int.from_bytes(data[8:12], byteorder="big")
    numColumns = int.from_bytes(data[12:16], byteorder="big")

    """
    print(magic)
    print(numImages)
    print(numRows)
    print(numColumns)
    """

    images = []
    for imageIndex in range(numImages):
        image = []
        for pixelIndex in range(numRows * numColumns):
            index = 16 + imageIndex * (numRows * numColumns) + pixelIndex
            value = (data[index] / 255)
            image.append(value)
        images.append(image)
    return images

# Example Usage:
#images = loadImages("MINST/train-images.idx3-ubyte")

def loadLabels(filename):
    with open(filename, "rb") as file:
        data = file.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    numLabels = int.from_bytes(data[4:8], byteorder="big")

    """
    print("Magic:", magic)
    print("Number of labels:", numLabels)
    """

    labels = []
    for labelIndex in range(numLabels):
        index = 8 + labelIndex
        value = data[index]
        labels.append(value)
    return labels

# Example Usage:
#labels = loadLabels("MINST/train-labels.idx1-ubyte")