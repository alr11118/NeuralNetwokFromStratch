def loadImages(filename, batchSize = 0):
    with open(filename, "rb") as file:
        data = file.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    numImages = int.from_bytes(data[4:8], byteorder="big")
    numRows = int.from_bytes(data[8:12], byteorder="big")
    numColumns = int.from_bytes(data[12:16], byteorder="big")

    if(batchSize == 0):
        batchSize = numImages

    images = []
    for imageIndex in range(batchSize):
        image = []
        for pixelIndex in range(numRows * numColumns):
            index = 16 + imageIndex * (numRows * numColumns) + pixelIndex
            value = (data[index] / 255)
            image.append(value)
        images.append(image)
    return images

def loadLabels(filename, batchSize = 0):
    with open(filename, "rb") as file:
        data = file.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    numLabels = int.from_bytes(data[4:8], byteorder="big")

    if(batchSize == 0):
        batchSize = numLabels

    labels = []
    for labelIndex in range(batchSize):
        index = 8 + labelIndex
        value = data[index]
        labels.append(value)
    return labels