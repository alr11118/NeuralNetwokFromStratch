def loadImages(filename):
    with open(filename, "rb") as file:
        data = file.read()

    print(data[0:16])
    magic = int.from_bytes(data[0:4], byteorder="big")
    numImages = int.from_bytes(data[4:8], byteorder="big")
    numRows = int.from_bytes(data[8:12], byteorder="big")
    numColumns = int.from_bytes(data[12:16], byteorder="big")

    print(magic)
    print(numImages)
    print(numRows)
    print(numColumns)

    firstImage = data[16: 16 + 784]

    print(len(firstImage))
    normalisedImage = []
    for row in range(28):
        for col in range(28):
            index = 28 * row + col
            pixelValue = (firstImage[index] / 255)
            normalisedImage.append(pixelValue)
            #print(pixelValue, end = " ")
        #print("")

images = loadImages("MINST/train-images.idx3-ubyte")

def loadLabels(filename):
    with open(filename, "rb") as file:
        data = file.read()

    magic = int.from_bytes(data[0:4], byteorder="big")
    numLabels = int.from_bytes(data[4:8], byteorder="big")

    print("Magic:", magic)
    print("Number of labels:", numLabels)

    firstLabel = data[8]

    print("First label:", firstLabel)

labels = loadLabels("MINST/train-labels.idx1-ubyte")