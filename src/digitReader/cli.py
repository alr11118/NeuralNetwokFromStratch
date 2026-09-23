import argparse
import cv2
from . import model

def readImage(filePath):
    image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from {filePath}")
        return None

    resized_image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_LINEAR
    )

    pixel_matrix = resized_image.flatten().tolist()

    # Normalize pixels to the same range used during training
    pixel_matrix = [pixel / 255 for pixel in pixel_matrix]

    return pixel_matrix

def main():
    ap = argparse.ArgumentParser(description="Load an image via CLI")
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(ap.parse_args())
    pixel_matrix = readImage(args["image"])
    if pixel_matrix is None:
        return 1
    prediction = model.predict(pixel_matrix)
    print("Predicted Number:", prediction)
    return 0

if(__name__ == "__main__"):
    raise SystemExit(main())