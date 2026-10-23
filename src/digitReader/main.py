import pygame
import cv2
import model

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Draw a Digit (0-9)!")

    BLACK = (255, 255, 255)
    WHITE = (0, 0, 0)

    screen.fill(WHITE)
    drawing = False
    last_position = None

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    drawing = True
                    last_position = event.pos
                    pygame.draw.circle(screen, BLACK, event.pos, 20)

            elif event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.line(
                    screen, BLACK, last_position, event.pos, 40
                )
                last_position = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_position = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    pygame.image.save(screen, "user_drawing.png")
                    print("Image saved as user_drawing.png!")

                elif event.key == pygame.K_c:
                    screen.fill(WHITE)

                elif event.key == pygame.K_d:
                    pygame.image.save(screen, "user_drawing.png")
                    print("Image saved as user_drawing.png!")
                    screen.fill(WHITE)
                    number = predict("user_drawing.png")

        pygame.display.flip()
        clock.tick(60)

    
    pygame.image.save(screen, "user_drawing.png")
    pygame.quit()

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
    cv2.imwrite("output_image.jpg", resized_image)

    pixel_matrix = resized_image.flatten().tolist()

    # Normalize pixels to the same range used during training
    pixel_matrix = [pixel / 255 for pixel in pixel_matrix]

    return pixel_matrix

def predict(filename):
    pixel_matrix = readImage(filename)
    if pixel_matrix is None:
        return 1
    prediction = model.predict(pixel_matrix)
    print("Predicted Number:", prediction)

if __name__ == "__main__":
    main()