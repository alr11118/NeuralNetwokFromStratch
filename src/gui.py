import pygame
import cv2
import model
from pathlib import Path

DRAWINGS_DIR = Path.home() / "DigitReader" / "drawings"
DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)

USER_DRAWING = DRAWINGS_DIR / "user_drawing.png"
OUTPUT_IMAGE = DRAWINGS_DIR / "output_image.jpg"


def main():
    # Initialize Pygame
    pygame.init()

    WIDTH = 600
    HEIGHT = 600 

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw a Digit (0-9)! | MANUAL: s: save, c: clear, d: get prediction")

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    screen.fill(BLACK)

    # Drawing state
    drawing = False
    last_pos = None
    brush_color = WHITE
    brush_radius = 20

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse button pressed: start drawing
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    drawing = True
                    last_pos = event.pos

            # Mouse button released: stop drawing
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    last_pos = None

            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    current_pos = event.pos
                    pygame.draw.line(screen, brush_color, last_pos, current_pos, brush_radius * 2)
                    pygame.draw.circle(screen, brush_color, current_pos, brush_radius)
                    last_pos = current_pos

            elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_s:
                                pygame.image.save(screen, str(USER_DRAWING))
            
                            elif event.key == pygame.K_c:
                                screen.fill(BLACK)
            
                            elif event.key == pygame.K_d:
                                pygame.image.save(screen, str(USER_DRAWING))
                                number = predict(str(USER_DRAWING))
                                screen.fill(BLACK)
                                my_font = pygame.font.Font(None, 50)
                                text_surface = my_font.render("Predicted Digit: "+str(number), True, (255, 255, 255))
                                screen.blit(text_surface, (50, 50))
                                pygame.display.flip()
                                pygame.event.pump()
                                pygame.time.wait(3000)
                                screen.fill(BLACK)


        pygame.display.flip()
        clock.tick(60)

    pygame.image.save(screen, "user_drawing.png")
    pygame.quit()

def readImage(filePath):
    image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        #print(f"Error: Could not load image from {filePath}")
        return None

    resized_image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite(str(OUTPUT_IMAGE), resized_image)

    pixel_matrix = resized_image.flatten().tolist()

    # Normalize pixels to the same range used during training
    pixel_matrix = [pixel / 255 for pixel in pixel_matrix]

    return pixel_matrix

def predict(filename):
    pixel_matrix = readImage(filename)
    if pixel_matrix is None:
        return 1
    prediction = model.predict(pixel_matrix)
    #("Predicted Number:", prediction)
    return prediction

if __name__ == "__main__":
     main()
