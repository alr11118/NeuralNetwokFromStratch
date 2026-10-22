import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Draw a Digit (0-9)!")

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

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
                    pygame.draw.circle(screen, BLACK, event.pos, 3)

            elif event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.line(
                    screen, BLACK, last_position, event.pos, 6
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

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()