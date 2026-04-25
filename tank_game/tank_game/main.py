import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tank Battle Game")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Game Constants
FPS = 60
clock = pygame.time.Clock()

# --- Tank Class ---
class Tank(pygame.sprite.Sprite):
    def __init__(self, x, y, color, speed):
        super().__init__()
        # Create a simple rectangle for the tank
        self.image = pygame.Surface([40, 40])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed

    def update(self, keys):
        # Handle movement based on key presses
        dx = 0
        dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = self.speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -self.speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = self.speed

        self.rect.x += dx
        self.rect.y += dy

        # Keep tank within screen boundaries
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

# --- Game Setup ---
all_sprites = pygame.sprite.Group()
tanks = pygame.sprite.Group()

# Create Player Tank
player_speed = 5
player = Tank(SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2, BLUE, player_speed)
all_sprites.add(player)
tanks.add(player)

# --- Game Loop ---
running = True
while running:
    # 1. Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Get current state of all keys pressed
    keys = pygame.key.get_pressed()

    # 2. Update
    all_sprites.update(keys)

    # 3. Drawing
    screen.fill(BLACK)  # Background
    
    # Draw some boundary (simple grid/border)
    pygame.draw.rect(screen, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 2)

    # Draw all sprites
    all_sprites.draw(screen)

    # 4. Refresh Screen
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()