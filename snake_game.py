import pygame
import random
import os
from collections import deque

# Initialize Pygame
pygame.init()

# Game settings
CELL_SIZE = 40
GRID_WIDTH = 19
GRID_HEIGHT = 15
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 10

# Colors
BLACK = (0, 0, 0)
DARK_GRAY = (30, 30, 30)


# Load assets
snake_head_img = pygame.image.load(os.path.join("assets", "snake-head.png"))
snake_body_img = pygame.image.load(os.path.join("assets", "snake-body.png"))
apple_img = pygame.image.load(os.path.join("assets", "apple.png"))

# Scale assets to cell size
snake_head_img = pygame.transform.scale(snake_head_img, (CELL_SIZE, CELL_SIZE))
snake_body_img = pygame.transform.scale(snake_body_img, (CELL_SIZE, CELL_SIZE))
apple_img = pygame.transform.scale(apple_img, (CELL_SIZE, CELL_SIZE))

# Scale body size down to create appearance of gap between segments 
body_scale = 0.98
body_size = int(CELL_SIZE * body_scale)
snake_body_img = pygame.transform.scale(snake_body_img, (body_size, body_size))


# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Snake setup
snake = [(9, 7)]
direction = (1, 0)
direction_queue = deque()


# Food setup
while True:
    food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    if food not in snake:
        break




# Functions
def get_head_rotation(direction):
    if direction == (1, 0):   # Right
        return 0
    elif direction == (0, 1): # Down
        return 270
    elif direction == (-1, 0):# Left
        return 180
    elif direction == (0, -1):# Up
        return 90
    
def get_direction_from_key(key):
    if key == pygame.K_UP:
        return (0, -1)
    elif key == pygame.K_DOWN:
        return (0, 1)
    elif key == pygame.K_LEFT:
        return (-1, 0)
    elif key == pygame.K_RIGHT:
        return (1, 0)


# Game loop
clock = pygame.time.Clock()
running = True
game_started = False

while running:
    clock.tick(FPS)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if not game_started:
                first_direction = get_direction_from_key(event.key)
                if first_direction:
                    direction = first_direction
                game_started = True
            else:
                next_direction = get_direction_from_key(event.key)
                if next_direction:
                    direction_queue.append(next_direction)

    if game_started:
        # Set direction once per frame to avoid illegal 180 crash
        while direction_queue:
            proposed = direction_queue.popleft()
            if (proposed[0] != -direction[0] or proposed[1] != -direction[1]):
                direction = proposed
                break

        # Move snake
        head_x, head_y = snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])
        snake.insert(0, new_head)

        # Check for food
        if new_head == food:
            # Place the apple on an unoccupied spot is possible
            if len(snake) == GRID_WIDTH * GRID_HEIGHT:
                # Snake has filled the board — game won
                running = False
            else:
                while True:
                    new_food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                    if new_food not in snake:
                        food = new_food
                        break
        else:
            snake.pop()

        # Check for collisions
        if (
            new_head in snake[1:] or
            new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT
        ):
            running = False

    # Draw everything
    screen.fill(BLACK)
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            color = BLACK if (row + col) % 2 == 0 else DARK_GRAY
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )


    
    
    screen.blit(apple_img, (food[0] * CELL_SIZE, food[1] * CELL_SIZE))


    for i, (x, y) in enumerate(snake):
        if i == 0:
            angle = get_head_rotation(direction)
            rotated_head = pygame.transform.rotate(snake_head_img, angle)
            screen.blit(rotated_head, (x * CELL_SIZE, y * CELL_SIZE))

        else:
            offset = (CELL_SIZE - body_size) // 2
            screen.blit(snake_body_img, (x * CELL_SIZE + offset, y * CELL_SIZE + offset))

    pygame.display.set_caption(f"Snake Game — FPS: {int(clock.get_fps())}")

    pygame.display.flip()

pygame.quit()