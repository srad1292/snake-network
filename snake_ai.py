import pygame
import random
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# For quicker training without visuals
HEADLESS = True  # Set to False to re-enable visuals
MAX_GAMES = 500
game_count = 0
scores = []

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

max_moves = 150
moves_left = max_moves


# Food setup
while True:
    food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    if food not in snake:
        break

# Tensorflow
class DQN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(3, activation=None)  # 3 actions

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


# Agent
class Agent:
    def __init__(self):
        if os.path.exists("snake_model.h5"):
            self.model = tf.keras.models.load_model("snake_model.keras")
        else:
            self.model = DQN()
            self.model.compile(optimizer='adam', loss='mse')
        self.memory = deque(maxlen=100_000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate - set based on where last run ended or at 1.0 for fresh agent
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 100

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 2)  # explore
        state = np.array([state], dtype=np.float32)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q = np.max(self.model.predict(np.array([next_state]), verbose=0)[0])
            target = reward + self.gamma * next_q

        target_f = self.model.predict(np.array([state]), verbose=0)[0]
        target_f[action] = target

        self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0)


    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q = np.max(self.model.predict(np.array([next_state]), verbose=0)[0])
                target = reward + self.gamma * next_q

            target_f = self.model.predict(np.array([state]), verbose=0)[0]
            target_f[action] = target

            states.append(state)
            targets.append(target_f)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = Agent()




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
    
def apply_action_to_direction(action, current_direction):
    # Map current direction to left/right/straight
    x, y = current_direction
    if action == 0:  # straight
        return (x, y)
    elif action == 1:  # left turn
        return (-y, x)
    elif action == 2:  # right turn
        return (y, -x)


def get_state(snake, food, direction):
    head = snake[0]
    point_l = (head[0] - direction[1], head[1] + direction[0])  # left turn
    point_r = (head[0] + direction[1], head[1] - direction[0])  # right turn
    point_s = (head[0] + direction[0], head[1] + direction[1])  # straight

    def is_danger(point):
        return (
            point in snake or
            point[0] < 0 or point[0] >= GRID_WIDTH or
            point[1] < 0 or point[1] >= GRID_HEIGHT
        )

    state = [
        is_danger(point_s),
        is_danger(point_l),
        is_danger(point_r),
        direction == (0, -1),  # up
        direction == (0, 1),   # down
        direction == (-1, 0),  # left
        direction == (1, 0),   # right
        food[1] < head[1],     # food up
        food[1] > head[1],     # food down
        food[0] < head[0],     # food left
        food[0] > head[0],     # food right
    ]
    return list(map(int, state))


# Game loop
clock = pygame.time.Clock()
running = True
game_started = False

while running:
    if not HEADLESS:
        clock.tick(FPS)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    state = get_state(snake, food, direction)
    prev_distance = abs(food[0] - snake[0][0]) + abs(food[1] - snake[0][1])

    action = agent.get_action(state)
    direction = apply_action_to_direction(action, direction)
    game_started = True
    game_won = False
    crashed = False

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
        new_distance = abs(food[0] - new_head[0]) + abs(food[1] - new_head[1])
        

        snake.insert(0, new_head)
        moves_left = moves_left - 1

        # Check for food
        if new_head == food:
            moves_left = moves_left + 80
            # Place the apple on an unoccupied spot is possible
            if len(snake) == GRID_WIDTH * GRID_HEIGHT:
                # Snake has filled the board — game won
                game_won = True
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
            crashed = True
    
    reward = 0
    done = False

    

    if new_head == food:
        reward = 20
    elif game_won:
        reward = 100
        done = True
    elif crashed:
        reward = -20
        done = True
    elif moves_left == 0:
        reward = -8
        done = True
        crashed = True 
    elif new_distance < prev_distance:
        reward += 0.3  # small bonus for moving closer
    else:
        reward -= 0.3

    next_state = get_state(snake, food, direction)

    agent.remember(state, action, reward, next_state, done)
    agent.train_short_memory(state, action, reward, next_state, done)


    if crashed or game_won:
        score = len(snake) - 1  # or however you define score
        scores.append(score)


        game_count += 1
        print(f"Game {game_count} — Score: {score} — Epsilon: {agent.epsilon:.3f}")
        agent.train_long_memory()
        if game_count >= MAX_GAMES:
            agent.model.save("snake_model.keras")
            with open("scores.txt", "a") as f:
                for s in scores:
                    f.write(f"{s}\n")
            running = False

        max_moves = 150
        moves_left = max_moves
        snake = [(9, 7)]
        direction = (1, 0)
        direction_queue.clear()
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in snake:
                break
        game_started = False
        continue

    # Draw everything
    if not HEADLESS:
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

        pygame.display.set_caption(f"Snake Game — FPS: {int(clock.get_fps())} - Moves Left: {moves_left}")

        pygame.display.flip()

pygame.quit()