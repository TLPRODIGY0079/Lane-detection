import pygame
import numpy as np
import random

#Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Car settings
CAR_WIDTH, CAR_HEIGHT = 60, 40  # Adjusted dimensions for a 2D car
car_x, car_y = WIDTH // 2, HEIGHT - CAR_HEIGHT - 10
car_speed = 10

# Obstacle settings
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 50, 50
obstacles = [
    (200, 450), (400, 450), (600, 450),  # First row
    (100, 300), (300, 300), (500, 300), (700, 300),  # Second row
    (200, 150), (400, 150), (600, 150)  # Third row
]

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
num_episodes = 50  # Increased number of episodes to allow for longer training

# Action space: left, right, up
actions = [(-car_speed, 0), (car_speed, 0), (0, -car_speed)]
state_space_size = (WIDTH // car_speed, HEIGHT // car_speed)

# Initialize Q-table
q_table = np.zeros((state_space_size[0], state_space_size[1], len(actions)))

def get_state(car_x, car_y):
    return car_x // car_speed, car_y // car_speed

def reset_car():
    global car_x, car_y
    car_x, car_y = WIDTH // 2, HEIGHT - CAR_HEIGHT - 10

def step(action):
    global car_x, car_y
    car_x = np.clip(car_x + action[0], 0, WIDTH - CAR_WIDTH)
    car_y = np.clip(car_y + action[1], 0, HEIGHT - CAR_HEIGHT)
    reward = -1
    done = False
    
    for obs in obstacles:
        if car_x < obs[0] + OBSTACLE_WIDTH and car_x + CAR_WIDTH > obs[0] and car_y < obs[1] + OBSTACLE_HEIGHT and car_y + CAR_HEIGHT > obs[1]:
            reward = -100
            done = True
    
    if car_y <= 0:
        reward = 100
        done = True

    return get_state(car_x, car_y), reward, done

# Q-learning algorithm
for episode in range(num_episodes):
    state = get_state(car_x, car_y)
    reset_car()
    total_reward = 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]

        next_state, reward, done = step(action)
        total_reward += reward

        old_value = q_table[state[0], state[1], actions.index(action)]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], actions.index(action)] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 100 == 0:
        print(f"Episode: {episode} | Total Reward: {total_reward}")

# Main loop
running = True
reset_car()
state = get_state(car_x, car_y)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = actions[np.argmax(q_table[state[0], state[1]])]
    state, _, done = step(action)

    screen.fill(WHITE)

    for obs in obstacles:
        pygame.draw.rect(screen, RED, pygame.Rect(obs[0], obs[1], OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

    pygame.draw.rect(screen, BLUE, pygame.Rect(car_x, car_y, CAR_WIDTH, CAR_HEIGHT))

    pygame.display.flip()
    pygame.time.delay(50)  # Adjusted delay for smoother movement

    if done:
        reset_car()
        state = get_state(car_x, car_y)

pygame.quit()

