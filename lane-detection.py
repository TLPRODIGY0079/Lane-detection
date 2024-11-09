
import gym
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the environment
class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # Actions: left, straight, right
        self.observation_space = gym.spaces.Discrete(5)  # Simplified state space for demo

        self.state = 2  # Start in the middle
        self.steps = 0
        self.max_steps = 50

    def reset(self):
        self.state = 2
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1

        if action == 0:  # Left
            self.state = max(0, self.state - 1)
        elif action == 2:  # Right
            self.state = min(4, self.state + 1)

        reward = 1 if self.state == 4 else -1  # Reward reaching the rightmost state
        done = self.steps >= self.max_steps or self.state == 4
        return self.state, reward, done, {}

    def render(self):
        print(f"Step: {self.steps} | Position: {self.state}")

env = CarEnv()

# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
num_episodes = 1000

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Q-learning formula
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 100 == 0:
        print(f"Episode: {episode} | Total Reward: {total_reward}")

# Plotting the Q-values
plt.imshow(q_table, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

# Run a demo to show the car's learning
for episode in range(5):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Episode: {episode} | Total Reward: {total_reward}")
