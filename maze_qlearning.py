import numpy as np
import matplotlib.pyplot as plt
import time

print("You have imported all the libraries.")

class Maze:
    def __init__(self, maze, start, goal, goal_reward=1, wall_penalty=-1, step_penalty=-0.1):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.current_position = start
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.step_penalty = step_penalty

    def reset(self):
        self.current_position = self.start
        return self.current_position

    def step(self, action):
        x, y = self.current_position
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.maze.shape[0] - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.maze.shape[1] - 1)
        
        if self.maze[x, y] == 1:  # Wall
            x, y = self.current_position  # No move
            reward = self.wall_penalty
        elif (x, y) == self.goal:
            reward = self.goal_reward
        else:
            reward = self.step_penalty
        
        self.current_position = (x, y)
        done = self.current_position == self.goal
        return self.current_position, reward, done, {}

    def render(self):
        maze_copy = self.maze.copy()
        x, y = self.current_position
        maze_copy[x, y] = 2
        plt.imshow(maze_copy, cmap='hot')
        plt.show()

class QLearningAgent:
    def __init__(self, maze, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((*maze.maze.shape, 4))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(4)  # Explore
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Exploit

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.alpha * (target - predict)

def train_agent(agent, maze, num_episodes=1000):
    for episode in range(num_episodes):
        state = maze.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = maze.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

def test_agent(agent, maze):
    state = maze.reset()
    maze.render()
    done = False
    steps = 0
    total_reward = 0
    while not done:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, _ = maze.step(action)
        total_reward += reward
        maze.render()
        state = next_state
        steps += 1
        time.sleep(0.5)
        if done:
            print(f"Reached the goal in {steps} steps with a total reward of {total_reward}!" if reward == maze.goal_reward else f"Fell into a hole in {steps} steps with a total reward of {total_reward}.")

# Define the maze layout
maze_layout = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0]
])

start = (0, 0)
goal = (4, 4)
goal_reward = 100
wall_penalty = -10
step_penalty = -1

# Create the maze environment
maze = Maze(maze_layout, start, goal, goal_reward, wall_penalty, step_penalty)

# Create the agent
agent = QLearningAgent(maze)

# Train the agent
train_agent(agent, maze, num_episodes=1000)

# Test the trained agent
test_agent(agent, maze)

# Example of plotting the Q-values for each state
plt.imshow(np.max(agent.q_table, axis=2), cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Q-values Heatmap")
plt.xlabel("Y")
plt.ylabel("X")
plt.show()
