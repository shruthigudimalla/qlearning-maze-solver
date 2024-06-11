# Q-learning Maze Solver

This project demonstrates the implementation of a Q-learning agent to solve a simple maze. The agent learns to navigate the maze by maximizing cumulative rewards, avoiding obstacles, and reaching the goal.

## Table of Contents
- Introduction
- Dependencies
- Setup
- Usage
- Project Structure
- Q-learning Algorithm
- Future Improvements
- Contributing
- License

## Introduction
Q-learning is a model-free reinforcement learning algorithm that enables an agent to learn how to achieve its goals by interacting with an environment. This project uses Q-learning to train an agent to navigate a 5x5 maze, avoiding walls and reaching a goal point.

## Dependencies
This project requires the following Python libraries:
- numpy
- matplotlib
- gym

Install the dependencies using pip:

```
pip install numpy matplotlib gym
```

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/shruthigudimalla/qlearning-maze-solver.git
   cd qlearning-maze-solver
   ```

2. Save the provided Python script as `maze_qlearning.py` in the project directory.

## Usage
Run the Python script to train and test the Q-learning agent:

```
python maze_qlearning.py
```

The script will:
1. Set up the maze environment.
2. Train the Q-learning agent for 1000 episodes.
3. Test the trained agent, displaying the maze and the agent's path.
4. Print the number of steps and the total rewards for each test run.
5. Display a heatmap of the Q-values for each state.

## Project Structure
```
qlearning-maze-solver/
│
├── maze_qlearning.py      # Main script to train and test the Q-learning agent
└── README.txt             # Project README file
```

## Q-learning Algorithm
The Q-learning agent uses the following parameters:
- `alpha` (Learning rate): 0.1
- `gamma` (Discount factor): 0.9
- `epsilon` (Exploration rate): 0.1

The reward system is defined as:
- Goal reward: 100
- Wall penalty: -10
- Step penalty: -1

The agent updates its Q-table based on the rewards received for each action taken during training. The trained agent is then tested in the maze, and its performance is visualized and printed.

## Future Improvements
- Experiment with different maze layouts and complexities.
- Adjust the reward system to see its impact on the agent's learning.
- Implement a more complex agent with additional features (e.g., deep Q-learning).
- Add visualization of the training process to observe how the agent's behavior evolves.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

