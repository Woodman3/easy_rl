import gymnasium as gym
from gym import spaces
import numpy as np

class CustomCliffWalkingEnv(gym.Env):
    def __init__(self):
        super(CustomCliffWalkingEnv, self).__init__()
        self.grid_size = (4, 12)
        self.start_state = (3, 0)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.goal_state = (3, 11)
        
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size[0]),
            spaces.Discrete(self.grid_size[1])
        ))

        self.reset()

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 0:  # up
            row -= 1
        elif action == 1:  # down
            row += 1
        elif action == 2:  # left
            col -= 1
        elif action == 3:  # right
            col += 1

        # Check if out of bounds
        if row < 0 or row >= self.grid_size[0] or col < 0 or col >= self.grid_size[1]:
            return self.state, -1, True, {"info": "Out of bounds"}

        self.state = (row, col)

        # Check if the agent fell off the cliff
        if self.state in self.cliff:
            return self.state, -100, True, {"info": "Fell off the cliff"}

        # Check if the agent reached the goal
        if self.state == self.goal_state:
            return self.state, 0, True, {"info": "Goal reached"}

        return self.state, -1, False, {}

    def render(self):
        grid = np.zeros(self.grid_size)
        for pos in self.cliff:
            grid[pos] = -1
        grid[self.goal_state] = 1
        grid[self.state] = 2
        print(grid)

# Testing the custom environment
env = CustomCliffWalkingEnv()
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    state, reward, done, info = env.step(action)
    env.render()
    print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
