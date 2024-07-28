import numpy as np
from collections.abc import Sequence


class Q_greedy:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.n_state = np.zeros((state_dim,action_dim))
        self.q_table = np.random.rand(state_dim, action_dim)
        for i in range(state_dim):
            self.q_table[i] = self.q_table[i] / np.sum(self.q_table[i])
    
    def choose_action(self, state):
        epsion = np.random.rand()
        if epsion > 0.1:
            return np.argmax(self.q_table[state])
        else:
            return np.random.choice(self.action_dim)
    
    def update(self, states:Sequence[np.array], actions:Sequence[np.array], reward:Sequence[np.array]):
        for i in range(len(states)):
            state = states[i]
            action = action[i]
            n_state[state][action] += 1
            self.q_table[state][action] += (reward[i] - self.q_table[state][action]) / n_state[state][action]
            
