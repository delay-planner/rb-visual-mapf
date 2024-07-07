import numpy as np
from pud.buffer import ReplayBuffer
from pud.algos.data_struct import inp_to_numpy

class VisualReplayBuffer (ReplayBuffer):
    def __init__(self, obs_dim, goal_dim, action_dim, max_size=int(1e6)):
        super(VisualReplayBuffer, self).__init__(
                obs_dim=2, 
                goal_dim=2, 
                action_dim=action_dim, 
                max_size=max_size,
                )

        obs_shape, goal_shape = None, None
        if isinstance(obs_dim, tuple):
            obs_shape = (max_size, *obs_dim)
            goal_shape = (max_size, *goal_dim)
        else:
            obs_shape = (max_size, obs_dim)
            goal_shape = (max_size, goal_dim)

        self.observation = np.zeros(obs_shape)
        self.goal = np.zeros(goal_shape)
        self.next_observation = np.zeros(obs_shape)
        self.next_goal = np.zeros(goal_shape)

        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        state = inp_to_numpy(state)
        next_state = inp_to_numpy(next_state)
        super().add(state, action, next_state, reward, done)
