import numpy as np
from pud.buffer import ReplayBuffer

class VisualReplayBuffer (ReplayBuffer):
    def __init__(self, obs_dim, goal_dim, action_dim, max_size=int(1e6)):
        super(VisualReplayBuffer, self).__init__(
                obs_dim=obs_dim, 
                goal_dim=goal_dim, 
                action_dim=action_dim, 
                max_size=max_size,
                )

        obs_shape = (max_size, obs_dim)
        if isinstance(obs_dim, tuple):
            obs_shape = (max_size, *obs_dim)

        self.observation = np.zeros(obs_shape)
        self.goal = np.zeros(obs_shape)
        self.next_observation = np.zeros(obs_shape)
        self.next_goal = np.zeros(obs_shape)
