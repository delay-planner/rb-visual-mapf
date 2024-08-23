from pud.dependencies import *

from pud.collector import Collector
from typing import Tuple, Dict, Union, List
import numpy as np
import gym
from copy import deepcopy

class VisualCollector (Collector):
    def __init__(self, policy, buffer, env, initial_collect_steps=0):
        super(VisualCollector, self).__init__(
             policy, buffer, env, initial_collect_steps=initial_collect_steps
        )

    @classmethod
    def eval_agent_n_trajs(cls, policy, eval_env, n, by_episode=True, verbose=False):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        c = 0
        r = 0
        traj = []

        rewards = []
        trajs = []
        success =  []

        state = eval_env.reset()
        traj.append(state)
        while c < n:
            action = policy.select_action(state)
            if verbose:
                print("episode {}, action: {}".format(c, action))

            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode: c += 1
            

            if not done:
                traj.append(deepcopy(state))
            else:
                traj.append(info["terminal_observation"])
            
            if verbose:
                print("obs:{}, action:{} goal:{}".format(info["grid"]["observation"], action, info["grid"]["goal"]))
            
            r += reward
            if done:
                rewards.append(r)
                if by_episode: c += 1
                r = 0
                trajs.append(traj)
                success.append(not info["timed_out"])
                traj = []
                if verbose:
                    print("#" * 15)
        return {"rewards": rewards, "trajs": trajs, "success": success}


