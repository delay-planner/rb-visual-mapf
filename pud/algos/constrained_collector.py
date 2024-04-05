from __future__ import annotations
import numpy as np
from pud.collector import Collector
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from typing import Union, Optional, List
from pud.policies import GaussianPolicy

def calc_cost_class_inds(cost:Union[float, List[float], np.ndarray], cost_classes:np.ndarray) -> Union[int, np.ndarray]:
    """class-independent method to discretize the float cost into classes
    """
    ret_scalar = False
    if isinstance(cost, float):
        ret_scalar = True
        cost = np.array([cost])
    elif isinstance(cost, list):
        cost = np.array(cost)
    
    # clip the cost range, otherwise the argmin trick will not work
    cost = np.clip(cost, cost_classes[0], cost_classes[-1])
    
    num_bins = int( len(cost_classes) )
    cost = np.expand_dims(cost, -1)
    class_mat = (cost_classes >= cost).astype(float) # batch_size, num_bins
    class_mat = class_mat * np.arange(-num_bins,0)
    class_inds = np.argmin(class_mat, axis=-1)

    if ret_scalar:
        return int(class_inds)
    return class_inds

class ConstrainedCollector (Collector):
    def __init__(
            self, 
            policy: GaussianPolicy, 
            buffer:ConstrainedReplayBuffer, 
            env, 
            initial_collect_steps:int=0,
            ):
        super(ConstrainedCollector, self).__init__(
            policy=policy, buffer=buffer, env=env, initial_collect_steps=initial_collect_steps,
        )
        assert isinstance(self.buffer, ConstrainedReplayBuffer), "Error: need to use ConstrainedReplayBuffer"

        self.past_eps = []
        self.num_eps = 0
        self.state, info = env.reset()
        self._reset_log()

    def _reset_log(self) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        self._ep_ret = 0.0
        self._ep_cost = 0.0
        self._ep_len = 0.0
        self._ep_cost_max = 0.0

    def _append_ep_log(self):
        self.past_eps.append(
            {
                "ep_ret": self._ep_ret,
                "ep_cost": self._ep_cost,
                "ep_len": self._ep_len,
                "ep_cost_max": self._ep_cost_max,
            }
        )
        self.num_eps += 1
    
    def _log_metric(self, reward:float, cost:float):
        self._ep_len += 1
        self._ep_ret += reward
        self._ep_cost += cost
        if self._ep_cost_max < cost:
            self._ep_cost_max = cost

    def step(self, num_steps):
        """step num_steps in the env. 
        ! the env is not reset before stepping, the env is kept alive after exiting this method 
        """
        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(self.state)

            next_state, reward, done, info = self.env.step(np.copy(action))
            self._log_metric(reward, cost=info["cost"])

            if info.get('last_step', False):
                self.buffer.add(self.state, action, info['terminal_observation'], reward, info['cost'], done)
                self._append_ep_log()
                self._reset_log()
            else:
                self.buffer.add(self.state, action, next_state, reward, info['cost'], done)
                
            self.state = next_state

            self.steps += 1

    @classmethod
    def sample_initial_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            s0, info = eval_env.reset()
            rb_vec.append(s0)
        rb_vec = np.array([x['observation'] for x in rb_vec])
        return rb_vec

    @classmethod
    def eval_agent(cls, policy, eval_env, n, by_episode=True):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """
        c = 0 # count
        r = 0 # reward
        rewards = []

        co = 0
        cum_co = 0
        max_co = 0
        max_costs = []
        cum_costs = []
        state = eval_env.reset()
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode: c += 1

            r += reward

            co = info.get('cost', 0.0)
            if co > max_co:
                max_co = co
            cum_co += co

            if done:
                rewards.append(r)
                cum_costs.append(cum_co)
                max_costs.append(max_co)
                if by_episode: c += 1
                r = 0

                co = 0
                cum_co = 0
                max_co = 0
        
        eval_outputs = {
            'returns': rewards,
            'max_costs': max_costs,
            'cum_costs': cum_costs,
        }
        return eval_outputs


    @classmethod
    def eval_agent_n_record_init_states(cls, policy, eval_env, n, by_episode=True):
        """
        Run evaluation and records the initial states for each episode
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        records = {}
        def new_record(init_state:Union[np.ndarray, dict]):
            key = len(records.keys())
            records[key] = {
                "rewards": 0.,
                "costs": 0.,
                "max_step_cost": 0.,
                "init_states": init_state,
                "steps": 0,
            }
            return key

        c = 0 # count

        r, co, max_co, cum_co = [0.] * 4
        state, info = eval_env.reset()
        cur_key = new_record(state)
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode: c += 1
            records[cur_key]["steps"] += 1

            r += reward

            co = info.get('cost', 0.0)
            if co > max_co:
                max_co = co
            cum_co += co

            if done:
                records[cur_key]["rewards"] = r
                records[cur_key]["costs"] = co
                records[cur_key]["max_step_cost"] = max_co
                if by_episode: 
                    c += 1
                    if c < n:
                        cur_key = new_record(state)
                        assert state['first_step']
                
                r, co, max_co, cum_co = [0.] * 4
        return records

    @classmethod
    def step_cleanup(cls, search_policy, eval_env, num_steps):
        # cls stands for class
        c = 0
        while c < num_steps:
            goal = search_policy.get_goal_in_rb()
            state = eval_env.reset()
            done = False

            while True:
                state['goal'] = goal
                try:
                    action = search_policy.select_action(state)
                except Exception as e:
                    raise e

                state, reward, done, info = eval_env.step(np.copy(action))
                c += 1

                if done or c >= num_steps or search_policy.reached_final_waypoint:
                    break

    @classmethod
    def get_trajectory(cls, policy, eval_env):
        ep_observation_list = []
        ep_waypoint_list = []
        ep_reward_list = []

        state = eval_env.reset()
        ep_goal = state['goal']
        while True:
            ep_observation_list.append(state['observation'])
            action = policy.select_action(state) # NOTE: state['goal'] may be modified
            ep_waypoint_list.append(state['goal'])
            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info['terminal_observation']['observation'])
                break

        return ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list