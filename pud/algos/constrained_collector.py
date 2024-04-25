from __future__ import annotations

import numpy as np
from pud.collector import Collector
from typing import Union, List, Tuple
from pud.policies import GaussianPolicy, SearchPolicy
from pud.algos.constrained_buffer import ConstrainedReplayBuffer


def calc_cost_class_inds(
    cost: Union[float, List[float], np.ndarray], cost_classes: np.ndarray
) -> Union[int, np.ndarray]:
    """Class-independent method to discretize the float cost into classes"""
    ret_scalar = False
    if isinstance(cost, float):
        ret_scalar = True

        cost = np.array([cost])
    elif isinstance(cost, list):
        cost = np.array(cost)

    # Clip the cost range, otherwise the argmin trick will not work
    cost = np.clip(cost, cost_classes[0], cost_classes[-1])

    num_bins = int(len(cost_classes))
    cost = np.expand_dims(cost, -1)
    class_mat = (cost_classes >= cost).astype(float)  # batch_size, num_bins
    class_mat = class_mat * np.arange(-num_bins, 0)
    class_inds = np.argmin(class_mat, axis=-1)

    if ret_scalar:
        return int(class_inds)
    return class_inds

def eval_agent_from_Q(policy, eval_env, collect_trajs=False):
    """
    Run evaluation and records the initial states for each episode
    until the pb Q from the env is empty
    """
    # verify the eval_env has an non-empty Q of pbs
    assert hasattr(eval_env, "pb_Q") and len(getattr(eval_env, "pb_Q")) > 0
    """At the end of the last pb in the Q, the step will trigger another
    reset, and it should be safely handled by the reset_orig, suppress the warning
    message by turning off verbose"""
    eval_env.set_verbose(False)
    eval_env.set_use_q(True)

    records = {}
    def new_record(init_state: Union[np.ndarray, dict], info:dict={}):
        key = len(records.keys())
        records[key] = {
            "rewards": 0.0,
            "cum_costs": info["cost"],
            "max_step_cost": 0.0,
            "steps": 0,
            "init_info": info,
        }
        if collect_trajs:
            records[key]["traj"] = [eval_env.get_internal_state()]
        records[key]["init_states"] = eval_env.de_normalize_goal_conditioned_obs(init_state)
        return key

    c = 0  # count
    n = eval_env.get_Q_size()

    state, info = eval_env.reset()
    cur_key = new_record(state, info)

    while c < n:
        action = policy.select_action(state)
        """when episode ends:
        - state is the new state of the new epsiode
        - reward, done, info are from the last step of the terminated epsiode
        """
        state, reward, done, info = eval_env.step(np.copy(action))

        records[cur_key]["steps"] += 1
        records[cur_key]["rewards"] += reward

        if (not done) and collect_trajs:
            records[cur_key]["traj"].append(eval_env.get_internal_state())

        co = info.get("cost", 0.0)
        if co > records[cur_key]["max_step_cost"]:
            records[cur_key]["max_step_cost"] = co
        records[cur_key]["cum_costs"] += co

        if done:
            records[cur_key]["success"] = info["success"]
            if "terminal_observation" in info:
                records[cur_key]["traj"].append(
                    eval_env.de_normalize_obs(info["terminal_observation"]["observation"])
                    )

            c += 1
            if c < n:
                cur_key = new_record(state, state["first_info"])
                assert state["first_step"]
            else:
                eval_env.set_use_q(False)
    
    eval_env.set_verbose(True)
    return records

class ConstrainedCollector(Collector):
    def __init__(
        self,
        policy: GaussianPolicy,
        buffer: ConstrainedReplayBuffer,
        env,
        initial_collect_steps: int = 0,
    ):
        super(ConstrainedCollector, self).__init__(
            policy=policy,
            buffer=buffer,
            env=env,
            initial_collect_steps=initial_collect_steps,
        )
        assert isinstance(
            self.buffer, ConstrainedReplayBuffer
        ), "Error: Need to use ConstrainedReplayBuffer"

        self.past_eps = []
        self.num_eps = 0
        self.state, info = env.reset()
        self._reset_log()

    def _reset_log(self) -> None:
        """
        Reset the episode return, episode cost and episode length.

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

    def _log_metric(self, reward: float, cost: float):
        self._ep_len += 1
        self._ep_ret += reward
        self._ep_cost += cost
        if self._ep_cost_max < cost:
            self._ep_cost_max = cost

    def step(self, num_steps):
        """
        Step num_steps in the env.
        NOTE: The env is not reset before stepping, the env is kept alive after exiting this method
        """
        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(self.state)

            next_state, reward, done, info = self.env.step(np.copy(action))
            self._log_metric(reward, cost=info["cost"])

            if info.get("last_step", False):
                self.buffer.add(
                    self.state,
                    action,
                    info["terminal_observation"],
                    reward,
                    info["cost"],
                    done,
                )
                self._append_ep_log()
                self._reset_log()
            else:
                self.buffer.add(
                    self.state, action, next_state, reward, info["cost"], done
                )

            self.state = next_state

            self.steps += 1

    @classmethod
    def sample_initial_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            s0, info = eval_env.reset()
            rb_vec.append(s0)
        rb_vec = np.array([x["observation"] for x in rb_vec])
        return rb_vec

    @classmethod
    def eval_agent(cls, policy, eval_env, n, by_episode=True):
        """
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """
        c = 0  # count
        r = 0  # reward
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
            if not by_episode:
                c += 1

            r += reward

            co = info.get("cost", 0.0)
            if co > max_co:
                max_co = co
            cum_co += co

            if done:
                rewards.append(r)
                cum_costs.append(cum_co)
                max_costs.append(max_co)
                if by_episode:
                    c += 1
                r = 0

                co = 0
                cum_co = 0
                max_co = 0

        eval_outputs = {
            "returns": rewards,
            "max_costs": max_costs,
            "cum_costs": cum_costs,
        }
        return eval_outputs

    @classmethod
    def eval_agent_n_record_init_states(cls, policy, eval_env, n, by_episode=True):
        """
        Run evaluation and records the initial states for each episode
        by_episode: if True, evals `n` episodes; otherwise, evals `n` environment steps
        """

        records = {}

        def new_record(init_state: Union[np.ndarray, dict]):
            key = len(records.keys())
            records[key] = {
                "rewards": 0.0,
                "costs": 0.0,
                "max_step_cost": 0.0,
                "init_states": init_state,
                "steps": 0,
            }
            return key

        c = 0  # count

        r, co, max_co, cum_co = [0.0] * 4
        state, info = eval_env.reset()
        cur_key = new_record(state)
        while c < n:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))
            if not by_episode:
                c += 1
            records[cur_key]["steps"] += 1

            r += reward

            co = info.get("cost", 0.0)
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
                        assert state["first_step"]

                r, co, max_co, cum_co = [0.0] * 4
        return records

    @classmethod
    def step_cleanup(cls, search_policy, eval_env, num_steps):
        # cls stands for class
        c = 0
        while c < num_steps:
            goal = search_policy.get_goal_in_rb()
            state, info = eval_env.reset()
            done = False

            while True:
                state["goal"] = goal
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

        state, info = eval_env.reset()
        ep_goal = state["goal"]
        while True:
            ep_observation_list.append(state["observation"])
            action = policy.select_action(state)  # NOTE: state['goal'] may be modified
            ep_waypoint_list.append(state["goal"])
            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info["terminal_observation"]["observation"])
                break

        return ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list

    @classmethod
    def get_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        input_starts: Union[List[Tuple[float, float]], None] = None,
        input_goals: Union[List[Tuple[float, float]], None] = None,
        threshold: float = 0.05,
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[List[Tuple[float, float]]],
        List[List[Tuple[float, float]]],
        List[List[int]],
    ]:
        augmented_ep_observation_list: List[List[Tuple[float, float]]] = [
            [] for _ in range(num_agents)
        ]
        augmented_ep_waypoint_list: List[List[Tuple[float, float]]] = [
            [] for _ in range(num_agents)
        ]
        augmented_ep_reward_list: List[List[int]] = [[] for _ in range(num_agents)]

        state, info = eval_env.reset()

        if input_starts is not None and input_goals is not None:
            starts = input_starts.copy()
            goals = input_goals.copy()
            state["observation"] = np.copy(starts[0])
            state["goal"] = np.copy(goals[0])
            state["composite_starts"] = starts.copy()
            state["composite_goals"] = goals.copy()
            state["agent_observations"] = starts.copy()
            state["agent_waypoints"] = goals.copy()

            eval_env.env.env.state = np.array(
                [
                    state["observation"][0] * eval_env.env.env._height,
                    state["observation"][1] * eval_env.env.env._width,
                ],
                dtype=np.float32,
            )
            print("Using the provided starts and goals")
        else:
            state["agent_observations"] = [state["observation"]]
            state["agent_waypoints"] = [state["goal"]]

            starts: List[Tuple[float, float]] = [state["observation"]]
            goals: List[Tuple[float, float]] = [state["goal"]]
            for _ in range(num_agents - 1):
                new_obs = eval_env.env.env.sample_safe_empty_state(
                    cost_limit=eval_env.env.env.cost_limit
                )
                new_goal = None
                count = 0
                while new_goal is None:
                    new_obs = eval_env.env.env.sample_safe_empty_state(
                        cost_limit=eval_env.env.env.cost_limit
                    )
                    (new_obs, new_goal) = eval_env.env._sample_goal(new_obs)
                    count += 1
                    if count > 1000:
                        print("WARNING: Unable to find goal within constraints.")
                new_obs = eval_env.env._normalize_obs(new_obs)
                new_goal = eval_env.env._normalize_obs(new_goal)
                starts.append(new_obs)
                goals.append(new_goal)

                state["agent_observations"].append(new_obs)
                state["agent_waypoints"].append(new_goal)

            state["composite_starts"] = starts
            state["composite_goals"] = goals
            print("Sampled the required starts and goals")

        agent_done = [False for _ in range(num_agents)]
        all_done = False

        while not all_done:

            state["observation"] = state["agent_observations"][0]
            state["goal"] = state["agent_waypoints"][0]

            eval_env.env.env.state = np.array(
                [
                    state["observation"][0] * eval_env.env.env._height,
                    state["observation"][1] * eval_env.env.env._width,
                ],
            )

            if isinstance(policy, SearchPolicy):
                actions, agent_goals = policy.select_multiple_actions(state)
            # NOTE: state['goal'] may be modified

            for agent_id in range(num_agents):

                if agent_done[agent_id]:
                    continue

                if isinstance(policy, SearchPolicy):
                    state["agent_waypoints"][agent_id] = agent_goals[agent_id]

                augmented_ep_observation_list[agent_id].append(
                    state["agent_observations"][agent_id]
                )
                augmented_ep_waypoint_list[agent_id].append(
                    state["agent_waypoints"][agent_id]
                )

                state_copy = state.copy()
                state["observation"] = state["agent_observations"][agent_id]
                state["goal"] = state["agent_waypoints"][agent_id]

                if isinstance(policy, SearchPolicy):
                    action = actions[agent_id]
                else:
                    action = policy.select_action(state)

                eval_env.env.env.state = np.array(
                    [
                        state["observation"][0] * eval_env.env.env._height,
                        state["observation"][1] * eval_env.env.env._width,
                    ],
                    dtype=np.float32,
                )

                state, reward, done, info = eval_env.step(
                    np.copy(action), num_agents=num_agents
                )

                state["composite_starts"] = state_copy["composite_starts"]
                state["composite_goals"] = state_copy["composite_goals"]
                state["agent_observations"] = state_copy["agent_observations"]
                state["agent_waypoints"] = state_copy["agent_waypoints"]

                state["agent_observations"][agent_id] = state["observation"]

                augmented_ep_reward_list[agent_id].append(reward)

                if done:
                    augmented_ep_observation_list[agent_id].append(
                        info["terminal_observation"]["observation"]
                    )
                    agent_done[agent_id] = True

            # Check if any of the agent's positions are within some threshold
            for agent_id in range(num_agents):
                for other_agent_id in range(num_agents):
                    if agent_id == other_agent_id:
                        continue
                    if (
                        np.linalg.norm(
                            np.array(state["agent_observations"][agent_id])
                            - np.array(state["agent_observations"][other_agent_id])
                        )
                        < threshold
                    ):
                        print("Collision!!!")

            all_done = all(agent_done)

        return (
            starts,
            goals,
            augmented_ep_observation_list,
            augmented_ep_waypoint_list,
            augmented_ep_reward_list,
        )
