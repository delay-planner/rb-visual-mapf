from __future__ import annotations

import numpy as np
from pud.collector import Collector
from typing import Union, List, Tuple
from pud.policies import BasePolicy, GaussianPolicy, SearchPolicy
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
    assert hasattr(eval_env, "pb_Q")
    """At the end of the last pb in the Q, the step will trigger another
    reset, and it should be safely handled by the reset_orig, suppress the warning
    message by turning off verbose"""
    bk_prob_constriant = eval_env.get_prob_constraint()
    eval_env.set_prob_constraint(1.0) # only use from the pb Q
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

    if n == 0:
        return records

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
            if collect_trajs and "terminal_observation" in info:
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
    eval_env.set_prob_constraint(bk_prob_constriant)
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
    def sample_initial_unconstrained_grid_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            obs = eval_env.sample_empty_state()
            s0 = eval_env.normalize_obs(obs)
            rb_vec.append(s0)
        rb_vec = np.array(rb_vec)
        return rb_vec

    @classmethod
    def sample_initial_unconstrained_visual_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            state = eval_env.reset()
            rb_vec.append(state)
        rb_vec_grid = np.array([x["grid"]["observation"] for x in rb_vec])
        rb_vec_visual = np.array([x["observation"] for x in rb_vec])
        return rb_vec_grid, rb_vec_visual

    @classmethod
    def sample_initial_unconstrained_states(cls, eval_env, num_states, habitat=False):
        if habitat:
            return cls.sample_initial_unconstrained_visual_states(eval_env, num_states)
        else:
            return cls.sample_initial_unconstrained_grid_states(eval_env, num_states)

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
    def get_grid_trajectory(cls, policy, eval_env, start=None, goal=None):
        ep_reward_list = []
        ep_waypoint_list = []
        ep_observation_list = []

        state, info = eval_env.reset()
        denormalize_factor = np.array([eval_env.unwrapped._height, eval_env.unwrapped._width], dtype=np.float32)

        if start is not None and goal is not None:
            state["goal"] = goal.copy()
            state["observation"] = start.copy()
            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goal * denormalize_factor
            eval_env.unwrapped.state = start * denormalize_factor

        ep_goal = state["goal"]
        ep_start = state["observation"]
        while True:
            ep_observation_list.append(state["observation"])

            # NOTE: state['goal'] may be modified
            action = policy.select_action(state)

            ep_waypoint_list.append(state["goal"])

            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info["terminal_observation"]["observation"])
                break

        return ep_start, ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list

    @classmethod
    def get_visual_trajectory(cls, policy, eval_env, start=None, goal=None):
        raise NotImplementedError

    @classmethod
    def get_trajectory(cls, policy, eval_env, input_start=None, input_goal=None, habitat=False):
        if habitat:
            if input_start is not None and input_goal is not None:
                assert len(input_start) == 2 and len(input_goal) == 2
            return cls.get_visual_trajectory(policy, eval_env, input_start, input_goal)
        else:
            return cls.get_grid_trajectory(policy, eval_env, input_start, input_goal)

    @classmethod
    def get_grid_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        starts=None,
        goals=None,
        threshold=0.05,
    ):
        augmented_ep_reward_list: List[List[int]] = [[] for _ in range(num_agents)]
        augmented_ep_waypoint_list = [[] for _ in range(num_agents)]
        augmented_ep_observation_list = [[] for _ in range(num_agents)]

        denormalize_factor = np.array([eval_env.unwrapped._height, eval_env.unwrapped._width], dtype=np.float32)

        state, info = eval_env.reset()

        if starts is not None and goals is not None:

            state["goal"] = goals.copy()
            state["observation"] = starts.copy()

            state["composite_goals"] = goals.copy()
            state["agent_waypoints"] = goals.copy()

            state["composite_starts"] = starts.copy()
            state["agent_observations"] = starts.copy()
        else:

            # Use the sampled start and goal for the first agent
            agent_goal = [state["goal"]]
            agent_start = [state["observation"]]

            # Mutable objects
            state["agent_waypoints"] = agent_goal.copy()
            state["agent_observations"] = agent_start.copy()

            goals = agent_goal.copy()
            starts = agent_start.copy()

            # Sample the starts and goals for the other agents
            for _ in range(num_agents - 1):

                agent_state, info = eval_env.reset()
                agent_goal = [agent_state["goal"]]
                agent_start = [agent_state["observation"]]

                # Add the new observations and goals to the state
                goals.extend(agent_goal.copy())
                starts.extend(agent_start.copy())
                state["agent_waypoints"].extend(agent_goal.copy())
                state["agent_observations"].extend(agent_start.copy())

            # Immutable objects - Should not change ever!
            state["composite_goals"] = goals.copy()
            state["composite_starts"] = starts.copy()
            print("Sampled the required starts and goals")

        all_done = False
        agent_done = [False for _ in range(num_agents)]

        while not all_done:

            state["goal"] = state["agent_waypoints"][0]
            state["observation"] = state["agent_observations"][0]

            if "goalconditioned" in type(eval_env.env).__name__.lower():
                eval_env.env._goal = goals[0] * denormalize_factor
            eval_env.unwrapped.state = state["observation"] * denormalize_factor

            # NOTE: state's agent_observations, agent_waypoints and goal are updated
            if isinstance(policy, BasePolicy):
                actions, agent_goals = policy.select_action(state)

            for agent_id in range(num_agents):

                if agent_done[agent_id]:
                    continue

                if isinstance(policy, BasePolicy):
                    state["agent_waypoints"][agent_id] = agent_goals[agent_id]

                current_agent_waypoint = state["agent_waypoints"][agent_id]
                current_agent_observation = state["agent_observations"][agent_id]

                augmented_ep_waypoint_list[agent_id].append(current_agent_waypoint)
                augmented_ep_observation_list[agent_id].append(current_agent_observation)

                state_copy = state.copy()

                state["goal"] = state["agent_waypoints"][agent_id]
                state["observation"] = state["agent_observations"][agent_id]

                action = actions[agent_id] if isinstance(policy, BasePolicy) else policy.select_action(state)

                if "goalconditioned" in type(eval_env.env).__name__.lower():
                    eval_env.env._goal = goals[agent_id] * denormalize_factor
                eval_env.unwrapped.state = state["observation"] * denormalize_factor

                state, reward, done, info = eval_env.step(np.copy(action), num_agents=num_agents)

                # At this point the state is changed and does not have the extra attributes so add them back
                state["composite_goals"] = state_copy["composite_goals"]
                state["composite_starts"] = state_copy["composite_starts"]

                state["agent_waypoints"] = state_copy["agent_waypoints"]
                state["agent_observations"] = state_copy["agent_observations"]

                # The agent's observations are updated based on the step function
                state["agent_observations"][agent_id] = state["observation"]

                augmented_ep_reward_list[agent_id].append(reward)

                if done:
                    terminal_agent_observation = info["terminal_observation"]["observation"]
                    augmented_ep_observation_list[agent_id].append(terminal_agent_observation)
                    agent_done[agent_id] = True

            # Check if any of the agent's positions are within some threshold
            for agent_id in range(num_agents):
                for other_agent_id in range(num_agents):
                    if agent_id == other_agent_id:
                        continue

                    agent_state = np.array(state["agent_observations"][agent_id])
                    other_agent_state = np.array(state["agent_observations"][other_agent_id])

                    if (np.linalg.norm(agent_state - other_agent_state) < threshold):
                        print(f"Agent {agent_id} is within threhsold of another agent {other_agent_id}")

            all_done = all(agent_done)

        return (
            starts,
            goals,
            augmented_ep_observation_list,
            augmented_ep_waypoint_list,
            augmented_ep_reward_list,
        )

    @classmethod
    def get_visual_trajectories(cls, policy, eval_env, num_agents, starts=None, goals=None, threshold=0.05):
        raise NotImplementedError

    @classmethod
    def get_trajectories(
        cls,
        policy,
        eval_env,
        num_agents,
        input_starts=None,
        input_goals=None,
        threshold=0.05,
        habitat=False,
    ):
        if habitat:
            if input_starts is not None and input_goals is not None:
                assert isinstance(input_starts, list) and isinstance(input_goals, list)
                assert len(input_starts[0]) == 2 and len(input_goals[0]) == 2

            return cls.get_visual_trajectories(
                policy, eval_env, num_agents, starts=input_starts, goals=input_goals, threshold=threshold
            )
        else:
            return cls.get_grid_trajectories(
                policy, eval_env, num_agents, starts=input_starts, goals=input_goals, threshold=threshold
            )
