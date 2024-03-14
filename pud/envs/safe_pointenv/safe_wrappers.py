import gym
import numpy as np
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv
from pud.envs.wrappers import TimeLimit
from dotmap import DotMap
from typing import Union


class SafeGoalConditionedPointWrapper(gym.Wrapper):
    """Wrapper that appends goal to observation produced by environment.
    Sample with safety constraints

    Option 1: sample goals of any distance subject to step cost == 0
        Potential long distances, but there exists viable solutions
    Option 2:  distance constraints subject to step cost == 0
        Distance constraints, but there exists viable solutions
    Option 3: distance constraints subject to 0< step cost < cost limit
        perhaps only good for training
        Distance constraints, but there may not exist viable solutions, but trajectories whose step cost < cost limit 
    """

    def __init__(self, 
                env:SafePointEnv, 
                prob_constraint:float=0.8,
                min_dist=0, 
                max_dist=4,
                threshold_distance=1.0
                ):
        """Initialize the environment.

        Args:
          env: an environment.
          prob_constraint: (float) Probability that the distance constraint is
            followed after resetting.
          min_dist: (float) When the constraint is enforced, ensure the goal is at
            least this far from the initial observation.
          max_dist: (float) When the constraint is enforced, ensure the goal is at
            most this far from the initial observation.
          threshold_distance: (float) States are considered equivalent if they are
            at most this far away from one another.
        """
        self._threshold_distance = threshold_distance
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._sample_key = "ub"
        super(SafeGoalConditionedPointWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Dict({
            'observation': env.observation_space,
            'goal': env.observation_space,
        })

        # setup initial and goal state sampling
        self.start_n_goal_candidates = {}
        mask_finite = self.env._safe_apsp["ub"] < np.inf
        mask_no_loop = self.env._safe_apsp["ub"] > 0
        mask_cands = mask_finite * mask_no_loop
        inds_cands = np.where(mask_cands)
        self.start_n_goal_candidates["ub"] = np.column_stack(inds_cands) # x1, y1, x2, y2
        
        mask_finite = self.env._safe_apsp["lb"] < np.inf
        mask_no_loop = self.env._safe_apsp["lb"] > 0
        mask_cands = mask_finite * mask_no_loop
        inds_cands = np.where(mask_cands)
        self.start_n_goal_candidates["lb"] = np.column_stack(inds_cands) # x1, y1, x2, y2

        #self.safe_empty_states_ub = self.env.gather_safe_empty_states(cost_limit=0.0)
        #self.safe_empty_states_lb = self.env.gather_safe_empty_states(cost_limit=self.env.cost_limit)

        self.env:SafePointEnv

    def sample_start_n_goal(self, key:Union[tuple, str]="ub"):
        """sample start and goal without distance constraint, 
        "ub": viable solutions are guaranteed to exist
        "lb": viable solutions are NOT guarantee to exist
        """
        ind = np.random.randint(len(self.start_n_goal_candidates[key]))
        self.start_n_goal_candidates[key][ind]
        
        start = self.start_n_goal_candidates[key][ind, 0:2]
        goal = self.start_n_goal_candidates[key][ind, 2:]
        return {
            "s0": start,
            "sg": goal,
        }

    def sample_safe_start_n_goal_in_dists(self, min_dist:float, max_dist:float, key="ub"):
        """
        sampling start and goal states with guaranteed solution
        with distance cosntraints
        """
        cand_key = (key, min_dist, max_dist)
        if not (cand_key in self.start_n_goal_candidates.keys()):
            mask_lb = self.env._safe_apsp[key] > min_dist
            mask_ub = self.env._safe_apsp[key] < max_dist
            mask_cands = mask_lb * mask_ub
            inds_cands = np.where(mask_cands)
            assert len(inds_cands[0]) > 0, "candidate set is empty"
            self.start_n_goal_candidates[cand_key] = np.column_stack(inds_cands) # x1, y1, x2, y2
        return self.sample_start_n_goal(key=cand_key)
        

    def _normalize_obs(self, obs):
        return np.array([
            obs[0] / float(self.env._height),
            obs[1] / float(self.env._width)
        ])

    def reset(self):
        out = None
        if np.random.random() < self._prob_constraint:
            out = self.sample_safe_start_n_goal_in_dists(self._min_dist, self._max_dist, key = self._sample_key)
        else:
            out = self.sample_start_n_goal(key=self._sample_key)
        
        self._goal = out["sg"]
        obs = out["s0"]
        self.state = obs.copy()
        cost = self.env.get_state_cost(self._goal)

        new_state = {
            'observation': self._normalize_obs(obs),
            'goal': self._normalize_obs(self._goal), 
            }
        info = {"cost": cost}

        return new_state, info

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        rew = -1.0
        done = self._is_done(obs, self._goal)
        return {'observation': self._normalize_obs(obs),
                'goal': self._normalize_obs(self._goal)}, rew, done, info

    def set_sample_goal_args(self, 
            prob_constraint=None,
            min_dist=None, 
            max_dist=None,
            sample_key="ub",
            ):
        assert prob_constraint is not None
        assert min_dist is not None
        assert max_dist is not None
        assert min_dist >= 0
        assert max_dist >= min_dist
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._sample_key = sample_key

    def _is_done(self, obs, goal):
        """Determines whether observation equals goal."""
        return np.linalg.norm(obs - goal) < self._threshold_distance
    
    @property
    def max_goal_dist(self):
        apsp = self.env._safe_apsp["ub"]
        return np.max(apsp[np.isfinite(apsp)])

class SafeTimeLimit (TimeLimit):
    def __init__(self, env, duration, terminate_on_timeout=False):
        super(SafeTimeLimit, self).__init__(
            env=env, 
            duration=duration, 
            terminate_on_timeout=terminate_on_timeout,
            )

    def reset(self):
        """reset adds a info dict"""
        self.step_count = 0
        observation, info = self.env.reset()
        observation['first_step'] = True
        return observation, info

def safe_env_load_fn(env_kwargs:dict,
                cost_f_kwargs:dict,
                max_episode_steps=None,
                gym_env_wrappers=(SafeGoalConditionedPointWrapper,),
                terminate_on_timeout=False,
                ):
    """Loads the selected environment and wraps it with the specified wrappers.

    Args:
      environment_name: Name for the environment to load.
      max_episode_steps: If None the max_episode_steps will be set to the default
        step limit defined in the environment's spec. No limit is applied if set
        to 0 or if there is no timestep_limit set in the environment's spec.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.
      terminate_on_timeout: Whether to set done = True when the max episode
        steps is reached.

    Returns:
      An environment instance.
    """
    env = SafePointEnv(
                    **env_kwargs, 
                    cost_f_args=cost_f_kwargs
                )

    for wrapper in gym_env_wrappers:
        env = wrapper(env)

    if max_episode_steps > 0:
        env = SafeTimeLimit(env, max_episode_steps, terminate_on_timeout=terminate_on_timeout)

    return env