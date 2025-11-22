import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
from copy import deepcopy


from pud.algos.ddpg import GoalConditionedCritic
from pud.collectors.constrained_collector import ConstrainedCollector
from pud.utils import set_env_seed, set_global_seed
from pud.envs.simple_navigation_env import PointEnv
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.pb_sampler import load_pb_set
from pud.algos.policies import ConstrainedMultiAgentSearchPolicy
from pud.visualizers.visualize import visualize_compare_search, visualize_search_path
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
)

TIMELIMIT = 60
MAX_TIMELIMIT = 600
COLLISION_THRESHOLD = 1e-3

PCOST_INDEX = 3
PROBLEM_INDEX = 4
REPLAY_BUFFER_INDEX = 0
CONSTRAINED_PDIST_INDEX = 2
UNCONSTRAINED_PDIST_INDEX = 1


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--team_size", type=int, default=1)
    parser.add_argument("--visual", type=str, default="False")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--use_hardware", type=str, default="False")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")
    parser.add_argument("--unconstrained_ckpt_file", type=str, default="")
    parser.add_argument("--sdf_path", type=str, default="pud/gzsim/src/rbmapf_gzsim/models/default.sdf")
    args, _ = parser.parse_known_args()
    return args


def pointenv_setup(args):
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    config.device = args.device
    config.replay_buffer.max_size = args.replay_buffer_size

    set_global_seed(config.seed)

    gym_env_wrappers = []
    gym_env_wrapper_kwargs = []
    for wrapper_name in config.wrappers:
        if wrapper_name == "SafeGoalConditionedPointWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointBlendWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointBlendWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())
        elif wrapper_name == "SafeGoalConditionedPointQueueWrapper":
            gym_env_wrappers.append(SafeGoalConditionedPointQueueWrapper)
            gym_env_wrapper_kwargs.append(config.wrappers[wrapper_name].toDict())

    eval_env = safe_env_load_fn(
        config.env.toDict(),
        config.cost_function.toDict(),
        max_episode_steps=config.time_limit.max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        wrapper_kwargs=gym_env_wrapper_kwargs,
        terminate_on_timeout=True,
    )

    set_env_seed(eval_env, config.seed + 2)

    obs_dim = eval_env.observation_space["observation"].shape[0]  # type: ignore
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = eval_env.action_space.shape[0]  # type: ignore
    max_action = float(eval_env.action_space.high[0])  # type: ignore
    logging.debug(
        f"Obs dim: {obs_dim},\n"
        f"Goal dim: {goal_dim},\n"
        f"State dim: {state_dim},\n"
        f"Action dim: {action_dim},\n"
        f"Max action: {max_action}"
    )

    agent = DRLDDPGLag(
        state_dim,  # Concatenating obs and goal
        action_dim,
        max_action,
        CriticCls=GoalConditionedCritic,
        device=torch.device(config.device),
        **config.agent,
    )

    agent.load_state_dict(
        torch.load(args.constrained_ckpt_file, map_location=torch.device(config.device))
    )
    agent.to(torch.device(config.device))
    agent.eval()

    return config, eval_env, agent

def denormalize(wp, height, width, z=2.0):
    print(f"Denormalizing: {wp}")
    ans = np.array([wp[0] * height, wp[1] * width], dtype=np.float32)
    print(f"Denormalized: {ans}")
    return np.array([*ans, z], dtype=np.float32)


def load_problem_set(file_path):
    load = np.load(file_path, allow_pickle=True)
    rb_vec = load["rb_vec"]
    unconstrained_pdist = load["unconstrained_pdist"]
    constrained_pdist = load["constrained_pdist"]
    pcost = load["pcost"]
    problems = load["problems"]
    return rb_vec, unconstrained_pdist, constrained_pdist, pcost, problems.tolist()


def extract_walls(args):
    _, eval_env, agent = pointenv_setup(args)
    assert isinstance(eval_env.unwrapped, PointEnv)

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore

    cols, rows = eval_env.get_map().shape
    normalizing_factor = np.ones(2)
    use_hardware = args.use_hardware == 'True'

    suffix = args.problem_set_file.split('.')[-1]
    if suffix == 'npz':
        problem_setup = load_problem_set(args.problem_set_file)
        problems = problem_setup[PROBLEM_INDEX].copy()
    else:
        problems = load_pb_set(args.problem_set_file, env=eval_env, agent=agent)  # type: ignore

    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    state, _ = eval_env.reset()  # type: ignore
    assert state is not None
    assert isinstance(state, dict)
    agent_goal = [state["goal"]]
    agent_start = [state["observation"]]

    goals = agent_goal.copy()
    starts = agent_start.copy()

    for _ in range(args.num_agents - 1):
        agent_state, _ = eval_env.reset()  # type: ignore
        assert agent_state is not None
        assert isinstance(agent_state, dict)
        agent_goal = [agent_state["goal"]]
        agent_start = [agent_state["observation"]]

        goals.extend(agent_goal.copy())
        starts.extend(agent_start.copy())

    denormed_starts = [
        denormalize(start / normalizing_factor, cols, rows) for start in starts]
    denormed_adjusted_starts = [
        adjust_positions(start, eval_env, use_hardware) for start in denormed_starts
    ]

    return eval_env.get_map(), denormed_adjusted_starts


def adjust_positions(position, env, hardware=False):
    walls = env.get_map()
    rows, cols = walls.shape
    origin = np.array([-cols / 2.0, -rows / 2.0, 0.0])
    position += origin
    if hardware:
        print("Using hardware adjustments")
        position /= np.array([cols / 6., rows / 8., 1.0])
    return position

def generate_wps(args, problem_start=0, recovery=list(), debug=False):

    config, eval_env, agent = pointenv_setup(args)

    suffix = args.problem_set_file.split('.')[-1]
    if suffix == 'npz':
        problem_setup = load_problem_set(args.problem_set_file)
        rb_vec = deepcopy(problem_setup[REPLAY_BUFFER_INDEX])
        pdist = problem_setup[UNCONSTRAINED_PDIST_INDEX].copy()
        problems = problem_setup[PROBLEM_INDEX].copy()
    else:
        rb_vec = ConstrainedCollector.sample_initial_unconstrained_states(eval_env, config.replay_buffer.max_size)
        agent.load_state_dict(torch.load(args.unconstrained_ckpt_file))
        agent.to(torch.device(config.device))
        agent.eval()
        pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore

        agent.load_state_dict(torch.load(args.constrained_ckpt_file))
        problems = load_pb_set(args.problem_set_file, env=eval_env, agent=agent)  # type: ignore

    # Only needed for multiple mission problems
    # Modify the problems object to ensure start of the current problem matches the goal of the previous problem
    if problem_start > 0:
        # TODO: make sure the range is correct
        for idx in range(problem_start * args.num_agents, (problem_start + 1) * args.num_agents):
            prev_idx = idx - args.num_agents
            problems[idx]['start'] = problems[prev_idx]['goal']
    problems = problems[problem_start * args.team_size:]

    # Recovery should be a list of dictionaries with 'start' and 'goal' keys
    if len(recovery) > 0:
        # Inject the recovery states into the problems at the beginning
        for idx, recov in enumerate(recovery):
            recov_object = {'start': np.array(recov['start']), 'goal': np.array(recov['goal'])}
            problems.insert(idx, recov_object)

    assert isinstance(eval_env.unwrapped, PointEnv)

    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["step", "cost"],
        "max_distance": eval_env.max_goal_dist,  # type: ignore
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "budget_allocater": "uniform",
        "risk_bound": 10.0, # np.inf,  # 6.615,  # Low is 1.47 and High is 11.76
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }
    constrained_ma_search_policy = ConstrainedMultiAgentSearchPolicy(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.team_size,
        pdist=pdist,
        pcost=pcost,
        open_loop=True,
        cbs_config=cbs_config,
        max_cost_limit=np.inf,
        max_search_steps=7,
        no_waypoint_hopping=True,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    constrained = constrained_ma_search_policy.constraints is not None

    if constrained_ma_search_policy.open_loop:

        state = eval_env.reset()

        assert state is not None
        state, _ = state if constrained else (state, None)

        assert isinstance(state, dict)
        agent_goal = [state["goal"]]
        agent_start = [state["observation"]]

        # Mutable objects
        state["agent_waypoints"] = agent_goal.copy()
        state["agent_observations"] = agent_start.copy()

        goals = agent_goal.copy()
        starts = agent_start.copy()

        for _ in range(args.team_size - 1):

            agent_state = eval_env.reset()
            assert agent_state is not None
            agent_state, _ = agent_state if constrained else (agent_state, None)

            assert isinstance(agent_state, dict)
            agent_goal = [agent_state["goal"]]
            agent_start = [agent_state["observation"]]

            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].extend(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not be modified ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        constrained_ma_search_policy.select_action(state)
        waypoints = constrained_ma_search_policy.get_augmented_waypoints()

    wps = []
    original_wps = []
    cols, rows = eval_env.get_map().shape
    use_hardware = args.use_hardware == 'True'

    for agent_id in range(args.team_size):

        agent_goal = goals[agent_id]
        agent_start = starts[agent_id]
        waypoint_vec = np.array(waypoints[agent_id])

        print(f"Agent: {agent_id}")
        print(f"Start: {agent_start}")
        print(f"Waypoints: {waypoint_vec}")
        print(f"Goal: {agent_goal}")
        print(f"Steps: {waypoint_vec.shape[0]}")
        print("-" * 10)

        waypoint_vec = np.array([agent_start, *waypoint_vec, agent_goal])
        denormed = [denormalize(wp, cols, rows) for wp in waypoint_vec]
        original_wps.append(deepcopy(denormed))
        denormed_adjusted = [adjust_positions(wp, eval_env, use_hardware) for wp in denormed]
        wps.append(denormed_adjusted)

    if debug:
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
        figdir = Path("log")
        figdir.mkdir(parents=True, exist_ok=True)
        visualize_search_path(
            constrained_ma_search_policy,
            eval_env,
            num_agents=args.team_size,
            difficulty=0.9,
            outpath=figdir.joinpath("vis_constrained_multi_agent_search.jpg").as_posix()
        )
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
        visualize_compare_search(
            agent,
            constrained_ma_search_policy,
            eval_env,
            num_agents=args.team_size,
            difficulty=0.9,
            outpath=figdir.joinpath("vis_compare_constrained_multi_agent.jpg").as_posix()
        )

    # Returned wps are denormalized and shifted to match the origin of simulation/hardware environment
    print("Wps are: ", wps)
    print("Original Wps are: ", original_wps)
    return wps, original_wps
