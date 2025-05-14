import yaml
import torch
import rclpy
import logging
import argparse
import numpy as np
from pathlib import Path
from dotmap import DotMap
from rclpy.executors import MultiThreadedExecutor

from pud.algos.ddpg import GoalConditionedCritic
from pud.utils import set_env_seed, set_global_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.pb_sampler import load_pb_set
from pud.envs.simple_navigation_env import PointEnv, set_env_difficulty
from pud.algos.policies import ConstrainedMultiAgentSearchPolicy
from pud.buffers.constrained_buffer import ConstrainedReplayBuffer
from pud.collectors.constrained_collector import ConstrainedCollector as Collector
from pud.visualizers.visualize import visualize_compare_search, visualize_search_path
from pud.gzsim.src.rbmapf_gzsim.rbmapf_gzsim.multi_drone_control import DroneController
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    set_safe_env_difficulty,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointQueueWrapper,
    SafeGoalConditionedPointBlendWrapper,
)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--illustration_pb_file", type=str, default="")
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

    replay_buffer = ConstrainedReplayBuffer(obs_dim, goal_dim, action_dim, **config.replay_buffer)

    return config, eval_env, agent, replay_buffer


def denormalize(wp, height, width, z=2.0):
    print(f"Denormalizing: {wp}")
    ans = np.array([wp[0] * height, wp[1] * width], dtype=np.float32)
    print(f"Denormalized: {ans}")
    return np.array([wp[0] * height, wp[1] * width, z], dtype=np.float32)


def extract_walls(args):
    config, eval_env, agent, replay_buffer = pointenv_setup(args)
    assert isinstance(eval_env.unwrapped, PointEnv)

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore

    problems = load_pb_set(file_path=args.illustration_pb_file, env=eval_env, agent=agent)  # type: ignore
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
    state, _ = eval_env.reset()
    print("State: ", state)
    print(state)
    assert state is not None
    assert isinstance(state, dict)
    agent_goal = [state["goal"]]
    agent_start = [state["observation"]]

    goals = agent_goal.copy()
    starts = agent_start.copy()

    for _ in range(args.num_agents - 1):
        agent_state, _ = eval_env.reset()
        assert agent_state is not None
        assert isinstance(agent_state, dict)
        agent_goal = [agent_state["goal"]]
        agent_start = [agent_state["observation"]]

        goals.extend(agent_goal.copy())
        starts.extend(agent_start.copy())

    denormed_starts = [
        denormalize(start, eval_env.unwrapped._height, eval_env.unwrapped._width) for start in starts]
    return eval_env.get_map(), denormed_starts


def generate_wps(args, debug=False):

    config, eval_env, agent, replay_buffer = pointenv_setup(args)
    assert isinstance(eval_env.unwrapped, PointEnv)
    rb_vec = Collector.sample_initial_unconstrained_states(eval_env, replay_buffer.max_size)  # type: ignore
    agent.load_state_dict(
        torch.load(
            args.unconstrained_ckpt_file, map_location=torch.device(config.device)
        )
    )
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore
    agent.load_state_dict(
        torch.load(args.constrained_ckpt_file, map_location=torch.device(config.device))
    )
    constrained_pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore
    
    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["cost"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(60 * args.num_agents, 60),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    constrained_ma_search_policy = ConstrainedMultiAgentSearchPolicy(
        agent=agent,
        rb_vec=rb_vec,
        n_agents=args.num_agents,
        pdist=constrained_pdist,
        pcost=pcost,
        open_loop=True,
        cbs_config=cbs_config,
        max_cost_limit=np.inf,
        no_waypoint_hopping=True,
        ckpts={
            "unconstrained": args.unconstrained_ckpt_file,
            "constrained": args.constrained_ckpt_file,
        },
    )

    problems = load_pb_set(file_path=args.illustration_pb_file, env=eval_env, agent=agent)  # type: ignore
    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    constrained = constrained_ma_search_policy.constraints is not None
    if constrained:
        cost_constraints = constrained_ma_search_policy.constraints
        set_safe_env_difficulty(eval_env, 0.5, **cost_constraints)  # type: ignore
    else:
        set_env_difficulty(eval_env, 0.5)

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

        for _ in range(args.num_agents - 1):

            agent_state = eval_env.reset()
            assert agent_state is not None
            agent_state, _ = agent_state if constrained else (agent_state, None)

            assert isinstance(agent_state, dict)
            agent_goal = [agent_state["goal"]]
            agent_start = [agent_state["observation"]]

            goals.extend(agent_goal.copy())
            starts.extend(agent_start.copy())
            state["agent_waypoints"].append(agent_goal.copy())
            state["agent_observations"].extend(agent_start.copy())

        # Immutable objects - Should not be modified ever!
        state["composite_goals"] = goals.copy()
        state["composite_starts"] = starts.copy()
        print("Sampled the required starts and goals")

        constrained_ma_search_policy.select_action(state)
        waypoints = constrained_ma_search_policy.get_augmented_waypoints()

    wps = []
    for agent_id in range(args.num_agents):

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
        denormed = [denormalize(wp, eval_env.unwrapped._height, eval_env.unwrapped._width) for wp in waypoint_vec]
        wps.append(denormed)

    if debug:
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
        figdir = Path("log")
        figdir.mkdir(parents=True, exist_ok=True)
        visualize_search_path(
            constrained_ma_search_policy,
            eval_env,
            num_agents=args.num_agents,
            difficulty=0.9,
            outpath=figdir.joinpath("vis_constrained_multi_agent_search.jpg").as_posix()
        )
        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore
        visualize_compare_search(
            agent,
            constrained_ma_search_policy,
            eval_env,
            num_agents=args.num_agents,
            difficulty=0.9,
            outpath=figdir.joinpath("vis_compare_constrained_multi_agent.jpg").as_posix()
        )

    return wps


if __name__ == "__main__":

    args = argument_parser()
    wps = generate_wps(args)

    rclpy.init()
    drone_ids = [i + 1 for i in range(1, args.num_agents + 1)]
    drone_namespaces = [f"/px4_{i+1}" for i in range(args.num_agents)]
    drone_nodes = [DroneController(ns, nid) for ns, nid in zip(drone_namespaces, drone_ids)]
    for i, node in enumerate(drone_nodes):
        node.set_waypoints(wps[i][1:])  # Ignore the start point

    executor = MultiThreadedExecutor()
    for node in drone_nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in drone_nodes:
            node.destroy_node()
        rclpy.shutdown()
