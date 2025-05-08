import torch
import logging
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path


from pud.mapf.cbs import CBSSolver
from collect_safe_trajectory_records import (
    MAX_TIMELIMIT,
    TIMELIMIT,
    load_problem_set,
    load_agent_and_env,
    pointenv_setup,
    habitat_setup,
)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--visual", default=False, action="store_true")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")
    parser.add_argument("--unconstrained_ckpt_file", type=str, default="")
    parser.add_argument(
        "--traj_difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
    )
    args = parser.parse_args()
    return args


def try_problems(agent, eval_env, problem_setup, args, config, basedir):
    habitat = args.visual
    agent, eval_env = load_agent_and_env(
        agent, eval_env, args, config, constrained=True
    )

    save_path = basedir / args.traj_difficulty / "problems"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    valid_pbs = []
    save_path = save_path / f"pbs_{args.num_agents}.npy"

    if not habitat:
        max_search_steps = 7
        rb_vec = problem_setup[0].copy()
        pdist = problem_setup[2].copy()
        pcost = problem_setup[3].copy()
    else:
        max_search_steps = 4
        rb_vec_grid, rb_vec = (
            problem_setup[0].copy(),
            problem_setup[1].copy(),
        )
        pdist = problem_setup[3].copy()
        pcost = problem_setup[4].copy()

    ckpts = {
        "unconstrained": args.unconstrained_ckpt_file,
        "constrained": args.constrained_ckpt_file,
    }

    cbs_config = {
        "seed": None,
        "use_experience": True,
        "use_cardinality": True,
        "collision_radius": 0.0,
        "risk_attribute": "cost",
        "split_strategy": "disjoint",
        "edge_attributes": ["step"],
        "max_distance": eval_env.max_goal_dist,
        "max_time": min(TIMELIMIT * args.num_agents, MAX_TIMELIMIT),
        "tree_save_frequency": 1,
        "logdir": "pud/mapf/unit_tests/logs/cbs",
    }

    problems = problem_setup[-1].copy()
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    rb_graph = nx.Graph()
    pdist_combined = np.max(pdist, axis=0)
    pcost_combined = np.max(pcost, axis=0)

    for i, _ in enumerate(rb_vec):
        for j, _ in enumerate(rb_vec):
            cost = pcost_combined[i, j]
            length = pdist_combined[i, j]
            if length < max_search_steps:
                rb_graph.add_edge(
                    i, j, weight=float(length), step=1.0, cost=float(cost)
                )

    idx = -1
    with tqdm(total=args.num_samples) as pbar:
        while len(valid_pbs) < args.num_samples:
            idx += 1
            skip_idx = False

            pb_graph = rb_graph.copy()

            pbs = problems[idx * args.num_agents : (idx + 1) * args.num_agents]  # noqa
            goals = [pb["goal"] for pb in pbs]
            starts = [pb["start"] for pb in pbs]

            # Ensure that each start, goal pair is unique
            tuple_starts = [tuple(start) for start in starts]
            if len(set(tuple_starts)) != args.num_agents:
                logging.debug(f"Duplicate starts for problem {idx}")
                continue
            tuple_goals = [tuple(goal) for goal in goals]
            if len(set(tuple_goals)) != args.num_agents:
                logging.debug(f"Duplicate goals for problem {idx}")
                continue

            normalized_goals = [eval_env.normalize_obs(goal) for goal in goals]
            normalized_starts = [eval_env.normalize_obs(start) for start in starts]

            # Ensure that the start and goal can be connected to the replay buffer
            goal_ids = []
            start_ids = []
            num_nodes = rb_vec.shape[0] - 1

            for agent_id in range(args.num_agents):

                start_ids.append(num_nodes + 1)
                goal_ids.append(num_nodes + 2)
                state = {
                    "goal": normalized_goals[agent_id],
                    "observation": normalized_starts[agent_id],
                }

                start_to_rb_dist = agent.get_pairwise_dist(
                    [state["observation"]],
                    rb_vec,
                    aggregate="min",
                    max_search_steps=max_search_steps,
                    masked=True,
                )
                rb_to_goal_dist = agent.get_pairwise_dist(
                    rb_vec,
                    [state["goal"]],
                    aggregate="min",
                    max_search_steps=max_search_steps,
                    masked=True,
                )
                agent.load_state_dict(
                    torch.load(
                        ckpts["unconstrained"], map_location="cuda:0", weights_only=True
                    )
                )
                start_to_rb_cost = agent.get_pairwise_cost(
                    [state["observation"]],
                    rb_vec,
                    aggregate="max",
                )
                rb_to_goal_cost = agent.get_pairwise_cost(
                    rb_vec,
                    [state["goal"]],
                    aggregate="max",
                )
                agent.load_state_dict(
                    torch.load(
                        ckpts["constrained"], map_location="cuda:0", weights_only=True
                    )
                )

                for i, (from_start, to_goal) in enumerate(
                    zip(
                        zip(start_to_rb_dist.flatten(), start_to_rb_cost.flatten()),
                        zip(rb_to_goal_dist.flatten(), rb_to_goal_cost.flatten()),
                    )
                ):
                    dist_from_start, cost_from_start = from_start
                    dist_to_goal, cost_to_goal = to_goal
                    if dist_from_start < max_search_steps:
                        pb_graph.add_edge(
                            start_ids[agent_id],
                            i,
                            weight=float(dist_from_start),
                            step=1.0,
                            cost=float(cost_from_start),
                        )
                    if dist_to_goal < max_search_steps:
                        pb_graph.add_edge(
                            i,
                            goal_ids[agent_id],
                            weight=float(dist_to_goal),
                            step=1.0,
                            cost=float(cost_to_goal),
                        )

                if not np.any(start_to_rb_dist < max_search_steps) or not np.any(
                    rb_to_goal_dist < max_search_steps
                ):
                    logging.debug(
                        f"Failed to connect start or goal to the replay buffer for problem {idx}"
                    )
                    skip_idx = True
                    break

                num_nodes += 2

            if skip_idx:
                continue

            # Ensure that vanilla CBS can find a solution for the problem

            if not habitat:
                augmented_wps = np.concatenate([rb_vec, normalized_starts, normalized_goals], axis=0)
            else:
                augmented_wps = np.concatenate([rb_vec_grid, starts, goals], axis=0)

            cbs_solver = CBSSolver(
                graph=pb_graph,
                graph_waypoints=augmented_wps,
                starts=start_ids,
                goals=goal_ids,
                config=cbs_config,
            )

            try:
                cbs_solver.find_paths()
            except Exception as e:
                logging.debug(f"Failed to find path for problem {idx} because {e}")
                continue

            valid_pb = {
                "graph": pb_graph,
                "starts": starts,
                "goals": goals,
                "pbs": pbs,
            }
            valid_pbs.append(valid_pb)
            pbar.update(1)

    np.save(save_path, valid_pbs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = argument_parser()
    if args.visual:
        config, eval_env, agent, trained_cost_limit = habitat_setup(args)
    else:
        config, eval_env, agent, trained_cost_limit = pointenv_setup(args)

    basedir = Path("pud/plots/data")
    if not args.visual:
        basedir = basedir / config.env.walls.lower()
    else:
        basedir = basedir / config.env.simulator_settings.scene.lower()

    problem_setup = load_problem_set(
        args.problem_set_file, eval_env, agent, args.visual
    )
    try_problems(agent, eval_env, problem_setup, args, config, basedir)
