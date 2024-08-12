import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dotmap import DotMap

from pud.ddpg import GoalConditionedCritic
from pud.utils import set_global_seed, set_env_seed
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.algos.constrained_collector import ConstrainedCollector
from pud.envs.safe_pointenv.pb_sampler import load_pb_set, sample_cost_pbs_by_agent
from pud.policies import (
    SearchPolicy,
    ConstrainedSearchPolicy,
    MultiAgentSearchPolicy,
    ConstrainedMultiAgentSearchPolicy,
)
from pud.envs.safe_pointenv.safe_wrappers import (
    safe_env_load_fn,
    SafeGoalConditionedPointWrapper,
    SafeGoalConditionedPointBlendWrapper,
    SafeGoalConditionedPointQueueWrapper,
)


def setup(args):
    assert len(args.config_file) > 0
    assert len(args.constrained_ckpt_file) > 0

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # User defined parameters for evaluation
    trained_cost_limit = config.agent.cost_limit

    config.device = args.device
    config.num_samples = args.num_samples
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

    obs_dim = eval_env.observation_space['observation'].shape[0]  # type: ignore
    goal_dim = obs_dim
    state_dim = obs_dim + goal_dim
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    print(f'Obs dim: {obs_dim},\n'
          f'Goal dim: {goal_dim},\n'
          f'State dim: {state_dim},\n'
          f'Action dim: {action_dim},\n'
          f'Max action: {max_action}')

    agent = DRLDDPGLag(
            state_dim,  # Concatenating obs and goal
            action_dim,
            max_action,
            CriticCls=GoalConditionedCritic,
            device=torch.device(config.device),
            **config.agent,
        )

    agent.load_state_dict(torch.load(args.constrained_ckpt_file))
    agent.to(torch.device(config.device))
    agent.eval()

    return config, eval_env, agent, trained_cost_limit


def load_agent_and_env(agent, eval_env, args, config, constrained=False):
    if constrained:
        agent.load_state_dict(torch.load(args.constrained_ckpt_file))
    else:
        agent.load_state_dict(torch.load(args.unconstrained_ckpt_file))
    agent.to(torch.device(config.device))
    agent.eval()

    eval_env.duration = 300  # type: ignore
    eval_env.set_use_q(True)  # type: ignore
    eval_env.set_prob_constraint(1.0)  # type: ignore

    return agent, eval_env


def setup_problems(eval_env, agent, trained_cost_limit, args, config, save=False):

    rb_vec = ConstrainedCollector.sample_initial_unconstrained_states(eval_env, config.replay_buffer.max_size)
    pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)  # type: ignore
    pcost = agent.get_pairwise_cost(rb_vec, aggregate=None)  # type: ignore

    if len(args.illustration_pb_file) > 0:
        problems = load_pb_set(file_path=args.illustration_pb_file, env=eval_env, agent=agent)  # type: ignore
    else:
        problems = sample_cost_pbs_by_agent(
            K=config.num_samples * 100,
            min_dist=0,
            agent=agent,  # type: ignore
            env=eval_env,  # type: ignore
            target_val=trained_cost_limit,
            num_states=1000,
            ensemble_agg="mean",
            use_uncertainty=False,
            max_dist=eval_env.max_goal_dist,  # type: ignore
        )
        assert len(problems) > 0

    if save:
        np.savez("pud/plots/illustration.npz",
                 rb_vec=rb_vec,
                 pdist=pdist,
                 pcost=pcost,
                 problems=problems)  # type: ignore
    return rb_vec, pdist, pcost, problems


def load_problem_set(file_path, env, agent):
    load = np.load(file_path, allow_pickle=True)
    rb_vec = load["rb_vec"]
    pdist = load["pdist"]
    pcost = load["pcost"]
    problems = load["problems"]
    return rb_vec, pdist, pcost, problems.tolist()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--problem_set_file", type=str, default="")
    parser.add_argument("--illustration_pb_file", type=str, default="")
    parser.add_argument("--constrained_ckpt_file", type=str, default="")
    parser.add_argument("--replay_buffer_size", type=int, default="1000")
    parser.add_argument("--unconstrained_ckpt_file", type=str, default="")
    parser.add_argument("--load_problem_set", default=False, action="store_true")
    parser.add_argument("--method_type", type=str,
                        choices=["unconstrained", "unconstrained_search", "constrained", "constrained_search"],
                        default="unconstrained")

    args = parser.parse_args()
    return args


def single_unconstrained_policy(agent, eval_env, problem_setup, args, config, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=False)

    problems = problem_setup[3]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    unconstrained_records = []
    for _ in range(config.num_samples):
        _, _, _, _, _, records = ConstrainedCollector.get_trajectory(agent, eval_env)
        unconstrained_records.append(records)

    if save:
        np.save("pud/plots/data/single_agent/unconstrained_records.npy", unconstrained_records)
    return unconstrained_records


def multi_unconstrained_policy(agent, eval_env, problem_setup, args, config, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=False)

    problems = problem_setup[3]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    unconstrained_records = []
    for _ in range(config.num_samples):
        _, _, _, _, _, records = ConstrainedCollector.get_trajectories(agent, eval_env, args.num_agents)
        unconstrained_records.append(records)

    if save:
        np.save(f"pud/plots/data/multi_agent/unconstrained_records_{args.num_agents}.npy", unconstrained_records)
    return unconstrained_records


def single_unconstrained_search_policy(agent, eval_env, problem_setup, args, config, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=False)

    rb_vec, pdist, problems = problem_setup[0], problem_setup[1], problem_setup[3]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    pbar = tqdm(total=config.num_samples)
    unconstrained_search_records = []
    for _ in range(config.num_samples):
        search_policy = SearchPolicy(agent, rb_vec.copy(), pdist=pdist.copy(), open_loop=True, no_waypoint_hopping=True)
        _, _, _, _, _, records = ConstrainedCollector.get_trajectory(search_policy, eval_env)
        unconstrained_search_records.append(records)
        pbar.update(1)
    pbar.close()

    if save:
        np.save("pud/plots/data/single_agent/unconstrained_search_records.npy", unconstrained_search_records)
    return unconstrained_search_records


def multi_unconstrained_search_policy(agent, eval_env, problem_setup, args, config, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=False)

    rb_vec, pdist, problems = problem_setup[0], problem_setup[1], problem_setup[3]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    pbar = tqdm(total=config.num_samples)
    unconstrained_search_records = []
    for _ in range(config.num_samples):
        ma_search_policy = MultiAgentSearchPolicy(
            agent, rb_vec, args.num_agents, pdist=pdist, open_loop=True, no_waypoint_hopping=True
        )
        _, _, _, _, _, records = ConstrainedCollector.get_trajectories(ma_search_policy, eval_env, args.num_agents)
        unconstrained_search_records.append(records)
        pbar.update(1)
    pbar.close()

    if save:
        np.save(
            f"pud/plots/data/multi_agent/unconstrained_search_records_{args.num_agents}.npy",
            unconstrained_search_records
        )
    return unconstrained_search_records


def single_constrained_policy(agent, eval_env, problem_setup, args, config, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=True)

    problems = problem_setup[3]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    constrained_records = []
    for _ in range(config.num_samples):
        _, _, _, _, _, records = ConstrainedCollector.get_trajectory(agent, eval_env)
        constrained_records.append(records)

    if save:
        np.save("pud/plots/data/single_agent/constrained_records.npy", constrained_records)
    return constrained_records


def multi_constrained_policy(agent, eval_env, problem_setup, args, config, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=True)

    problems = problem_setup[3]
    eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

    constrained_records = []
    for _ in range(config.num_samples):
        _, _, _, _, _, records = ConstrainedCollector.get_trajectories(agent, eval_env, args.num_agents)
        constrained_records.append(records)

    if save:
        np.save(f"pud/plots/data/multi_agent/constrained_records_{args.num_agents}.npy", constrained_records)
    return constrained_records


def single_constrained_search_policy(agent, eval_env, problem_setup, args, config, trained_cost_limit, save=False):
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=True)

    rb_vec, pdist, pcost, problems = problem_setup
    eval_env.set_prob_constraint(1.0)  # type: ignore

    constrained_search_factored_records = []
    edge_cost_limit_factors = [0.1, 0.5, 1.0]
    for factor in edge_cost_limit_factors:
        print(f"Factor: {factor}")

        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        pbar = tqdm(total=config.num_samples)
        constrained_search_records = []
        edge_cost_limit = trained_cost_limit * factor

        for _ in range(config.num_samples):
            constrained_search_policy = ConstrainedSearchPolicy(
                agent,
                rb_vec,
                pdist=pdist,
                pcost=pcost,
                open_loop=True,
                no_waypoint_hopping=True,
                max_cost_limit=edge_cost_limit
            )
            _, _, _, _, _, records = ConstrainedCollector.get_trajectory(constrained_search_policy, eval_env)
            constrained_search_records.append(records)
            pbar.update(1)
        pbar.close()

        if save:
            np.save(f"pud/plots/data/single_agent/constrained_search_records_{factor}.npy", constrained_search_records)

        constrained_search_factored_records.append(constrained_search_records)

    if save:
        np.save(
            "pud/plots/data/single_agent/constrained_search_factored_records.npy", constrained_search_factored_records
        )
    return constrained_search_factored_records


def multi_constrained_search_policy(agent, eval_env, problem_setup, args, config, trained_cost_limit, save=False):
    rb_vec, pdist, pcost, problems = problem_setup
    agent, eval_env = load_agent_and_env(agent, eval_env, args, config, constrained=True)

    constrained_search_factored_records = []
    edge_cost_limit_factors = [0.1, 0.5, 1.0]
    for factor in edge_cost_limit_factors:
        print(f"Factor: {factor}")

        eval_env.set_pbs(pb_list=problems.copy())  # type: ignore

        pbar = tqdm(total=config.num_samples)
        constrained_search_records = []
        edge_cost_limit = trained_cost_limit * factor

        for _ in range(config.num_samples):
            constrained_ma_search_policy = ConstrainedMultiAgentSearchPolicy(
                agent,
                rb_vec,
                args.num_agents,
                pdist=pdist,
                pcost=pcost,
                open_loop=True,
                no_waypoint_hopping=True,
                max_cost_limit=edge_cost_limit
            )
            _, _, _, _, _, records = ConstrainedCollector.get_trajectories(
                constrained_ma_search_policy, eval_env, args.num_agents
            )
            constrained_search_records.append(records)
            pbar.update(1)
        pbar.close()

        if save:
            np.save(
                f"pud/plots/data/multi_agent/constrained_search_records_{args.num_agents}_{factor}.npy",
                constrained_search_records
            )

        constrained_search_factored_records.append(constrained_search_records)

    if save:
        np.save(
            f"pud/plots/data/multi_agent/constrained_search_factored_records_{args.num_agents}.npy",
            constrained_search_factored_records
        )
    return constrained_search_factored_records


if __name__ == "__main__":

    args = argument_parser()
    config, eval_env, agent, trained_cost_limit = setup(args)
    if args.load_problem_set:
        assert len(args.problem_set_file) > 0
        problem_setup = load_problem_set(args.problem_set_file, eval_env, agent)
    else:
        problem_setup = setup_problems(eval_env, agent, trained_cost_limit, args, config, save=True)

    if args.method_type == "unconstrained":
        if args.num_agents == 1:
            single_unconstrained_policy(agent, eval_env, problem_setup, args, config, save=True)
        else:
            multi_unconstrained_policy(agent, eval_env, problem_setup, args, config, save=True)
    elif args.method_type == "unconstrained_search":
        if args.num_agents == 1:
            single_unconstrained_search_policy(agent, eval_env, problem_setup, args, config, save=True)
        else:
            multi_unconstrained_search_policy(agent, eval_env, problem_setup, args, config, save=True)
    elif args.method_type == "constrained":
        if args.num_agents == 1:
            single_constrained_policy(agent, eval_env, problem_setup, args, config, save=True)
        else:
            multi_constrained_policy(agent, eval_env, problem_setup, args, config, save=True)
    elif args.method_type == "constrained_search":
        if args.num_agents == 1:
            single_constrained_search_policy(
                agent, eval_env, problem_setup, args, config, trained_cost_limit, save=True
            )
        else:
            multi_constrained_search_policy(agent, eval_env, problem_setup, args, config, trained_cost_limit, save=True)
    else:
        raise ValueError("Invalid method type")
