import time
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
from dotmap import DotMap
from tqdm.auto import tqdm
from pud.vision_agent import LagVisionUVFDDPG
from pud.envs.safe_habitatenv.safe_habitat_wrappers import (
    SafeGoalConditionedHabitatPointQueueWrapper, SafeTimeLimit)
from pud.envs.simple_navigation_env import set_env_difficulty
from pud.algos.visual_collector import ConstrainedVisualCollector, eval_agent_from_Q
from pud.algos.visual_buffer import ConstrainedVisualReplayBuffer
from pud.policies import GaussianPolicy
from pud.envs.habitat_navigation_env import plot_wall, plot_traj, plot_start_n_goals
from pud.envs.safe_pointenv.pb_sampler import (sample_cost_pbs_by_agent, 
    sample_pbs_by_agent, load_pb_set)


# generated according to https://medialab.github.io/iwanthue/
distinct_colors=np.array([[190,121,68],
                            [176,86,193],
                            [104,184,84],
                            [105,125,204],
                            [190,168,66],
                            [191,113,172],
                            [98,128,63],
                            [201,80,113],
                            [77,184,175],
                            [211,82,56]],dtype=float)/255


def log_time(step:int=0, log:dict=None):
    if log is None:
        log = {
            "time": [time.time()],
            "step": [step],
            "speed": [],
        }
        return log
    
    log["time"].append(time.time())
    log["step"].append(step)
    log["speed"].append(
        float(log["step"][-1]-log["step"][-2])/(float(log["time"][-1])-float(log["time"][-2]))
    )
    return log

def visualize_visual_eval_records(eval_records, 
        eval_env, ax:plt.axes, starts=[], goals=[], use_pbar=False, color=None, 
        normalize_map=False,
        ):
    """
    for habitat env
    """
    
    list_trajs = []
    for id in eval_records.keys():
        list_trajs.append(eval_records[id]["traj"])

    starts = np.stack(starts)
    goals = np.stack(goals)

    ax = plot_wall(eval_env.walls.copy(), ax, normalize=normalize_map)

    for i in eval_records.keys():
        ax = plot_traj(
            traj=np.stack(eval_records[0]["traj"]), walls=eval_env.walls.copy(), 
            normalize=normalize_map,
            ax=ax,
            color=distinct_colors[i],
            label="traj{:0>2d}".format(i),
            marker="o",
            markersize=4,
                    )

        ax.scatter(
            goals[i:i+1,0],
            goals[i:i+1,1], 
            marker="*", s=12, color=distinct_colors[i],
            )

    return ax


def train_eval(
    policy:GaussianPolicy,
    agent:LagVisionUVFDDPG,
    replay_buffer: ConstrainedVisualReplayBuffer,
    env: SafeGoalConditionedHabitatPointQueueWrapper,
    eval_env: SafeGoalConditionedHabitatPointQueueWrapper,
    num_iterations=int(1e6),
    initial_collect_steps:int=1000,
    collect_steps:int=1,
    opt_steps:int=1,
    batch_size_opt:int=64,
    eval_func=lambda agent, eval_env: None,
    opt_log_interval:int=100,
    eval_interval:int=10000,
    verbose:bool=True,
    pbar:bool=False,
    logger:dict={}
):
    env.set_verbose(False)
    env.set_use_q(True)
    # train cost critic but not penalize unsafe actions
    agent.set_lag_status(turn_on_lag=False)


    time_logs = log_time(step=0)
    collector = ConstrainedVisualCollector(
        policy, replay_buffer, env, initial_collect_steps=initial_collect_steps
    )
    collector.step(collector.initial_collect_steps)

    for i in tqdm(range(1, num_iterations + 1), total=num_iterations, disable=not pbar):

        logger["i"] = i
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(
            replay_buffer, iterations=opt_steps, batch_size=batch_size_opt
        )

        if i % opt_log_interval == 0:
            if verbose:
                print(f"iteration = {i}, opt_info = {opt_info}")

        if i % eval_interval == 0:
            if "ckpt" in logger and isinstance(logger["ckpt"], Path):
                torch.save(agent.state_dict(), logger["ckpt"].joinpath("ckpt_{:0>7d}".format(i)))

            agent.eval()
            if verbose:
                print(f"evaluating iteration = {i}")
            eval_info = eval_func(agent, eval_env, logger=logger,)
            if verbose:
                print("-" * 10)

        if "tb" in logger:
            if i>1 and i % opt_log_interval == 0:
                logger["tb"].add_scalar(
                    "Opt/actor_loss", np.mean(opt_info["actor_loss"]), global_step=i
                )
                logger["tb"].add_scalar(
                    "Opt/critic_loss", np.mean(opt_info["critic_loss"]), global_step=i
                )
            
            if i > 1 and i % eval_interval == 0:
                field_header = "Eval Dist ~ "
                for d_ref in eval_info:
                    logger["tb"].add_scalars(field_header+"{:0>2d}/mean".format(d_ref),
                    tag_scalar_dict={
                        "pred": np.mean(eval_info[d_ref]["pred_dist"]),
                        "val": -np.mean(eval_info[d_ref]["returns"]),
                        }, global_step=i)

                    logger["tb"].add_scalars(field_header+"{:0>2d}/std".format(d_ref),
                    tag_scalar_dict={
                        "pred": np.std(eval_info[d_ref]["pred_dist"]),
                        "val": -np.std(eval_info[d_ref]["returns"]),
                        }, global_step=i)

                    N_success = np.array(eval_info[d_ref]["success"], dtype=float)
                    if len(N_success) > 0:
                        success_rate = np.sum(N_success) / len(N_success)
                        logger["tb"].add_scalar(field_header+"{:0>2d}/success_rate".format(d_ref), 
                                success_rate, global_step=i)

                time_logs = log_time(step=i, log=time_logs)
                logger["tb"].add_scalar(
                    "Time/Iters per Seconds", time_logs["speed"][-1], global_step=i
                )
                logger["tb"].add_scalar(
                    "Time/Total Time", time_logs["time"][-1], global_step=i
                )

def eval_pointenv_dists(
    agent, 
    eval_env:SafeGoalConditionedHabitatPointQueueWrapper, 
    num_evals=10, 
    sample_size:int=100,
    eval_distances=[1,2,3,4], 
    verbose=True, 
    logger:dict={},
):
    eval_info = DotMap()
    if "eval_distances" in logger:
        eval_distances = logger["eval_distances"]
    
    eval_img_dir: Path = None
    if "imgs" in logger:
        eval_img_dir = logger["imgs"].joinpath("eval_{:0>5d}".format(logger["i"]))
        eval_img_dir.mkdir(exist_ok=True, parents=True)
    
    for ii_d in range(len(eval_distances)):
        pbs = sample_pbs_by_agent(env=eval_env, 
                agent=agent, 
                num_states=sample_size,
                target_val=eval_distances[ii_d],
                K=num_evals,
                min_dist=0,
                max_dist=20,
                use_uncertainty=False,
                ensemble_agg="mean",
                )
        if len(pbs) > 0:
            eval_env.append_pbs(pb_list=pbs)
            dist_eval_i = eval_agent_from_Q(
                            policy=agent, 
                            eval_env=eval_env, 
                            collect_trajs=not (eval_img_dir is None),
                            )
            if eval_img_dir:
                fig, ax = plt.subplots()
                start_list = [p["start"].tolist() for p in pbs]
                goal_list = [p["goal"].tolist() for p in pbs]
                visualize_visual_eval_records(
                    eval_records=dist_eval_i,
                    eval_env=eval_env,
                    ax=ax,
                    starts=start_list,
                    goals=goal_list,
                )
        
                fig.savefig(eval_img_dir.joinpath("dist={}".format(eval_distances[ii_d])), dpi=300)
                #fig.savefig(Path("temp").joinpath("eval_{:0>5d}_dist={}".format(logger["i"], dist)), dpi=300)

                plt.close(fig=fig)
    return eval_info


def eval_search_policy(search_policy, eval_env, num_evals=10, constrained=False):
    eval_start = time.perf_counter()

    successes = 0.0
    for _ in range(num_evals):
        try:
            if constrained:
                _, _, _, _, ep_reward_list, _ = ConstrainedCollector.get_trajectory(
                    search_policy, eval_env
                )
            else:
                _, _, _, _, ep_reward_list, _ = Collector.get_trajectory(
                    search_policy, eval_env
                )
            successes += int(len(ep_reward_list) < eval_env.duration)
        except Exception:
            pass

    eval_end = time.perf_counter()
    eval_time = eval_end - eval_start
    success_rate = successes / num_evals
    return success_rate, eval_time


def take_cleanup_steps(
    search_policy,
    eval_env,
    num_cleanup_steps,
    cost_constraints: dict = {},
    constrained=False,
):
    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, 0.95, **cost_constraints)
    else:
        set_env_difficulty(eval_env, 0.95)

    search_policy.set_cleanup(True)
    cleanup_start = time.perf_counter()
    # Collector.eval_agent(search_policy, eval_env, num_cleanup_steps, by_episode=False) # random goals in env
    if constrained:
        ConstrainedCollector.step_cleanup(search_policy, eval_env, num_cleanup_steps)
    else:
        Collector.step_cleanup(
            search_policy, eval_env, num_cleanup_steps
        )  # samples goals from nodes in state graph
    cleanup_end = time.perf_counter()
    search_policy.set_cleanup(False)
    cleanup_time = cleanup_end - cleanup_start
    return cleanup_time


def cleanup_and_eval_search_policy(
    search_policy,
    eval_env,
    num_evals=10,
    difficulty=0.5,
    cost_constraints: dict = {},
    constrained=False,
):

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(
        search_policy, eval_env, num_evals=num_evals, constrained=constrained
    )

    # Initial sparse graph
    print(
        f"Initial {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds"
    )
    initial_g, initial_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    # Filter search policy
    search_policy.filter_keep_k_nearest()

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(
        search_policy, eval_env, num_evals=num_evals, constrained=constrained
    )
    print(
        f"Filtered {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds"
    )
    filtered_g, filtered_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    # Cleanup steps
    num_cleanup_steps = int(1e4)
    cleanup_time = take_cleanup_steps(
        search_policy, eval_env, num_cleanup_steps, constrained=constrained
    )
    print(f"Took {num_cleanup_steps} cleanup steps in {cleanup_time:.2f} seconds")

    if isinstance(eval_env, SafeTimeLimit) or isinstance(
        eval_env, SafeGoalConditionedPointWrapper
    ):
        set_safe_env_difficulty(eval_env, difficulty, **cost_constraints)
    else:
        set_env_difficulty(eval_env, difficulty)

    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(
        search_policy, eval_env, num_evals=num_evals, constrained=constrained
    )
    print(
        f"Cleaned {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds"
    )
    cleaned_g, cleaned_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    return (initial_g, initial_rb), (filtered_g, filtered_rb), (cleaned_g, cleaned_rb)
