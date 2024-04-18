"""
Evaluate the accuracy and reliability of reward and cost critics
"""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from termcolor import cprint
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.algos.constrained_collector import ConstrainedCollector as Collector
from pud.algos.constrained_collector import eval_agent_from_Q
from pud.algos.data_struct import init_embedded_dict
from pud.algos.lagrange.drl_ddpg_lag import DRLDDPGLag
from pud.envs.safe_pointenv.safe_wrappers import \
    SafeGoalConditionedPointWrapper, SafeGoalConditionedPointQueueWrapper
from pud.policies import GaussianPolicy
from pud.envs.safe_pointenv.pb_sampler import sample_pbs_by_agent, calc_pairwise_cost, calc_pairwise_dist
from pathlib import Path


def train_eval(
    policy:GaussianPolicy,
    agent:DRLDDPGLag,
    replay_buffer:ConstrainedReplayBuffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    collect_steps=2,
    opt_steps=1,
    batch_size_opt=64,
    eval_func=None, # make this a partial func
    opt_log_interval=100,
    eval_interval=10000,
    eval_distances=[2, 5, 10], # reference grouping based on estimated distances
    eval_cost_intervals=[0., 0.2, 0.5, 1.0], # grouping cost eval results
    tensorboard_writer:Optional[SummaryWriter]=None,
    warmup_epochs:int =100,
    num_eval_episodes:int=10,
    pbar=True,
    verbose=True,
    ckpt_dir:Path=Path(""),
    ):
    """train constrained RL agent"""
    collector = Collector(policy, replay_buffer, env, initial_collect_steps=initial_collect_steps)
    
    num_eps = collector.num_eps
    ep_cost = 0.0
    collector.step(collector.initial_collect_steps)

    pbar = tqdm(total=num_iterations, disable=not pbar)
    t_mark = time.time()
    for i in range(1, num_iterations + 1):
        pbar.update()
        
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(replay_buffer, iterations=opt_steps, batch_size=batch_size_opt)

        if collector.num_eps > num_eps:
            ep_cost = collector.past_eps[-1]["ep_cost"]
            ep_len = collector.past_eps[-1]["ep_len"]
            if verbose:
                cprint("[INFO] eps Jc={:.2f}, eps length={}".format(ep_cost, ep_len), "green")
            num_eps = collector.num_eps

            if i > warmup_epochs:
                agent.optimize_lagrange(ep_cost=ep_cost)


        if i % opt_log_interval == 0:
            if verbose:
                print(f'iteration = {i}, opt_info = {opt_info}')

        if i % eval_interval == 0:
            if isinstance(ckpt_dir, Path):
                torch.save(agent.state_dict(), ckpt_dir.joinpath("ckpt_{:0>7d}".format(i)))

            agent.eval()
            if verbose:
                print(f'evaluating iteration = {i}')

            #eval_func = eval_pointenv_cost_constrained_dists
            eval_info = eval_func(
                agent=agent, 
                eval_env=eval_env,
                eval_distances=eval_distances,
                num_evals=num_eval_episodes,
                cost_intervals=eval_cost_intervals,
                )
            if verbose:
                print('-' * 10)

        if tensorboard_writer:
            tensorboard_writer.add_scalar("Opt/actor_loss", np.mean(opt_info["actor_loss"]), global_step=i)
            tensorboard_writer.add_scalar("Opt/critic_loss", np.mean(opt_info["critic_loss"]), global_step=i)
            tensorboard_writer.add_scalar("Opt/cost_critic_loss", np.mean(opt_info["cost_critic_loss"]), global_step=i)
            tensorboard_writer.add_scalar("Opt/Lagrange_Multiplier", agent.lagrange.lagrangian_multiplier.item(), global_step=i)

            if i % eval_interval == 0:
                rate = float(eval_interval) / (time.time() - t_mark)
                tensorboard_writer.add_scalar("Opt/Rate(Iter per sec)", rate, global_step=i)

                for d_ref in eval_info:
                    for c_ref in eval_info[d_ref]:
                        field_header = "Eval_D={:0>2d} C={:.2f}".format(d_ref, c_ref)
                        # logging for distance prediction
                        tensorboard_writer.add_scalar(field_header+"/d_pred_mean", np.mean(eval_info[d_ref][c_ref]["r"]["pred"]), global_step=i)
                        tensorboard_writer.add_scalar(field_header+"/d_pred_std", np.std(eval_info[d_ref][c_ref]["r"]["pred"]), global_step=i)
                        tensorboard_writer.add_scalar(field_header+"/d_true_mean", np.mean(eval_info[d_ref][c_ref]["r"]["true"]), global_step=i)
                        tensorboard_writer.add_scalar(field_header+"/d_true_std", np.std(eval_info[d_ref][c_ref]["r"]["true"]), global_step=i)

                        # logging for cost prediction
                        tensorboard_writer.add_scalar(field_header+"/c_pred_mean", np.mean(eval_info[d_ref][c_ref]["c"]["pred"]), global_step=i)
                        tensorboard_writer.add_scalar(field_header+"/c_pred_std", np.std(eval_info[d_ref][c_ref]["c"]["pred"]), global_step=i)
                        tensorboard_writer.add_scalar(field_header+"/c_true_mean", np.mean(eval_info[d_ref][c_ref]["c"]["true"]), global_step=i)
                        tensorboard_writer.add_scalar(field_header+"/c_true_std", np.std(eval_info[d_ref][c_ref]["c"]["true"]), global_step=i)
                
                # reset timer 
                t_mark = time.time()


def eval_agent_by_metric(
        agent: DRLDDPGLag,
        eval_env: SafeGoalConditionedPointQueueWrapper,
        num_evals:int,
        sample_size:int,
        target_val:float,
        pval_f, # pair-wise function
        ensemble_agg:str,
    ):
    """"""
    assert num_evals <= sample_size
    pbs = sample_pbs_by_agent(env=eval_env, 
            agent=agent, 
            num_states=sample_size,
            target_val=target_val,
            pval_f=pval_f,
            K=num_evals,
            ensemble_agg=ensemble_agg,
            )
    eval_env.append_pbs(pb_list=pbs)
    eval_stats = eval_agent_from_Q(policy=agent, eval_env=eval_env)
    return eval_stats

def gather_log(eval_stats:dict, attr:str):
    attr_vals = []
    attr_pred = []
    for id in eval_stats:
        attr_vals.append(
            eval_stats[id][attr]
        )
        attr_pred.append(
            eval_stats[id]["init_info"]["prediction"]
        )
    return attr_vals, attr_pred
        

def eval_pointenv_cost_constrained_dists(agent, 
        eval_env:SafeGoalConditionedPointWrapper, 
        num_evals:int=10, 
        sample_size:int=100,
        eval_distances=[2, 5, 10, 20],
        cost_intervals=[0., 0.2, 0.5, 1.0],
        ):
    
    eval_env.set_prob_constraint(1.0)
    
    dist_eval_stats = dict()
    eval_env.eval()

    for ii_d in range(len(eval_distances)):
        dist_eval_i = eval_agent_by_metric(
                    agent=agent, 
                    eval_env=eval_env, 
                    num_evals=num_evals, 
                    sample_size=sample_size, 
                    target_val=eval_distances[ii_d], 
                    pval_f=calc_pairwise_dist,
                    ensemble_agg="mean")
        attr_vals, attr_pred = gather_log(dist_eval_i, attr="rewards")
        dist_eval_stats[ii_d] = {
            "vals": attr_vals,
            "pred": attr_pred,
            "ref": eval_distances[ii_d],
        }

    cost_eval_stats = dict()
    for ii in range(len(cost_intervals)):
        cost_eval_i = eval_agent_by_metric(
                    agent=agent, 
                    eval_env=eval_env, 
                    num_evals=num_evals, 
                    sample_size=sample_size, 
                    target_val=cost_intervals[ii], 
                    pval_f=calc_pairwise_cost,
                    ensemble_agg="mean")
        attr_vals, attr_pred = gather_log(cost_eval_i, attr="costs")
        cost_eval_stats[ii] = {
            "vals": attr_vals,
            "pred": attr_pred,
            "ref": eval_distances[ii],
        }

    eval_stats = {}
    eval_stats["dists"] = dist_eval_stats
    eval_stats["costs"] = cost_eval_stats
    return eval_stats