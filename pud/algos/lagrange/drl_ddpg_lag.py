import copy
import functools
from typing import Any, Dict, List, Union

import numpy as np
import torch

from pud.ddpg import (EnsembledCritic, GoalConditionedActor, GoalConditionedCritic)
from pud.algos.distributional_ops import CategoricalActivation
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.ddpg import UVFDDPG

nn = torch.nn
F = nn.functional

class DRLDDPGLag(UVFDDPG):
    """
    Base Distributional DDPG, removing unnecessary compat to non-distributional RL code
    Aim to have a clean separation from unconstrained and constrained, have costs in separate loops, even at the cost of efficiency
    """
    def __init__(self, 
            # DDPG args
            state_dim,
            action_dim,
            max_action,
            discount=1,
            actor_update_interval=1,
            targets_update_interval=1,
            tau=0.005,
            ActorCls=GoalConditionedActor, 
            CriticCls=GoalConditionedCritic,

            # lr configs
            actor_lr:float=3e-4,
            critic_lr:float=3e-4,
            device:str='cpu',

            # UVFDDPG args
            num_bins=20,
            use_distributional_rl=True,
            ensemble_size=3,
            
            # cost configs
            cost_min:float = 0,
            cost_max:float = 2.0,
            cost_N=20,
            cost_critic_lr:float=1e-3,
            ):
        super(DRLDDPGLag, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=discount,
            actor_update_interval=actor_update_interval,
            targets_update_interval=targets_update_interval,
            tau=tau,
            #ActorCls=ActorCls, 
            #CriticCls=CriticCls,
            # lr configs
            #actor_lr=actor_lr,
            #critic_lr=critic_lr,
            #device=device,
            # UVFDDPG args
            num_bins=num_bins,
            use_distributional_rl=use_distributional_rl,
            ensemble_size=ensemble_size,
        )

        # add cost critic
        CostCriticCls = functools.partial(CriticCls, output_dim=cost_N)
        self.F_categorical = CategoricalActivation(vmin=cost_min, vmax=cost_max, N=cost_N)
        self.cost_critic = CostCriticCls(state_dim, action_dim)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_critic_lr, eps=1e-07)

        if self.ensemble_size > 1:
            self.cost_critic = EnsembledCritic(self.cost_critic, ensemble_size=ensemble_size)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            for i in range(1, len(self.cost_critic.critics)): # first copy already added
                cost_critic_copy = self.cost_critic.critics[i]
                self.cost_critic_optimizer.add_param_group({'params': cost_critic_copy.parameters()})

    def update_cost_critic_target(self):
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def optimize(self, replay_buffer:ConstrainedReplayBuffer, iterations=1, batch_size=128):
        opt_info = super().optimize(
            replay_buffer=replay_buffer,
            iterations=iterations,
            batch_size=batch_size,
            )
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches 
            state, next_state, action, reward, cost, done =     replay_buffer.sample_w_cost(batch_size)
            
            current_q = self.cost_critic(state, action)
            target_q = self.cost_critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.cost_critic_loss(current_q, target_q, cost, done)


    def cost_critic_loss(self, 
            current_q:Union[torch.Tensor, List[torch.Tensor]], 
            target_q:Union[torch.Tensor, List[torch.Tensor]],
            cost:torch.Tensor, # (N,1) 
            done:torch.Tensor, # (N,1)
            ):
        """loss on cumulative costs
        current_q: a torch tensor if the cost critic is not an ensemble, or a list of torch tensors if the cost critic is an ensemble that contains the outputs of all the cost critic, (N, 1)

        """
        current_q_list = current_q
        target_q_list = target_q
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        
        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):
            # Compute distributional td targets
            new_target_probs = None
            with torch.no_grad():
                target_q_probs = F.softmax(target_q, dim=1)
                batch_size = target_q_probs.shape[0]

                zs = self.F_categorical.zs.tile([batch_size, 1])
                new_zs = cost + ((1 - done) * self.discount * zs) # batch_size, num_classes
                new_target_probs = self.F_categorical.forward(probs=target_q_probs, new_zs=new_zs)
            # cross entry loss: $$-\sum_{i}m_{i}\log p_{i}\left(x_{t},a_{t}\right)$$
            critic_loss = torch.mean(-torch.sum(new_target_probs * torch.log_softmax(current_q, dim=1), dim=1)) # 
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss

    def get_cost_q_values(self, 
                    state: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    actions:Union[None, torch.Tensor], 
                    aggregate='mean'):
        """"""
        pass

    def get_cost_to_goal(self, state, **kwargs):
        pass

    def get_pairwise_cost(self, obs_vec, goal_vec=None, aggregate='mean', max_search_steps=7, masked=False):
        pass

    def optimize_cost_critic(self, buffer:ConstrainedReplayBuffer, batch_size:int=128, opt_info:Dict[str, List[float]]={}):
        pass

    def state_dict(self):
        out = super.state_dict()
        # add cost info


    def load_state_dict(self, state_dict:dict):
        unconstrained_keys = []
        unconstrained_state_dict = {}
        for key in unconstrained_keys:
            unconstrained_state_dict[key] = state_dict[key]
        super().load_state_dict(unconstrained_state_dict)

        # load cost-relevant info
