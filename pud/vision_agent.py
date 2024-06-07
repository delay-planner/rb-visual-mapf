import torch
from torch import nn
from typing import Union
from torch.nn import functional as F
import copy
import numpy as np
from pud.utils import variance_initializer_
from pud.algos.data_struct import inp_to_torch_device
from pud.ddpg import UVFDDPG, merge_obs_goal, EnsembledCritic
from pud.visual_models import VisualEncoder
from termcolor import colored
import functools

class VisualActor(nn.Module): # TODO: [256, 256], MLP class
    def __init__(self, 
            state_dim,
            action_dim, 
            max_action, 
            embedding_size:int=256, 
            act_fn=nn.SELU,
            in_channels:int=4, 
            device=torch.device("cpu"),
            ):
        super().__init__()

        self.encoder = VisualEncoder(
                    in_channels=in_channels, 
                    embedding_size=embedding_size, 
                    act_fn=act_fn
                    )
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.device = device
        self.max_action = max_action
        self.reset_parameters()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = self.max_action * a 
        return a

    def reset_parameters(self):
        self.encoder.reset_parameters()
        variance_initializer_(self.l1.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l1.bias)
        variance_initializer_(self.l2.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l3.bias)
    

class VisualCritic(nn.Module):
    def __init__(self, 
            state_dim,
            action_dim, 
            embedding_size:int, 
            act_fn=nn.SELU,
            output_dim=1,
            in_channels:int=4, 
            device=torch.device("cpu"),
            ):
        super().__init__()

        self.encoder = VisualEncoder(
                    in_channels=in_channels, 
                    embedding_size=embedding_size, 
                    act_fn=act_fn
                    )
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256 + action_dim, 256)
        self.l3 = nn.Linear(256, output_dim)
        self.device = device

        self.reset_parameters()

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], dim=1)))
        q = self.l3(q)
        return q

    def reset_parameters(self):
        self.encoder.reset_parameters()
        variance_initializer_(self.l1.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l1.bias)
        variance_initializer_(self.l2.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l3.bias)

class VisualGoalConditionedActor(VisualActor):
    def forward(self, state):
        latent_state = self.encoder.get_latent_state(state, self.device)
        modified_state = merge_obs_goal(latent_state)
        return super().forward(modified_state)

class VisualGoalConditionedCritic(VisualCritic):
    def forward(self, state, action):
        latent_state = self.encoder.get_latent_state(state, self.device)
        modified_state = merge_obs_goal(latent_state)
        return super().forward(modified_state, action)

class VisionUVFDDPG (nn.Module):
    def __init__(
            self,
            # encoder args
            in_channels:int,
            embedding_size:int,
            act_fn,
            device:str,
            uvfddpg_kwargs:dict,
            ActorCls=VisualGoalConditionedActor, 
            CriticCls=VisualGoalConditionedCritic,
            ensemble_size:int=2,
            num_bins:int=20,
            actor_update_interval:int=1,
            use_distributional_rl:bool=True,
            # policy args
            ):
        super(VisionUVFDDPG, self).__init__()
        self.state_dim = embedding_size*2
        self.action_dim = uvfddpg_kwargs["action_dim"]
        self.max_action = uvfddpg_kwargs["max_action"]
        self.discount = uvfddpg_kwargs["discount"]
        self.targets_update_interval = uvfddpg_kwargs["targets_update_interval"]
        self.tau = uvfddpg_kwargs["tau"]
        self.ensemble_size = ensemble_size
        self.device = torch.device(device)
        self.use_distributional_rl = use_distributional_rl
        self.num_bins = num_bins
        self.actor_update_interval = actor_update_interval

        if self.use_distributional_rl:
            self.discount = 1
            CriticCls = functools.partial(CriticCls, output_dim=self.num_bins)

        self.actor = ActorCls(
            state_dim=embedding_size*2,
            action_dim=uvfddpg_kwargs["action_dim"], 
            max_action=uvfddpg_kwargs["max_action"], 
            embedding_size=embedding_size,
            act_fn=act_fn,
            in_channels=in_channels,
            device=device,
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-07)

        self.critic = CriticCls(
            state_dim=embedding_size*2,
            action_dim=uvfddpg_kwargs["action_dim"], 
            embedding_size=embedding_size,
            act_fn=act_fn,
            in_channels=in_channels,
            device=device,         
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-07)

        if self.ensemble_size > 1:
            self.critic = EnsembledCritic(self.critic, ensemble_size=ensemble_size)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            for i in range(1, len(self.critic.critics)): # first copy already added
                critic_copy = self.critic.critics[i]
                self.critic_optimizer.add_param_group({'params': critic_copy.parameters()})
                # https://stackoverflow.com/questions/51756913/in-pytorch-how-do-you-use-add-param-group-with-a-optimizer

        self.optimize_iterations = 0

    def select_action(self, state):
        with torch.no_grad():
            return self.actor(state).cpu().detach().numpy().flatten()
    
    def get_q_values(self, state, aggregate='mean'):
        actions = self.actor(state)
        q_values = self.critic(state, actions)
        return q_values

    def optimize(self, replay_buffer, iterations=1, batch_size=128):
        opt_info = dict(actor_loss=[], critic_loss=[])
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches 
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            state = inp_to_torch_device(state, self.device)
            next_state = inp_to_torch_device(next_state, self.device)
            action = inp_to_torch_device(action, self.device)
            reward = inp_to_torch_device(reward, self.device)
            done = inp_to_torch_device(done, self.device)

            current_q = self.critic(state, action)
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.critic_loss(current_q, target_q, reward, done)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            opt_info['critic_loss'].append(critic_loss.cpu().detach().numpy())

            if self.optimize_iterations % self.actor_update_interval == 0:
                # Compute actor loss
                actor_loss = -self.get_q_values(state).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                opt_info['actor_loss'].append(actor_loss.cpu().detach().numpy())

            # Update the frozen target models
            if self.optimize_iterations % self.targets_update_interval == 0:
                self.update_actor_target()
                self.update_critic_target()

        return opt_info

    def critic_loss(self, current_q, target_q, reward, done):
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        else:
            current_q_list = current_q
            target_q_list = target_q

        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):
            if self.use_distributional_rl:
                # Compute distributional td targets
                target_q_probs = F.softmax(target_q, dim=1)
                batch_size = target_q_probs.shape[0]
                one_hot = torch.zeros(batch_size, self.num_bins).to(reward.device)
                one_hot[:, 0] = 1

                # Calculate the shifted probabilities
                # Fist column: Since episode didn't terminate, probability that the
                # distance is 1 equals 0.
                col_1 = torch.zeros((batch_size, 1)).to(reward.device)
                # Middle columns: Simply the shifted probabilities.
                col_middle = target_q_probs[:, :-2]
                # Last column: Probability of taking at least n steps is sum of
                # last two columns in unshifted predictions:
                col_last = torch.sum(target_q_probs[:, -2:], dim=1, keepdim=True)
                shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim=1)
                assert one_hot.shape == shifted_target_q_probs.shape
                td_targets = torch.where(done.bool(), one_hot, shifted_target_q_probs).detach()

                critic_loss = torch.mean(-torch.sum(td_targets * torch.log_softmax(current_q, dim=1), dim=1)) # https://github.com/tensorflow/tensorflow/issues/21271
            else:
                critic_loss = super().critic_loss(current_q, target_q, reward, done)
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss
