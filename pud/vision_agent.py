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
        state = self.encoder.get_latent_state(state, self.device)
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
            # policy args
            ):
        super(VisionUVFDDPG, self).__init__()

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

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-07)

        self.optimize_iterations = 0

    def select_action(self, state):
        self.actor(state)
        import IPython
        IPython.embed(colors="Linux")

        pass