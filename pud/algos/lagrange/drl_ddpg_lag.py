import copy
import functools
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import torch

from pud.ddpg import (
    EnsembledCritic,
    GoalConditionedActor,
    GoalConditionedCritic,
    AutoEncoder,
)
from pud.algos.distributional_ops import CategoricalActivation
from pud.algos.constrained_buffer import ConstrainedReplayBuffer
from pud.ddpg import UVFDDPG
from pud.algos.lagrange.lagrange import Lagrange
from pud.algos.data_struct import inp_to_device

nn = torch.nn
F = nn.functional


class DRLDDPGLag(UVFDDPG):
    """
    Base Distributional DDPG, removing unnecessary compat to non-distributional RL code
    Aim to have a clean separation from unconstrained and constrained, have costs in separate loops, even at the cost of efficiency

    todo: missing Lagrange update
        - in external loop
        - get Jc from the collector
        - needs to add Lagrange multiplier to the actor loss
    """

    def __init__(
        self,
        # DDPG args
        state_dim,
        action_dim,
        max_action,
        discount=1,
        actor_update_interval=1,
        targets_update_interval=1,
        tau=0.005,
        CriticCls=GoalConditionedCritic,
        # UVFDDPG args
        num_bins=20,
        use_distributional_rl=True,
        ensemble_size=3,
        # cost configs
        cost_min: float = 0,
        cost_max: float = 2.0,
        cost_N=20,
        cost_critic_lr: float = 1e-3,
        cost_limit: float = 1.0,
        lambda_lr: float = 0.001,
        lambda_optimizer: str = "Adam",
        # Image observation parameters
        obs_dim: Tuple = (4, 256, 256, 4),
        base_channel_size: int = 32,
        latent_dimension: int = 512,
        AutoEncoderCls: Union[object, None] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(DRLDDPGLag, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=discount,
            actor_update_interval=actor_update_interval,
            targets_update_interval=targets_update_interval,
            tau=tau,
            # ActorCls=ActorCls,
            # CriticCls=CriticCls,
            # lr configs
            # actor_lr=actor_lr,
            # critic_lr=critic_lr,
            # device=device,
            # UVFDDPG args
            num_bins=num_bins,
            use_distributional_rl=use_distributional_rl,
            ensemble_size=ensemble_size,
        )

        self.lagrange = Lagrange(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=0.0,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
        )
        self.lagrange_on = False
        self.device = device

        # Add the autoencoder
        if AutoEncoderCls is not None:
            self.image_directions = 4
            self.use_autoencoder = True
            self.latent_dimension = latent_dimension
            self.autoencoder = AutoEncoderCls(
                width=obs_dim[1],
                height=obs_dim[2],
                input_channels=obs_dim[3],
                base_channel_size=base_channel_size,  # default 32
                latent_dimension=latent_dimension,  # default 512
            )  # type: ignore
            self.autoencoder.to(device)
            self.autoencoder_optimizer = torch.optim.Adam(
                self.autoencoder.parameters(), lr=1e-3
            )

        # add cost critic
        CostCriticCls = functools.partial(CriticCls, output_dim=cost_N)
        self.F_categorical = CategoricalActivation(
            vmin=cost_min, vmax=cost_max, N=cost_N
        )
        self.cost_critic = CostCriticCls(state_dim, action_dim)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=cost_critic_lr, eps=1e-07
        )

        if self.ensemble_size > 1:
            self.cost_critic = EnsembledCritic(
                self.cost_critic, ensemble_size=ensemble_size
            )
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            for i in range(
                1, len(self.cost_critic.critics)
            ):  # first copy already added
                cost_critic_copy = self.cost_critic.critics[i]
                self.cost_critic_optimizer.add_param_group(
                    {"params": cost_critic_copy.parameters()}
                )

    def get_img_embedding(self, img):
        """Get the embedding of an image"""
        with torch.no_grad():
            img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
            img = inp_to_device(img, self.device)
            return self.autoencoder.embed(img)

    def get_img_embedding_batch(self, img):
        """Get the embedding of an image"""
        with torch.no_grad():
            img = img.permute(0, 3, 1, 2)
            img = inp_to_device(img, self.device)
            return self.autoencoder.embed(img)

    def extract_latent(self, state):

        forward, left, right, backward = 0, 1, 2, 3
        forward_observation_emb = self.get_img_embedding(state["observation"][forward])
        forward_goal_emb = self.get_img_embedding(state["goal"][forward])
        left_observation_emb = self.get_img_embedding(state["observation"][left])
        left_goal_emb = self.get_img_embedding(state["goal"][left])
        right_observation_emb = self.get_img_embedding(state["observation"][right])
        right_goal_emb = self.get_img_embedding(state["goal"][right])
        backward_observation_emb = self.get_img_embedding(
            state["observation"][backward]
        )
        backward_goal_emb = self.get_img_embedding(state["goal"][backward])
        state_observation = torch.cat(
            (
                forward_observation_emb,
                left_observation_emb,
                right_observation_emb,
                backward_observation_emb,
            ),
            dim=0,
        )
        state_goal = torch.cat(
            (
                forward_goal_emb,
                left_goal_emb,
                right_goal_emb,
                backward_goal_emb,
            ),
            dim=0,
        )
        return state_observation, state_goal

    def extract_latent_batch(self, state):

        forward, left, right, backward = 0, 1, 2, 3
        forward_observation_emb = self.get_img_embedding_batch(
            state["observation"][:, forward]
        )
        forward_goal_emb = self.get_img_embedding_batch(state["goal"][:, forward])
        left_observation_emb = self.get_img_embedding_batch(
            state["observation"][:, left]
        )
        left_goal_emb = self.get_img_embedding_batch(state["goal"][:, left])
        right_observation_emb = self.get_img_embedding_batch(
            state["observation"][:, right]
        )
        right_goal_emb = self.get_img_embedding_batch(state["goal"][:, right])
        backward_observation_emb = self.get_img_embedding_batch(
            state["observation"][:, backward]
        )
        backward_goal_emb = self.get_img_embedding_batch(state["goal"][:, backward])
        state_observation = torch.cat(
            (
                forward_observation_emb,
                left_observation_emb,
                right_observation_emb,
                backward_observation_emb,
            ),
            dim=1,
        )
        state_goal = torch.cat(
            (
                forward_goal_emb,
                left_goal_emb,
                right_goal_emb,
                backward_goal_emb,
            ),
            dim=1,
        )
        return state_observation, state_goal

    def select_action(self, state):
        """enable options on cuda, move input to self.device"""
        with torch.no_grad():
            if self.use_autoencoder:
                state_observation, state_goal = self.extract_latent(state)
                state = dict(
                    observation=state_observation.reshape(1, -1),
                    goal=state_goal.reshape(1, -1),
                )
            else:
                state = dict(
                    observation=torch.FloatTensor(state["observation"].reshape(1, -1)),
                    goal=torch.FloatTensor(state["goal"].reshape(1, -1)),
                )
            state = inp_to_device(state, self.device)

            return self.actor(state).cpu().detach().numpy().flatten()

    def get_dist_to_goal(self, state, **kwargs):
        with torch.no_grad():
            if self.use_autoencoder:
                state_observation, state_goal = self.extract_latent(state)
                state = dict(
                    observation=state_observation.reshape(1, -1),
                    goal=state_goal.reshape(1, -1),
                )
            else:
                state = dict(
                    observation=torch.FloatTensor(state["observation"]),
                    goal=torch.FloatTensor(state["goal"]),
                )
            state = inp_to_device(state, self.device)
            q_values = self.get_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy().squeeze(-1)

    def update_cost_critic_target(self):
        for param, target_param in zip(
            self.cost_critic.parameters(), self.cost_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def set_lag_status(self, turn_on_lag: bool):
        self.lagrange_on = turn_on_lag

    def optimize(
        self,
        replay_buffer: ConstrainedReplayBuffer,
        iterations=1,
        batch_size=128,
    ):
        """override the default optimize to include the Lagrange optimize and have an unified self.optimize_iterations count"""
        opt_info = dict(actor_loss=[], critic_loss=[], cost_critic_loss=[])
        if self.use_autoencoder:
            opt_info["autoencoder_loss"] = []
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches
            state, next_state, action, reward, cost, done = replay_buffer.sample_w_cost(
                batch_size
            )
            state = inp_to_device(state, self.device)
            next_state = inp_to_device(next_state, self.device)
            action = inp_to_device(action, self.device)
            reward = inp_to_device(reward, self.device)
            cost = inp_to_device(cost, self.device)
            done = inp_to_device(done, self.device)

            if self.use_autoencoder:
                autoencoder_loss = self.autoencoder._get_reconstruction_loss(
                    state["observation"][:, 0].permute(0, 3, 1, 2)
                ) + self.autoencoder._get_reconstruction_loss(
                    state["goal"][:, 0].permute(0, 3, 1, 2)
                )
                for direction in range(1, self.image_directions):
                    autoencoder_loss += self.autoencoder._get_reconstruction_loss(
                        state["observation"][:, direction].permute(0, 3, 1, 2)
                    )
                    autoencoder_loss += self.autoencoder._get_reconstruction_loss(
                        state["goal"][:, direction].permute(0, 3, 1, 2)
                    )

                # Optimize the autoencoder
                self.autoencoder_optimizer.zero_grad()
                autoencoder_loss.backward()
                self.autoencoder_optimizer.step()
                opt_info["autoencoder_loss"].append(
                    autoencoder_loss.cpu().detach().numpy()
                )

                state_observation, state_goal = self.extract_latent_batch(state)
                state = dict(
                    observation=state_observation,
                    goal=state_goal,
                )

                next_state_observation, next_state_goal = self.extract_latent_batch(
                    next_state
                )
                next_state = dict(
                    observation=next_state_observation,
                    goal=next_state_goal,
                )

            current_q = self.critic(state, action)
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.critic_loss(current_q, target_q, reward, done)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            opt_info["critic_loss"].append(critic_loss.cpu().detach().numpy())

            # for cost critic
            cost_current_q = self.cost_critic(state, action)
            cost_target_q = self.cost_critic_target(
                next_state, self.actor_target(next_state)
            )
            cost_critic_loss = self.cost_critic_loss(
                cost_current_q, cost_target_q, cost, done
            )
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic_optimizer.step()
            opt_info["cost_critic_loss"].append(cost_critic_loss.cpu().detach().numpy())

            if self.optimize_iterations % self.actor_update_interval == 0:
                # Compute actor loss
                actor_loss_r = -self.get_q_values(state)
                actor_loss = 0.0  # placeholder
                if self.lagrange_on:
                    actor_loss_c = self.get_cost_q_values(state)
                    # todo: figure out whether the masked version is better, it should enforce a <= constraint?
                    mask_loss_c = actor_loss_c > self.lagrange.cost_limit
                    lag = self.lagrange.lagrangian_multiplier.item()
                    masked_actor_loss_c = (actor_loss_c * mask_loss_c).mean() * lag
                    actor_loss =  (actor_loss_r.mean() + masked_actor_loss_c) / (1 + lag)
                else:
                    actor_loss = actor_loss_r.mean()  # debug training, drop lagrange
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                opt_info["actor_loss"].append(actor_loss.cpu().detach().numpy())

            # Update the frozen target models
            if self.optimize_iterations % self.targets_update_interval == 0:
                self.update_actor_target()
                self.update_critic_target()
                self.update_cost_critic_target()

        return opt_info

    def optimize_lagrange(self, ep_cost: float):
        """wrapper of lagrange update, lagrange is updated separately as there are different options:
        1. update lagrange when new episode finishes
        2. update lagrange along with the main optimize call regardless new episode has finished
        NOTE: the omnisafe ddpg updates lagrange multiplier regardless a new episode has finished
        """
        self.lagrange.update_lagrange_multiplier(ep_cost)
        return self.lagrange.lagrangian_multiplier.item()

    def cost_critic_loss(
        self,
        current_q: Union[torch.Tensor, List[torch.Tensor]],
        target_q: Union[torch.Tensor, List[torch.Tensor]],
        cost: torch.Tensor,  # (N,1)
        done: torch.Tensor,  # (N,1)
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
                zs = self.F_categorical.zs.tile([batch_size, 1]).to(self.device)
                new_zs = cost + (
                    (1 - done) * self.discount * zs
                )  # batch_size, num_classes
                new_target_probs = self.F_categorical.forward(
                    probs=target_q_probs, new_zs=new_zs
                )
            # cross entry loss: $$-\sum_{i}m_{i}\log p_{i}\left(x_{t},a_{t}\right)$$
            critic_loss = torch.mean(
                -torch.sum(
                    new_target_probs * torch.log_softmax(current_q, dim=1), dim=1
                )
            )  #
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss

    def _get_cost_q_values(self, state):
        actions = self.actor(state)
        q_values = self.cost_critic(state, actions)
        return q_values

    def get_cost_q_values(
        self, state: Union[torch.Tensor, Dict[str, torch.Tensor]], aggregate="mean"
    ):
        """"""
        q_values = self._get_cost_q_values(state)
        q_values_list = []
        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values

        expected_q_values_list = []
        for q_values in q_values_list:
            q_probs = F.softmax(q_values, dim=1)
            batch_size = q_probs.shape[0]
            zs = self.F_categorical.zs.tile([batch_size, 1]).to(self.device)
            # Take the inner product between these two tensors
            expected_q_values = torch.sum(q_probs * zs, dim=1, keepdim=True)
            expected_q_values_list.append(expected_q_values)

        expected_q_values = torch.stack(expected_q_values_list)
        if aggregate is not None:
            if aggregate == "mean":
                expected_q_values = torch.mean(expected_q_values, dim=0)
            elif aggregate == "min":
                expected_q_values, _ = torch.min(expected_q_values, dim=0)
            else:
                raise ValueError
        return expected_q_values

    def get_cost_to_goal(self, state, **kwargs):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state["observation"]),
                goal=torch.FloatTensor(state["goal"]),
            )
            state = inp_to_device(state, self.device)
            q_values = self.get_cost_q_values(state, **kwargs)
            return q_values.cpu().detach().numpy().squeeze(-1)

    def get_pairwise_cost(self, obs_vec, goal_vec=None, aggregate="mean"):
        """Estimates the pairwise costs. Return ensemble_size, obs_vec, goal_vec

        obs_vec: Array containing observations
        goal_vec: (optional) Array containing a second set of observations. If
                  not specified, computes the pairwise distances between obs_tensor and
                  itself.
        aggregate: (str) How to combine the predictions from the ensemble. Options
                   are to take the minimum predicted q value (i.e., the maximum distance),
                   the mean, or to simply return all the predictions.
        max_search_steps: (int)
        masked: (bool) Whether to ignore edges that are too long, as defined by
                max_search_steps.
        """
        if goal_vec is None:
            goal_vec = obs_vec

        dist_matrix = []
        for obs_index in range(len(obs_vec)):
            obs = obs_vec[obs_index]
            # obs_repeat_tensor = np.ones_like(goal_vec) * np.expand_dims(obs, 0)
            obs_repeat_tensor = np.repeat([obs], len(goal_vec), axis=0)
            state = {"observation": obs_repeat_tensor, "goal": goal_vec}
            dist = self.get_cost_to_goal(state, aggregate=aggregate)
            dist_matrix.append(dist)

        pairwise_dist = np.stack(dist_matrix)
        if aggregate is None:
            pairwise_dist = np.transpose(pairwise_dist, [1, 0, 2])
        # else:
        #     pairwise_dist = np.expand_dims(pairwise_dist, 0)

        return pairwise_dist

    def state_dict(self):
        out = super().state_dict()
        out["cost_critic"] = self.cost_critic.state_dict()
        out["cost_critic_optimizer"] = self.cost_critic_optimizer.state_dict()
        # out["lagrange"] = self.lagrange.state_dict()
        return out

    def load_state_dict(self, state_dict: dict):
        unconstrained_keys = [
            "actor",
            "actor_optimizer",
            "critic",
            "critic_optimizer",
            "optimize_iterations",
        ]
        unconstrained_state_dict = {}
        for key in unconstrained_keys:
            unconstrained_state_dict[key] = state_dict[key]
        super().load_state_dict(unconstrained_state_dict)

        self.cost_critic.load_state_dict(state_dict["cost_critic"])
        self.cost_critic_optimizer.load_state_dict(state_dict["cost_critic_optimizer"])
        # self.lagrange.load_state_dict(state_dict["lagrange"])
