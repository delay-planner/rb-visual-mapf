"""Implementation of the Lagrange version of the PPO algorithm."""

import shutil
import time
from pathlib import Path

import numpy as np
import torch
#from special_opt.mtl.utils import get_grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
#from omnisafe.common.constrained_pareto_2d import ConstrainedPareto2D
from omnisafe.algorithms.on_policy.pareto.solver import ConstrainedPareto as ConstrainedPareto2D
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from omnisafe.algorithms.on_policy.pareto.utils import RNDModel, get_grad

@registry.register
class PPOParetoBTLS2DRND(PPO):
    """PPO Pareto w backtrack line search.

    A simple combination of the Lagrange method and the Proximal Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the PPOPareto specific model.

        The PPOPareto algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        print("Algorithm: PPOParetoBTLS2D RND")
        for key in self._cfgs.pareto_cfgs:
            print("{}: {}".format(key, self._cfgs.pareto_cfgs[key]))
        self.grad_size = len(torch.nn.utils.parameters_to_vector(self._actor_critic.actor.parameters()))
        self.pareto_solver = ConstrainedPareto2D(**self._cfgs.pareto_cfgs)
        self.prev_epoch = -1

        # setup RND
        obs_dim = self._env.observation_space.shape[0]

        if self._cfgs.pareto_train_cfgs.use_rnd:
            self.rnd_model = RNDModel(
                sizes=[obs_dim, obs_dim*2, obs_dim*2],
                activation="relu",
                output_activation="identity",
                device=self._device,
                clip=self._cfgs.rnd_configs.clip,
                scale=self._cfgs.rnd_configs.scale,
            )
            self._env.add_addons({
                "rnd": self.rnd_model,
            })

        self.update_count = 0

    def _init_log(self) -> None:
        """Log the PPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key("Value/reward_int")

        ## add additional log dirs
        log_dir = Path(self._logger._log_dir)
        eval_dir = log_dir.joinpath("eval")
        eval_dir.mkdir(parents=True, exist_ok=True)

        self.src_bk_dir = log_dir.joinpath("src")
        self.src_bk_dir.mkdir(parents=True, exist_ok=True)

        self.pareto_error_dir = log_dir.joinpath("error_logs")
        self.pareto_error_dir.mkdir(parents=True, exist_ok=True)

        # make file backup
        file_backups = [
            "omnisafe/algorithms/on_policy/pareto/ppo_pareto_btls_2d_rnd.py",
            "omnisafe/common/constrained_pareto_2d.py",
            "sbatch/job_train_ppo_pareto.sh",
            "sbatch/sv_job_train_ppo_pareto.sh",
            "omnisafe/algorithms/on_policy/pareto/solver.py",
        ]

        for fi in file_backups:
            fi_path = Path(fi)
            if fi_path.exists():
                shutil.copyfile(fi, self.src_bk_dir.joinpath(fi_path.name).as_posix())

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        #Jc = self._logger.get_stats("Metrics/EpCost")[0]
        Jc = self.get_eps_rollout_stats("c")
        assert not np.isnan(Jc), "cost for updating lagrange multiplier is nan"

        ###--------------------------------------------
        ## Below is modified update, where actor update uses the whole dataset as opposed to sub-batches, similar to Natural Policy Gradient and Constrained Policy Optimization
        ###--------------------------------------------

        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data["obs"],
            data["act"],
            data["logp"],
            data["target_value_r"],
            data["target_value_c"],
            data["adv_r"],
            data["adv_c"],
        )

        # update actor using the entire dataset
        self._update_actor(obs, act, logp, adv_r, adv_c)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        #for i in track(range(self._cfgs.algo_cfgs.update_iters), description="Updating..."):
        for _ in tqdm(range(self._cfgs.algo_cfgs.update_iters), disable=True):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in tqdm(dataloader, total=len(dataloader), desc="Updating...", disable=True):
                # mse loss + weight l2 reg
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    # mse loss + weight l2 reg
                    self._update_cost_critic(obs, target_value_c)

    def constrained_line_search(self, 
            loss_inputs:dict,
            loss_pareto: dict,
            decay:float=0.1, # step size search multiplicative step
            total_steps:int=15, # max number of search attempts
            verbose=True,
            ) -> float:
        r"""
        calculate the descent step size to ensure constraint satisfaction

        (a) absolute cost bound constraint
        $$J_{C}\left(\pi_{k}\right)+\frac{1}{1-\gamma}\underset{\substack{s\sim\rho^{\pi_{k}}\\
                            a\sim\pi
                        }
                    }{\mathbb{E}}\left[A_{\pi_{k}}^{C}(s,a)\right]\le d$$
        (b) KL divergence constraint over parametric policy space
        $$\bar{D}_{KL}\left(\pi\parallel\pi_{k}\right)\le\delta$$

        (c) reward improvement constraint
        if $$\alpha_{R}^{*}\ge1/2$$
		$$J_{sg}\left(\pi\right)\ge J_{sg}\left(\pi_{k}\right)$$

        (d) relative cost increase bound constraint
		$$\underset{\substack{s\sim\rho^{\pi_{k}}a\sim\pi}}{\mathbb{E}}\left[A_{\pi_{k}}^{C}(s,a)\right]-J_{C}\left(\pi_{k}\right) \le\left(1-h\left(J_{c},d\right)\right)\varDelta$$

        Also as done in TRPO, actor optimizer is replaced by line search here
        Update the actor model in-place

        Take reference from: omnisafe/algorithms/on_policy/base/trpo.py
        "c": loss_c,
        "r": loss_r,
        "loss": loss_agg,
        "grad": grad_agg,
        "alpha": alpha,

        loss_info
            alpha:Union[np.ndarray, List[float]], # pareto solver output weights
            grad: torch.Tensor, # pareto solver output gradient
            loss_r_old: float = None,
            loss_q_c_old: Union[List[float], None] = None,

        """
        # the actor param should be the very first actor before the any update (not start of the mini/sub batch)

        # keep track of gradient accumulation between consecutive batches
        actor_params_old = None
        with torch.no_grad():
            actor_params_old = torch.nn.utils.parameters_to_vector(self._actor_critic.actor.parameters()).detach().clone()
        
        if loss_pareto["r"] is None or loss_pareto["c"] is None:
            with torch.no_grad():
                loss_pareto["r"], loss_pareto["c"] = self._calc_surrogate_adv(**loss_inputs)

        p_dist_old = self._actor_critic.actor._distribution(loss_inputs["obs"])
        lr = 1.0

        eps_len = self.get_eps_rollout_stats("len", agg="mean")
        grad_norm_raw = torch.linalg.norm(loss_pareto["grad"])
        loss_r_new, loss_q_c_new = 0, 0

        final_kl = 0.0
        done = False
        status = 0

        for step in range(total_steps):
            actor_params_new = actor_params_old - loss_pareto["grad"] * lr

            # apply the model parameter
            torch.nn.utils.vector_to_parameters(
                actor_params_new, 
                self._actor_critic.actor.parameters()
                )
            
            with torch.no_grad():
                new_loss_info = self._calc_surrogate_adv(**loss_inputs)
                loss_r_new = new_loss_info["r"]
                loss_q_c_new = new_loss_info["c"]
                q_dist_new = self._actor_critic.actor._distribution(loss_inputs["obs"])

            kl = torch.distributions.kl.kl_divergence(p_dist_old, q_dist_new).mean().item()
            kl = distributed.dist_avg(kl).mean().item()

            if (not torch.isfinite(loss_r_new)) or (not torch.isfinite(loss_q_c_new)):
                status = -1
                if verbose:
                    self._logger.log("[BTLS]: alpha={}, loss_pi not finite".format(loss_pareto["alpha"][0]))
            elif kl > self.pareto_solver.target_kl:
                status = -2
                if verbose:
                    self._logger.log("[BTLS]: alpha={}, violated KL constraint.".format(loss_pareto["alpha"][0]))
            elif torch.all(loss_q_c_new-loss_pareto["c"] > (1-self.pareto_solver.bias_c)*self._cfgs.pareto_cfgs.cost_step_limit):
                status = -3
                if verbose:
                    self._logger.log("[BTLS]: alpha={} cost increased too much".format(loss_pareto["alpha"][0]))
            elif loss_pareto["c"] + eps_len*loss_q_c_new > self._cfgs.pareto_cfgs.cost_limit:
                status = -4
                if verbose:
                    self._logger.log("[BTLS]: alpha={} absolute cost above cost limit".format(loss_pareto["alpha"][0]))
            #elif loss_pareto["alpha"][0] >= 0.5 and loss_r_new-loss_pareto["r"] >= 0.0:
            #    # only check this condition when the cost constraint is NOT activated
            #    status = -5
            #    if verbose:
            #        self._logger.log("[BTLS]: alpha={}, reward loss went up".format(loss_pareto["alpha"][0]))
            else:
                done = True
                if verbose:
                    self._logger.log("[BTLS]: alpha={}, accepted grad at lr={}, at step={}".format(loss_pareto["alpha"][0], lr, step+1))
                status = 1
                final_kl = kl
                break

            lr = lr * decay

        # backup update rules in case backtrack line search fails
        if not done:
            if loss_pareto["alpha"][0] >= 0.5:
                lr = self._cfgs.pareto_train_cfgs["blr"]
                actor_params_new = actor_params_old - loss_pareto["grad_r"] * lr
                torch.nn.utils.vector_to_parameters(
                    actor_params_new, 
                    self._actor_critic.actor.parameters()
                    )
                if verbose:
                    self._logger.log("[BTLS]: use alpha={}, use backup rule for sole reward".format(loss_pareto["alpha"][0]))
            else:
                # if failed to find sol, reset back
                lr = 0.0
                torch.nn.utils.vector_to_parameters(
                    actor_params_old, 
                    self._actor_critic.actor.parameters()
                    )
                if verbose:
                    self._logger.log("[BTLS]: use alpha={}, failed to find solution".format(loss_pareto["alpha"][0]))

        self.logger._tensorboard_writer.add_scalar("Pareto/KL", final_kl, global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/GradStep", lr, global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/GradNorm", lr*grad_norm_raw, global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/LineSearchStep", step+1, global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/LineSearchStatus", status, global_step=self.update_count)
        return lr

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        Jc = self.get_eps_rollout_stats("c")
        loss_inputs = {
                "obs": obs, 
                "act": act, 
                "logp": logp, 
                "adv": adv,
                "Jc": Jc,
                }
        
        loss_pareto = self._loss_pi(**loss_inputs)
        self._actor_critic.actor_optimizer.zero_grad()

        if self._cfgs.pareto_train_cfgs.disable_linesearch:
            # ablation: disable line search
            loss_pareto["loss"].backward()
            self._actor_critic.actor_optimizer.step()
        else:
            self.constrained_line_search(loss_inputs=loss_inputs, loss_pareto=loss_pareto, total_steps=15)
        
        if self._cfgs.pareto_train_cfgs.use_rnd:
            loss_int = self.rnd_model.fit(obs=obs, norm_coef=self._cfgs.rnd_configs.norm_coef)
            self.logger._tensorboard_writer.add_scalar("Loss/Loss_Intr", loss_int, global_step=self.update_count)
        self.update_count += 1

    def _calc_surrogate_adv(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: dict, 
        add_kl_penalty: bool = False,
        add_exploration_incentive: bool = False,
        **kwargs, # other redundant inputs
    ):
        """helper function to calculate the surrogate advantage values"""
        adv_r = adv["r"]
        adv_c = adv["c"]

        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)

        ## reward loss
        ## the reward loss is negative so the optimizer maximize instead of minimize
        ## the expected return (advantage for rewards)
        loss_r = -torch.mean(ratio * adv_r)
        #if add_kl_penalty:
        #    loss_r -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        ## cost loss
        ## the cost loss should NOT have the negative sign in front
        loss_c = torch.mean(adv_c * ratio)

        loss_info = {
            "r": loss_r,
            "c": loss_c,
        }
        return loss_info


    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: dict,
        Jc: float,
        add_kl_penalty: bool = False,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        self.pareto_solver.update_bias_c(Jc=Jc)

        adv_r = adv["r"]
        adv_c = adv["c"]

        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)

        if self._cfgs.pareto_train_cfgs.use_rnd:
            with torch.no_grad():
                # the rnd is technically not the same as the paper, but 
                # empirically this works better
                adv_r = adv_r + self.rnd_model.compute_intrinsic_reward(obs=obs)

        ## reward loss
        ## the reward loss is negative so the optimizer maximize instead of minimize
        ## the expected return (advantage for rewards)
        loss_r = -torch.mean(ratio * adv_r)
        # NOTE: KL penalty is optional
        #if add_kl_penalty:
        #    loss_r -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        self._actor_critic.actor_optimizer.zero_grad() 
        #loss_q_c_mean.backward(retain_graph=True) # use adam opt
        loss_r.backward(retain_graph=True) # use line search opt, release autograd graph
        grad_r = get_grad(self._actor_critic.actor) # flattened
        pi_grad_t = grad_r.unsqueeze(-1) 

        ## cost loss
        loss_c = torch.mean(ratio * adv_c)
        self._actor_critic.actor_optimizer.zero_grad() 
        loss_c.backward(retain_graph=True)
        grad_c = get_grad(self._actor_critic.actor) # flattened
        c_grad_t = grad_c.unsqueeze(-1)

        alpha, _ = self.pareto_solver.solve(
            pi_grad_t=pi_grad_t,
            c_grad_t=c_grad_t,
            c_c=self.pareto_solver.bias_c)

        grad_r_norm = grad_r / torch.linalg.norm(grad_r)
        grad_c_norm = grad_c / torch.linalg.norm(grad_c)
        
        if Jc < self.pareto_solver.cost_margin or np.random.rand() <= self._cfgs.pareto_train_cfgs["greedy_prob"]:
            alpha = np.array([1.0, 0.0])

        grad_agg = grad_r_norm * alpha[0] + grad_c_norm * alpha[1]
        loss_agg = loss_r * alpha[0] + loss_c * alpha[1]

        entropy = distribution.entropy().mean().item()
        
        ## tensorboard
        self.logger._tensorboard_writer.add_scalar("Pareto/CostWeight", alpha[1], global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/CostBias", self.pareto_solver.bias_c, global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/LossR", loss_r.item(), global_step=self.update_count)
        self.logger._tensorboard_writer.add_scalar("Pareto/LossC", loss_c.item(), global_step=self.update_count)
        
        self._logger.store(
            {
                "Train/Entropy": entropy,
                "Train/PolicyRatio": ratio,
                "Train/PolicyStd": std,
                "Loss/Loss_pi": loss_r.mean().item(),
            },
        )

        loss_info = {
            "c": loss_c,
            "r": loss_r,
            "loss": loss_agg,
            "grad_r": grad_r_norm,
            "grad_c": grad_c_norm,
            "grad": grad_agg,
            "alpha": alpha,
        }
        return loss_info


    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> dict:
        r"""Compute surrogate loss.

        PPOLag uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        return {
            "r": adv_r,
            "c": adv_c,
        }

    def log_eps_stats(self, all_eps_stats:dict):
        """perform analysis on the stats from all rollout stats (as opposed to the last one), and plot to tensorboard
        Calc:
            max magnitude of cost violations
            mean magnitude of cost violations
            std magnitude of cost violations
            num of eps that violate cost constraints
        """
        cat_all_eps_r = []
        cat_all_eps_c = []
        cat_all_eps_len = []
        for idx in all_eps_stats.keys():
            cat_all_eps_r.extend(all_eps_stats[idx]["r"])
            cat_all_eps_c.extend(all_eps_stats[idx]["c"])
            cat_all_eps_len.extend(all_eps_stats[idx]["len"])
        
        cat_all_eps_r = torch.FloatTensor(cat_all_eps_r)
        cat_all_eps_c = torch.FloatTensor(cat_all_eps_c)
        cat_all_eps_len = torch.FloatTensor(cat_all_eps_len)

        cost_limit = self._cfgs.pareto_cfgs.cost_limit
        cost_violate = cat_all_eps_c - cost_limit
        eps_inds_violate,  = torch.where(cost_violate > 0)
        cost_violate_max, cost_violate_mean, cost_violate_std = 0., 0., 0.
        if len(eps_inds_violate) > 0:
            cost_violate_max = torch.max(cost_violate[eps_inds_violate])
            cost_violate_mean = torch.mean(cost_violate[eps_inds_violate])
            cost_violate_std = torch.std(cost_violate[eps_inds_violate])

        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Cost_Max", cat_all_eps_c.max(), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Cost_Mean", cat_all_eps_c.mean(), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Cost_Std", cat_all_eps_c.std(), self._epoch)

        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Ret_Max", cat_all_eps_r.max(), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Ret_Mean", cat_all_eps_r.mean(), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Ret_Std", cat_all_eps_r.std(), self._epoch)

        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Len_Max", cat_all_eps_len.max(), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Len_Mean", cat_all_eps_len.mean(), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Len_Std", cat_all_eps_len.std(), self._epoch)

        self._logger._tensorboard_writer.add_scalar("RolloutTrain/NumEpsViolate", len(eps_inds_violate), self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Violation_Max", cost_violate_max, self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Violation_Mean", cost_violate_mean, self._epoch)
        self._logger._tensorboard_writer.add_scalar("RolloutTrain/Violation_Std", cost_violate_std, self._epoch)