from __future__ import annotations

import torch
from pud.algos.gco.constraint_act import PiecewiseCosineScheduler, PiecewiseLinearScheduler, JumpScheduler, ConstantZeroScheduler
from typing import Union

def get_grad(net:torch.nn.Module):
    grad = []
    with torch.no_grad():
        for param in net.parameters():
            if param.grad is not None:
                grad.extend(torch.autograd.Variable(param.grad.data.clone().flatten(), requires_grad=False))
        grad = torch.stack(grad)
    return grad

class GradParetoConstrainedAnalytical2DOpt:
    def __init__(self):
        super(GradParetoConstrainedAnalytical2DOpt, self).__init__()

    def solve(self, 
            pi_grad_t:torch.Tensor,
            c_grad_t:torch.Tensor,
            c:float=0.0,
            ):
        """
        pi_grad_t: (N, 1)
        c_grad_t: (N, 1)
        c: float

        argmin:
        $$\beta_{R}^{*}=\begin{cases}
        1/2 & \frac{1-c}{1-\cos\left(\rho\right)}\ge1/2\\
        \frac{1-c}{1-\cos\left(\rho\right)} & \textnormal{otherwise}
        \end{cases}$$
        $$\beta_{C}^{*}=1-\beta_{R}^{*}$$

        min:
        $$1-2\beta_{R}\left(1-\beta_{R}\right)\left(1-\cos\left(\rho\right)\right)$$
        keep variables within torch, no need to move tensors to cpus explicitly
        """
        device = pi_grad_t.device

        # normalization required from analytical expression
        pi_grad_t_unit = pi_grad_t
        c_grad_t_unit = c_grad_t 
        with torch.no_grad():
            pi_grad_t_unit = pi_grad_t / float(torch.linalg.norm(pi_grad_t))
            c_grad_t_unit = c_grad_t / float(torch.linalg.norm(c_grad_t))

        grad_c_T = c_grad_t_unit.t()
        cos_rho = float(grad_c_T @ pi_grad_t_unit)

        beta_c = 0.5
        beta_r = 0.5
        obj = 1.0 - 2.0*beta_c*beta_r*(1.0-cos_rho)
        if cos_rho == 1:
            return torch.FloatTensor([beta_r, beta_c]).to(device=device), obj

        alpha_ub = (1.0-c)/(1.0-cos_rho)
        if alpha_ub >= 0.5:
            return torch.FloatTensor([beta_r, beta_c]).to(device=device), obj
        
        beta_r = alpha_ub
        beta_c = 1.0-alpha_ub
        obj = 1.0 - 2.0*beta_c*beta_r*(1.0-cos_rho)
        return torch.FloatTensor([beta_r, beta_c]).to(device=device), obj


class ConstrainedPareto:
    """
    constrained pareto module for one-constraint CMDP
    """
    def __init__(
        self,
        cost_limit: float, # d in paper
        cost_step_limit: float, 
        cost_margin: float = -1, # d_h in paper
        bias_c: float = 0.2, # bias reward or cost when needed
        bias_sched='cos',
        target_kl:float=0.01,
        ):

        self.cost_limit = cost_limit
        self.cost_step_limit = cost_step_limit
        self.bias_c = bias_c
        self.target_kl = target_kl

        self.cost_bias_scheduler = None
        if bias_sched == 'cos':
            self.cost_bias_scheduler = PiecewiseCosineScheduler(target_margin=cost_margin, limit=cost_limit)
        elif bias_sched == 'linear':
            self.cost_bias_scheduler = PiecewiseLinearScheduler(target_margin=cost_margin, limit=cost_limit)
        elif bias_sched == "jump":
            self.cost_bias_scheduler = JumpScheduler(target_margin=cost_margin, limit=cost_limit)
        elif bias_sched == "zero":
            self.cost_bias_scheduler = ConstantZeroScheduler(target_margin=cost_margin, limit=cost_limit)
            

        self.c_solver = GradParetoConstrainedAnalytical2DOpt()

        # a safety margin for the solver to stop
        assert cost_margin >= 0, "cost margin needs to >= 0"
        self.cost_margin = cost_margin

        self.solver_output :Union[None, dict] = None

    def update_bias_c(self, Jc:float):
        """
        manually clipping the bias c for Pareto solver
        """
        # update the cost constraint
        if Jc < self.cost_margin:
            self.bias_c = 0.0
        else:
            self.bias_c = self.cost_bias_scheduler(Jc, symmetric=False)
        return self.bias_c

    def solve(self, 
              pi_grad_t:torch.Tensor,
              c_grad_t:torch.Tensor,
              c_c:float):
        """
        Note: c_r and c_c cannot be both >= 0 at the same time
        """
        alpha, cost = self.c_solver.solve(pi_grad_t=pi_grad_t, c_grad_t=c_grad_t, c=c_c)
        return alpha, cost
