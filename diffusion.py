import math

import torch

from model import EnergyFlowModel


class VESchedule:
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 10.0, T: float = 1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        self._log_ratio = math.log(sigma_max / sigma_min)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        frac = t / self.T
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** frac

    def d_sigma_sq(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * self.sigma(t) ** 2 * self._log_ratio / self.T


def dsm_loss(
    model: EnergyFlowModel,
    schedule: VESchedule,
    obs: torch.Tensor,
    action: torch.Tensor,
    t: torch.Tensor | None = None,
    eps: torch.Tensor | None = None,
) -> torch.Tensor:
    B = action.shape[0]
    if t is None:
        t = torch.rand(B, device=action.device) * schedule.T
    if eps is None:
        eps = torch.randn_like(action)
    sigma = schedule.sigma(t)
    a_t = action + sigma.view(-1, 1, 1) * eps

    score = model(a_t, obs, t)
    return (sigma.view(-1, 1, 1) * score + eps).pow(2).mean()


@torch.no_grad()
def sample_actions(
    model: EnergyFlowModel,
    schedule: VESchedule,
    obs: torch.Tensor,
    num_steps: int = 20,
    gamma: float = 1e-3,
    init: torch.Tensor | None = None,
    return_energy: bool = False,
):
    B = obs.shape[0]
    device = obs.device
    if init is None:
        a = torch.randn(B, model.pred_horizon, model.action_dim, device=device)
        a = a * schedule.sigma_max
    else:
        a = init.clone()
    T = schedule.T
    dt = (T - gamma) / num_steps
    for k in range(num_steps):
        t_k = T - k * dt
        t = torch.full((B,), t_k, device=device)
        grad = -model.score(a, obs, t, create_graph=False)
        g = 0.5 * schedule.d_sigma_sq(t).view(-1, 1, 1) * grad
        a = a - dt * g
    if return_energy:
        t_end = torch.full((B,), gamma, device=device)
        energy = model.energy(a, obs, t_end)
        return a, energy
    return a
