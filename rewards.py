import torch
import torch.nn as nn

from model import EnergyFlowModel


class CenteredReward(nn.Module):
    def __init__(
        self,
        model: EnergyFlowModel,
        num_baseline_samples: int = 16,
        gamma: float = 1e-3,
        seed: int = 0,
    ):
        super().__init__()
        self.model = model
        self.gamma = gamma
        gen = torch.Generator().manual_seed(seed)
        reference = torch.randn(
            num_baseline_samples, model.pred_horizon, model.action_dim, generator=gen
        )
        self.register_buffer("reference_actions", reference)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        t = torch.full((B,), self.gamma, device=obs.device)

        energy = self.model.energy(action, obs, t)

        ref = self.reference_actions.to(obs.device)
        M = ref.shape[0]
        obs_rep = obs.unsqueeze(1).expand(B, M, *obs.shape[1:]).reshape(B * M, *obs.shape[1:])
        ref_rep = ref.unsqueeze(0).expand(B, M, *ref.shape[1:]).reshape(B * M, *ref.shape[1:])
        t_rep = t.repeat_interleave(M)
        baseline = self.model.energy(ref_rep, obs_rep, t_rep).view(B, M).mean(dim=1)

        return -(energy - baseline)
