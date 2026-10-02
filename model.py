import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[..., None].float() * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Softplus(beta=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_channels)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        film = self.cond_proj(cond)
        scale, bias = film.chunk(2, dim=-1)
        out = out * (1.0 + scale[..., None]) + bias[..., None]
        out = self.conv2(out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        down_dims=(64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [action_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
        self.down_modules = down_modules

        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)
        self.mid_block2 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)

        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # Same indexing as Diffusion Policy: every up block upsamples, so
            # the output length equals the input length.
            is_last = ind >= len(in_out) - 1
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        nn.Identity()
                        if is_last
                        else nn.Sequential(
                            nn.Upsample(scale_factor=2.0, mode="nearest"),
                            nn.Conv1d(dim_in, dim_in, 3, padding=1),
                        ),
                    ]
                )
            )
        self.up_modules = up_modules

        self.out_dim = down_dims[0]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        skips = []
        for block1, block2, downsample in self.down_modules:
            x = block1(x, cond)
            x = block2(x, cond)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for block1, block2, upsample in self.up_modules:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = upsample(x)

        return x


class ScalarEnergyHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dims=(256, 128)):
        super().__init__()
        dims = [in_channels] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            last = i == len(dims) - 2
            layers.append(
                nn.utils.spectral_norm(nn.Linear(dims[i], dims[i + 1], bias=not last))
            )
            if not last:
                layers.append(nn.Softplus(beta=1.0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=-1)
        return self.mlp(pooled).squeeze(-1)


@torch.no_grad()
def refresh_spectral_norm(state: dict, n_iter: int = 1) -> dict:
    for key in [k for k in state if k.endswith("weight_orig")]:
        prefix = key[: -len("orig")]
        W = state[key].flatten(1)
        u = state[prefix + "u"]
        for _ in range(n_iter):
            v = F.normalize(W.t() @ u, dim=0, eps=1e-12)
            u = F.normalize(W @ v, dim=0, eps=1e-12)
        state[prefix + "u"].copy_(u)
        state[prefix + "v"].copy_(v)
    return state


class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, obs_horizon: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(beta=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class EnergyFlowModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        down_dims=(64, 128, 256),
        state_emb_dim: int = 128,
        time_emb_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.state_encoder = StateEncoder(obs_dim, obs_horizon, out_dim=state_emb_dim)
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        cond_dim = state_emb_dim + time_emb_dim
        self.backbone = ConditionalUnet1D(
            action_dim=action_dim,
            cond_dim=cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
        )
        self.energy_head = ScalarEnergyHead(self.backbone.out_dim, hidden_dims=(256, 128))

    def energy(self, action: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_encoder(obs)
        time_emb = self.time_mlp(self.time_pos_emb(t))
        cond = torch.cat([state_emb, time_emb], dim=-1)
        features = self.backbone(action, cond)
        return self.energy_head(features)

    def score(
        self, action: torch.Tensor, obs: torch.Tensor, t: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        if create_graph:
            action = action.requires_grad_(True)
            energy = self.energy(action, obs, t)
            grad = torch.autograd.grad(
                energy.sum(), action, create_graph=True
            )[0]
            return -grad
        with torch.enable_grad():
            action = action.detach().requires_grad_(True)
            energy = self.energy(action, obs, t)
            grad = torch.autograd.grad(energy.sum(), action)[0]
        return -grad.detach()

    def forward(self, action: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(action, obs, t, create_graph=True)
