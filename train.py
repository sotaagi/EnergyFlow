import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EnergyFlowModel, refresh_spectral_norm
from diffusion import VESchedule, dsm_loss
from data import MetaworldDataset, get_robomimic_dataset


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            # Spectral-norm u/v vectors are not averaged; they are re-derived
            # from the averaged weights by power iteration below.
            if k.endswith(("weight_u", "weight_v")):
                continue
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)
        refresh_spectral_norm(self.shadow, n_iter=1)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


def robust_step(optimizer, model: torch.nn.Module, grad_clip: float) -> tuple[bool, float]:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return False, float("nan")
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        return False, float("nan")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return True, float(grad_norm)


def parse_args():
    p = argparse.ArgumentParser(description="Train EnergyFlow")

    p.add_argument("--dataset", choices=["robomimic", "metaworld"], required=True)
    p.add_argument("--task", type=str, required=True,
                   help="e.g. lift/can/square/transport/tool_hang or a Meta-World task")
    p.add_argument("--data", type=str, required=True, help="HDF5 path or .npz/dir")
    p.add_argument("--obs_keys", nargs="+", default=None,
                   help="override default low-dim observation keys (robomimic)")
    p.add_argument("--max_demos", type=int, default=None)

    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--down_dims", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--kernel_size", type=int, default=5)

    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=10.0)
    p.add_argument("--time_T", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def build_dataset(args):
    common = dict(obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon)
    if args.dataset == "robomimic":
        return get_robomimic_dataset(
            task=args.task,
            hdf5_path=args.data,
            obs_keys=args.obs_keys,
            max_demos=args.max_demos,
            **common,
        )
    return MetaworldDataset(data_path=args.data, max_episodes=args.max_demos, **common)


def cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Dataset: {len(dataset)} windows, "
          f"obs_dim={dataset.obs_dim}, action_dim={dataset.action_dim}")

    model = EnergyFlowModel(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")

    schedule = VESchedule(args.sigma_min, args.sigma_max, args.time_T)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ema = EMAModel(model, decay=args.ema_decay)

    out_dir = Path(args.out_dir) / f"{args.dataset}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(step: int, final: bool = False):
        ckpt = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "action_standardizer": dataset.action_standardizer.state_dict(),
            "config": vars(args),
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "step": step,
        }
        name = "final.pt" if final else f"step{step}.pt"
        torch.save(ckpt, out_dir / name)
        torch.save(ckpt, out_dir / "latest.pt")

    step = 0
    n_skipped = 0
    t_start = time.time()
    losses = []
    pbar = tqdm(total=args.max_steps, desc="train")
    while step < args.max_steps:
        for batch in loader:
            lr = cosine_warmup_lr(step, args.warmup_steps, args.max_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            obs = batch["obs"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)

            loss = dsm_loss(model, schedule, obs, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            stepped, grad_norm = robust_step(optimizer, model, args.grad_clip)
            if not stepped:
                n_skipped += 1
                tqdm.write(f"step {step}: non-finite gradient, skipped "
                           f"({n_skipped} total)")
                continue
            ema.update(model)

            losses.append(loss.item())
            step += 1
            pbar.update(1)
            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}", lr=f"{lr:.2e}")
                losses = []
            if step % args.save_every == 0:
                save(step)
            if step >= args.max_steps:
                break

    save(args.max_steps, final=True)
    pbar.close()
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
