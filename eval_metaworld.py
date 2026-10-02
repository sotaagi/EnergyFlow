import argparse
from collections import deque

import numpy as np
import torch

from model import EnergyFlowModel, refresh_spectral_norm
from diffusion import VESchedule, sample_actions
from data import METAWORLD_TASKS, Standardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=500, help="environment-step budget")
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--ode_steps", type=int, default=20)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--no_ema", action="store_true", help="evaluate raw instead of EMA weights")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import metaworld

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = EnergyFlowModel(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        obs_horizon=cfg["obs_horizon"],
        pred_horizon=cfg["pred_horizon"],
        down_dims=tuple(cfg["down_dims"]),
        kernel_size=cfg["kernel_size"],
    ).to(device)
    state = ckpt["model"] if args.no_ema else ckpt["ema"]["shadow"]
    model.load_state_dict(refresh_spectral_norm(state, n_iter=50))
    model.eval()
    schedule = VESchedule(cfg["sigma_min"], cfg["sigma_max"], cfg["time_T"])
    standardizer = Standardizer.from_state_dict(ckpt["action_standardizer"])

    env_name = METAWORLD_TASKS.get(args.task, args.task)
    ml1 = metaworld.ML1(env_name, seed=args.seed)
    env = ml1.train_classes[env_name]()
    tasks = ml1.train_tasks

    successes = []
    for ep in range(args.n_rollouts):
        env.set_task(tasks[ep % len(tasks)])
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_deque = deque([obs.astype(np.float32)] * cfg["obs_horizon"],
                          maxlen=cfg["obs_horizon"])
        success = finished = False
        n_steps = 0
        while n_steps < args.max_episode_steps and not finished:
            obs_t = torch.from_numpy(np.stack(obs_deque)).unsqueeze(0).to(device)
            with torch.no_grad():
                action_seq = sample_actions(
                    model, schedule, obs_t, num_steps=args.ode_steps, gamma=args.gamma
                )[0]
            action_seq = standardizer.unnormalize(action_seq).cpu().numpy()
            for a in action_seq[: min(args.exec_horizon, args.max_episode_steps - n_steps)]:
                result = env.step(a)
                n_steps += 1
                if len(result) == 5:
                    obs, _, done, trunc, info = result
                else:
                    obs, _, done, info = result
                    trunc = False
                obs_deque.append(obs.astype(np.float32))
                success = bool(info.get("success", False))
                finished = success or done or trunc
                if finished:
                    break
        successes.append(success)
        print(f"rollout {ep}: {'success' if success else 'fail'} "
              f"(running {np.mean(successes) * 100:.1f}%)")
    print(f"\nSuccess rate: {np.mean(successes) * 100:.1f}% over {len(successes)} rollouts")


if __name__ == "__main__":
    main()
