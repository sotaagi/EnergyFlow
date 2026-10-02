import argparse
import json
from collections import deque

import h5py
import numpy as np
import torch

from model import EnergyFlowModel, refresh_spectral_norm
from diffusion import VESchedule, sample_actions
from data import DEFAULT_OBS_KEYS, Standardizer

DEFAULT_MAX_STEPS = {"lift": 400, "can": 400, "square": 400, "transport": 700, "tool_hang": 700}


def create_env(hdf5_path: str, render: bool = False):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["object-state"], "rgb": []}}
    )
    with h5py.File(hdf5_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta["env_kwargs"]["has_renderer"] = render
    env_meta["env_kwargs"]["has_offscreen_renderer"] = False
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=render, render_offscreen=False
    )


def get_obs(env, obs_keys) -> np.ndarray:
    obs = env.get_observation()
    return np.concatenate([obs[k].astype(np.float32) for k in obs_keys], axis=-1)


def rollout_one(env, model, schedule, standardizer, obs_keys, args, device):
    obs_deque = deque(maxlen=model.obs_horizon)
    first = get_obs(env, obs_keys)
    for _ in range(model.obs_horizon):
        obs_deque.append(first)

    n_steps = 0
    while n_steps < args.max_episode_steps:
        obs = torch.from_numpy(np.stack(obs_deque)).unsqueeze(0).to(device)
        with torch.no_grad():
            action_seq = sample_actions(
                model, schedule, obs, num_steps=args.ode_steps, gamma=args.gamma
            )[0]
        action_seq = standardizer.unnormalize(action_seq).cpu().numpy()

        for a in action_seq[: min(args.exec_horizon, args.max_episode_steps - n_steps)]:
            env.step(a)
            n_steps += 1
            obs_deque.append(get_obs(env, obs_keys))
            if env.is_success()["task"]:
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="HDF5 used for env metadata")
    p.add_argument("--task", required=True)
    p.add_argument("--obs_keys", nargs="+", default=None)
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="environment-step budget (default: per-task, see DEFAULT_MAX_STEPS)")
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--ode_steps", type=int, default=20)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--no_ema", action="store_true", help="evaluate raw instead of EMA weights")
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.max_episode_steps is None:
        args.max_episode_steps = DEFAULT_MAX_STEPS.get(args.task, 400)

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
    obs_keys = args.obs_keys or DEFAULT_OBS_KEYS[args.task]

    env = create_env(args.data, render=args.render)
    successes = []
    for ep in range(args.n_rollouts):
        env.env.hard_reset = False
        env.reset()
        torch.manual_seed(args.seed + ep)
        np.random.seed(args.seed + ep)
        ok = rollout_one(env, model, schedule, standardizer, obs_keys, args, device)
        successes.append(ok)
        print(f"rollout {ep}: {'success' if ok else 'fail'} "
              f"(running {np.mean(successes) * 100:.1f}%)")
    print(f"\nSuccess rate: {np.mean(successes) * 100:.1f}% over {len(successes)} rollouts")


if __name__ == "__main__":
    main()
