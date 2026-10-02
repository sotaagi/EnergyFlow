from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), eps)

    @classmethod
    def from_data(cls, data: np.ndarray) -> "Standardizer":
        flat = data.reshape(-1, data.shape[-1])
        return cls(flat.mean(axis=0), flat.std(axis=0))

    @classmethod
    def identity(cls, dim: int) -> "Standardizer":
        return cls(np.zeros(dim), np.ones(dim))

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * std + mean
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state) -> "Standardizer":
        return cls(np.asarray(state["mean"]), np.asarray(state["std"]))


DEFAULT_OBS_KEYS = {
    "lift": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    "can": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    "square": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    "tool_hang": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    "transport": [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
    ],
}


class RobomimicDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        obs_keys: list[str],
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        standardize_actions: bool = True,
        max_demos: int | None = None,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_keys = list(obs_keys)
        self.obs_seqs: list[np.ndarray] = []
        self.action_seqs: list[np.ndarray] = []
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if max_demos is not None:
                demos = demos[:max_demos]
            for demo_key in demos:
                demo = f["data"][demo_key]
                obs = np.concatenate(
                    [demo["obs"][k][:] for k in self.obs_keys], axis=-1
                ).astype(np.float32)
                actions = demo["actions"][:].astype(np.float32)
                n = min(len(obs), len(actions))
                self.obs_seqs.append(obs[:n])
                self.action_seqs.append(actions[:n])
        self.indices = np.array(
            [(ep, t) for ep, a in enumerate(self.action_seqs) for t in range(len(a))],
            dtype=np.int64,
        )
        self.obs_dim = self.obs_seqs[0].shape[-1]
        self.action_dim = self.action_seqs[0].shape[-1]
        if standardize_actions:
            self.action_standardizer = Standardizer.from_data(
                np.concatenate(self.action_seqs, axis=0)
            )
        else:
            self.action_standardizer = Standardizer.identity(self.action_dim)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        ep, t = self.indices[idx]
        obs_seq = self.obs_seqs[ep]
        act_seq = self.action_seqs[ep]
        t0 = t - self.obs_horizon + 1
        obs = obs_seq[max(t0, 0) : t + 1]
        if t0 < 0:
            pad = np.repeat(obs_seq[0:1], -t0, axis=0)
            obs = np.concatenate([pad, obs], axis=0)
        action = act_seq[t : t + self.pred_horizon]
        if len(action) < self.pred_horizon:
            pad = np.repeat(act_seq[-1:], self.pred_horizon - len(action), axis=0)
            action = np.concatenate([action, pad], axis=0)
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(self.action_standardizer.normalize(action)),
        }


def get_robomimic_dataset(
    task: str,
    hdf5_path: str,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    obs_keys: list[str] | None = None,
    **kwargs,
) -> RobomimicDataset:
    if obs_keys is None:
        if task not in DEFAULT_OBS_KEYS:
            raise ValueError(
                f"Unknown task '{task}'; pass obs_keys explicitly. "
                f"Known tasks: {sorted(DEFAULT_OBS_KEYS)}"
            )
        obs_keys = DEFAULT_OBS_KEYS[task]
    return RobomimicDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        **kwargs,
    )


METAWORLD_TASKS = {
    "button_press": "button-press-v2",
    "drawer_open": "drawer-open-v2",
    "assembly": "assembly-v2",
    "bin_picking": "bin-picking-v2",
    "hammer": "hammer-v2",
}


class MetaworldDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        standardize_actions: bool = True,
        max_episodes: int | None = None,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        data_path = Path(data_path)
        if data_path.is_dir():
            npz_files = sorted(data_path.glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz demo files under {data_path}")
        else:
            npz_files = [data_path]
        self.obs_seqs: list[np.ndarray] = []
        self.action_seqs: list[np.ndarray] = []
        for path in npz_files:
            with np.load(path) as data:
                obs, actions = data["obs"], data["actions"]
                lengths = (
                    data["lengths"]
                    if "lengths" in data
                    else np.full(len(obs), obs.shape[1])
                )
            for i in range(len(obs)):
                if max_episodes is not None and len(self.obs_seqs) >= max_episodes:
                    break
                L = int(lengths[i])
                self.obs_seqs.append(obs[i, :L].astype(np.float32))
                self.action_seqs.append(actions[i, :L].astype(np.float32))
        self.indices = np.array(
            [(ep, t) for ep, a in enumerate(self.action_seqs) for t in range(len(a))],
            dtype=np.int64,
        )
        self.obs_dim = self.obs_seqs[0].shape[-1]
        self.action_dim = self.action_seqs[0].shape[-1]
        if standardize_actions:
            self.action_standardizer = Standardizer.from_data(
                np.concatenate(self.action_seqs, axis=0)
            )
        else:
            self.action_standardizer = Standardizer.identity(self.action_dim)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        ep, t = self.indices[idx]
        obs_seq = self.obs_seqs[ep]
        act_seq = self.action_seqs[ep]
        t0 = t - self.obs_horizon + 1
        obs = obs_seq[max(t0, 0) : t + 1]
        if t0 < 0:
            pad = np.repeat(obs_seq[0:1], -t0, axis=0)
            obs = np.concatenate([pad, obs], axis=0)
        action = act_seq[t : t + self.pred_horizon]
        if len(action) < self.pred_horizon:
            pad = np.repeat(act_seq[-1:], self.pred_horizon - len(action), axis=0)
            action = np.concatenate([action, pad], axis=0)
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(self.action_standardizer.normalize(action)),
        }
