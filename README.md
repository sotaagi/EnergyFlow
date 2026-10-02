# EnergyFlow

[![arXiv](https://img.shields.io/badge/arXiv-2605.00623-b31b1b.svg)](https://arxiv.org/abs/2605.00623)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)

Official implementation of **"Recovering Hidden Reward in Diffusion-Based Policies"**.

EnergyFlow is a diffusion-style policy whose denoising field is the gradient of a
learned scalar energy. The single denoising score matching objective
gives both

- an **action generator**: actions are sampled by integrating the
  probability-flow ODE, as in a standard diffusion policy, and
- a **reward model**: the energy recovers the soft Q-function up to a
  state-dependent offset, with no adversarial training.


## Method Components

| Component | Definition | Code |
|---|---|---|
| Energy / score | `E_phi(a, s, t)` scalar; `S_phi = -grad_a E_phi` via autograd | `model.py` |
| Noise schedule | VE, `sigma(t) = sigma_min^(1-t/T) * sigma_max^(t/T)`, `sigma_min=0.01`, `sigma_max=10`, `T=1` | `diffusion.py` |
| Training loss | `E[ sigma(t)^2 * ‖S_phi(a_t, s, t) + eps / sigma(t)‖^2 ]`, `a_t = a_0 + sigma(t) * eps` | `diffusion.dsm_loss` |
| Sampling | Euler integration of the probability-flow ODE from `t=T` to `t=gamma=1e-3`, `K=20` steps | `diffusion.sample_actions` |
| Reward | `r(a, s) = -(E(a, s, gamma) - mean_{a' ~ N(0, I)} E(a', s, gamma))` with 16 fixed reference actions | `rewards.CenteredReward` |


## Installation

```bash
conda create -n energyflow python=3.10 -y && conda activate energyflow
pip install -r requirements.txt
```

`torch`, `numpy`, `h5py` and `tqdm` are enough for training. Rollout
evaluation needs `robomimic` + `robosuite` (tested with robomimic 0.5.0 and
robosuite 1.5.1, which match the `*_v15.hdf5` datasets) or `metaworld`.

## Data

**RoboMimic** (Lift, Can, Square, Transport, ToolHang; proficient-human, low-dim):

```bash
bash download_robomimic.sh data/robomimic
# -> data/robomimic/<task>/ph/low_dim_v15.hdf5
```

**Meta-World** (ButtonPress, DrawerOpen, Assembly, BinPicking, Hammer).
Meta-World has no demonstration datasets, so we roll out its scripted expert
policies:

```bash
python collect_metaworld_demos.py \
    --tasks button_press drawer_open assembly bin_picking hammer \
    --num_episodes 200 --out_dir data/metaworld
```

Actions are standardized to zero mean and unit variance with statistics from
the training demonstrations. The statistics are saved in each checkpoint.

## Training

```bash
python train.py --dataset robomimic --task can \
    --data data/robomimic/can/ph/low_dim_v15.hdf5

python train.py --dataset metaworld --task button_press \
    --data data/metaworld/button_press.npz
```


## Evaluation

```bash
python eval_robomimic.py --checkpoint outputs/robomimic_can/final.pt \
    --data data/robomimic/can/ph/low_dim_v15.hdf5 --task can --n_rollouts 50

python eval_metaworld.py --checkpoint outputs/metaworld_button_press/final.pt \
    --task button_press --n_rollouts 50
```

## Reward extraction

```python
import torch
from model import EnergyFlowModel, refresh_spectral_norm
from rewards import CenteredReward

ckpt = torch.load("outputs/robomimic_can/final.pt", weights_only=False)
cfg = ckpt["config"]
model = EnergyFlowModel(
    obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
    obs_horizon=cfg["obs_horizon"], pred_horizon=cfg["pred_horizon"],
    down_dims=tuple(cfg["down_dims"]), kernel_size=cfg["kernel_size"],
)
model.load_state_dict(refresh_spectral_norm(ckpt["ema"]["shadow"], n_iter=50))
model.eval()

reward_fn = CenteredReward(model, num_baseline_samples=16, gamma=1e-3)
# obs: [B, To, Do]; action: [B, Tp, Da] in standardized action space
r = reward_fn(obs, action)  # [B]
```


## Results

Success rates (%) reported in the paper.

| RoboMimic (ph) | Lift | Can | Square | Transport | ToolHang | Avg. |
|---|---|---|---|---|---|---|
| EnergyFlow | 100.0±0.0 | 100.0±0.0 | 95.3±0.5 | 89.4±1.6 | 84.2±1.4 | 93.8 |

| Meta-World | Button | Drawer | Assembly | Bin | Hammer | Avg. |
|---|---|---|---|---|---|---|
| EnergyFlow | 100.0±0.0 | 94.2±1.4 | 82.6±2.8 | 90.9±1.9 | 94.6±1.5 | 92.5 |

See the paper for baselines, the RL experiments, OOD generalization and
ablations.

## Citation

```bibtex
@inproceedings{ji2026recovering,
title={Recovering Hidden Reward in Diffusion-Based Policies},
author={Yanbiao Ji and Qiuchang Li and Yuting Hu and Shaokai Wu and Wenyuan XIE and Guodong ZHANG and Qichen He and Deyi Ji and Yue Ding and Hongtao Lu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=KibOuVwmor}
}
```

## Acknowledgements

The U-Net backbone is adapted from
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
Benchmarks: [RoboMimic](https://github.com/ARISE-Initiative/robomimic) and
[Meta-World](https://github.com/Farama-Foundation/Metaworld).
