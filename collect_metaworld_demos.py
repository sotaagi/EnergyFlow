import argparse
from pathlib import Path

import numpy as np

from data import METAWORLD_TASKS


def collect_task(env_name: str, num_episodes: int, max_steps: int, seed: int):
    import metaworld
    import metaworld.policies as policies

    policy_name = "Sawyer" + "".join(
        w.capitalize() for w in env_name.replace("-", "_").split("_")
    ) + "Policy"
    policy_cls = getattr(policies, policy_name)

    ml1 = metaworld.ML1(env_name, seed=seed)
    env = ml1.train_classes[env_name]()
    tasks = ml1.train_tasks

    all_obs, all_actions, lengths = [], [], []
    for ep in range(num_episodes):
        env.set_task(tasks[ep % len(tasks)])
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        policy = policy_cls()
        ep_obs, ep_actions = [obs], []
        for _ in range(max_steps):
            action = policy.get_action(obs)
            result = env.step(action)
            if len(result) == 5:
                obs, _, done, trunc, _ = result
            else:
                obs, _, done, _ = result
                trunc = False
            ep_obs.append(obs)
            ep_actions.append(action)
            if done or trunc:
                break
        all_obs.append(np.stack(ep_obs[:-1]))
        all_actions.append(np.stack(ep_actions))
        lengths.append(len(ep_actions))

    L = max(lengths)
    N, Do = len(all_obs), all_obs[0].shape[-1]
    Da = all_actions[0].shape[-1]
    obs_arr = np.zeros((N, L, Do), dtype=np.float32)
    act_arr = np.zeros((N, L, Da), dtype=np.float32)
    for i, (o, a) in enumerate(zip(all_obs, all_actions)):
        obs_arr[i, : len(o)] = o
        act_arr[i, : len(a)] = a
    return obs_arr, act_arr, np.asarray(lengths, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(METAWORLD_TASKS))
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="data/metaworld")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        env_name = METAWORLD_TASKS.get(task, task)
        obs, actions, lengths = collect_task(
            env_name, args.num_episodes, args.max_steps, args.seed
        )
        path = out_dir / f"{task}.npz"
        np.savez_compressed(path, obs=obs, actions=actions, lengths=lengths)
        succ = (lengths < args.max_steps).mean()
        print(f"{task}: saved {len(lengths)} episodes to {path} "
              f"(mean len {lengths.mean():.0f}, finished-early frac {succ:.2f})")


if __name__ == "__main__":
    main()
