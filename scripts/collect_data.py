from pathlib import Path

import gymnasium as gym
import numpy as np
import tyro
import zarr
from flax import nnx
from tqdm import tqdm, trange

from wm import Agent
from wm.initializer import cauchy_initializer
from wm.utils import normalise_obs, prep_obs

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_ID = "CarRacing-v3"
NUM_ROLLOUTS = 10_000


def create_or_read_datastore(data_path: Path) -> tuple[zarr.Array, zarr.Array]:
    if data_path.exists():
        print(f"Opening existing datastore at {data_path}")
        z = zarr.open_group(store=data_path)
        obs: zarr.Array = z["obs"]  # type:ignore
        act: zarr.Array = z["act"]  # type:ignore
    else:
        print(f"Creating new datastore at {data_path}")
        z = zarr.create_group(store=data_path, overwrite=True)
        obs = z.create_array(
            "obs",
            shape=(0, 1001, 64, 64, 3),
            chunks=(1, 1001, 64, 64, 3),
            dtype=np.uint8,
        )
        act = z.create_array(
            "act", shape=(0, 1000, 3), chunks=(1000, 1000, 3), dtype=np.float32
        )

    return obs, act


# Continuously loop until we've collected enough rollouts:
#   1. Instantiate a new agent with a random seed
#   2. Roll it out across a series of batched episodes
#   3. Save the data.
#   4. Repeat
def collect_rollout(
    seed: int, envs: gym.vector.VectorEnv
) -> tuple[np.ndarray, np.ndarray]:
    init_stddev = 0.01 * np.random.uniform(0, 1)
    agent = Agent(rngs=nnx.Rngs(seed), kernel_init=cauchy_initializer(init_stddev))
    states, _ = envs.reset(seed=seed)
    o = prep_obs(states)
    # Instantiate data containers for a give rollout - which we'll then append to our
    # persistent data store
    num_envs = envs.num_envs
    rollout_obs = np.zeros(shape=(num_envs, 1001, 64, 64, 3), dtype=np.uint8)
    rollout_act = np.zeros(shape=(num_envs, 1000, 3), dtype=np.float32)
    rollout_obs[:, 0] = o

    carry = agent.initialize_carry(num_envs)

    for t in trange(1000, desc="Steps", position=1, leave=False):
        a, carry = agent(normalise_obs(o), carry)  # type: ignore
        states, _rewards, _truncateds, _terminateds, _infos = envs.step(np.array(a))
        o = prep_obs(states)

        # We shift the observation by one (recording 1,001) so we
        # can store both the initial and final observation
        rollout_obs[:, t + 1] = o
        rollout_act[:, t] = a
    return rollout_obs, rollout_act


def main(num_envs: int = 16, run_name: str = "vae"):
    print(f"Data collection run `{run_name}`\n  num_envs: {num_envs}")

    # Load or create the datastore we'll save to
    data_path = ROOT_DIR / "data" / run_name
    obs, act = create_or_read_datastore(data_path)

    start_ix = obs.shape[0]
    envs = gym.make_vec(id=ENV_ID, num_envs=num_envs, vectorization_mode="async")

    try:
        with tqdm(
            total=NUM_ROLLOUTS,
            initial=start_ix,
            desc="Rollouts",
            position=0,
        ) as outer:
            for rollout in range(start_ix, NUM_ROLLOUTS, num_envs):
                # gymnasium distributes seeds across vectorised environments like (s, s+1, s+2, ...).
                # I don't want to reuse seeds and reduce stochasticity, so multiply the seed to prevent reuse
                seed = rollout * 250
                rollout_obs, rollout_act = collect_rollout(seed, envs)

                # Save the rollouts to the persistent stores
                obs.append(rollout_obs)
                act.append(rollout_act)

                outer.update(num_envs)
    finally:
        envs.close()


if __name__ == "__main__":
    tyro.cli(main)
