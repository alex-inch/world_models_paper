from os import PathLike

import grain.python as grain
import numpy as np
import zarr
from jaxtyping import Shaped


class VAESource(grain.RandomAccessDataSource):
    def __init__(self, path: str | PathLike):
        z = zarr.open_group(str(path), mode="r")
        self.obs: zarr.Array = z["obs"]  # type: ignore

        shape = self.obs.shape
        self.n_episodes = shape[0]
        self.n_timesteps = shape[1]
        self.n_samples = self.n_episodes * self.n_timesteps

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Shaped[np.ndarray, "H W C"]:
        # We use divmod here as the inverse equivalent of...
        #   ix = ep_ix * num_timesteps + time_ix
        ep_idx, time_idx = divmod(index, self.n_timesteps)
        return np.asarray(self.obs[ep_idx, time_idx])


def get_vae_dataloader(
    path: str | PathLike,
    frac: float = 1.0,
    batch_size: int = 64,
    num_workers: int = 0,
    num_epochs: int = 1,
) -> grain.DataLoader:
    """Returns a grain dataloader for"""
    source = VAESource(path)

    n_records = int(len(source) * frac)  # not ideal, but good enough for this
    sampler = grain.IndexSampler(
        num_records=n_records, shuffle=True, num_epochs=num_epochs, seed=0
    )
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=num_workers,
        operations=[grain.BatchOperation(batch_size, drop_remainder=True)],
    )

    return loader


class RNNSource(grain.RandomAccessDataSource):
    """Random-access source that returns one latent episode at a time."""

    def __init__(self, path: str | PathLike, vae_name: str = "vae"):
        z = zarr.open_group(str(path), mode="r")
        self.acts: zarr.Array = z["act"]  # type: ignore
        self.mus: zarr.Array = z[f"latents/{vae_name}/mu"]  # type: ignore
        self.logvars: zarr.Array = z[f"latents/{vae_name}/logvar"]  # type: ignore

        self.latent_dim = self.mus.shape[2]
        self.action_dim = self.acts.shape[2]

    def __len__(self) -> int:
        return self.acts.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {
            "acts": np.asarray(self.acts[index]),
            "mus": np.asarray(self.mus[index]),
            "logvars": np.asarray(self.logvars[index]),
        }


def get_rnn_dataloader(
    path: str | PathLike,
    vae_name: str = "vae",
    batch_size: int = 32,
    num_workers: int = 0,
    num_epochs: int = 1,
    seed: int = 0,
) -> grain.DataLoader:
    """Return a shuffled Grain loader of complete latent episodes."""
    source = RNNSource(path, vae_name)
    sampler = grain.IndexSampler(
        num_records=len(source), shuffle=True, num_epochs=num_epochs, seed=seed
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=num_workers,
        operations=[grain.BatchOperation(batch_size, drop_remainder=True)],
    )
