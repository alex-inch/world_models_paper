import cv2
import numpy as np
from jaxtyping import Float32, UInt8


def crop_obs(state: UInt8[np.ndarray, "... H W 3"]) -> UInt8[np.ndarray, "... 64 64 3"]:
    "Crops out the status bar and resizes 96x96 car racing observations to 64x64"
    if state.ndim not in (3, 4):
        raise ValueError(
            f"Expected input to have 3 or 4 dims, got {state.ndim}: shape={state.shape}"
        )
    cropped = state[..., :84, :, :]

    if cropped.ndim == 4:
        return np.array([cv2.resize(i, (64, 64)) for i in cropped])

    return np.array(cv2.resize(cropped, (64, 64)))


def normalise_obs(
    obs: UInt8[np.ndarray, "... 64 64 3"],
) -> Float32[np.ndarray, "... 64 64 3"]:
    "Takes uint8 observations in [0, 255], returns a floating point array bounded between [0, 1]"
    obs = obs.astype(np.float32) / 255
    return np.clip(obs, 0, 1)


def unnormalise_obs(
    obs: Float32[np.ndarray, "... 64 64 3"],
) -> UInt8[np.ndarray, "... 64 64 3"]:
    "Takes float32 observations in [0, 1] and returns a uint8 array bounded between [0, 255]"
    obs = np.clip(obs * 255, 0, 255)
    return obs.astype(np.uint8)
