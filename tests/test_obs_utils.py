import numpy as np
import pytest

from wm.utils.obs import crop_obs, normalise_obs, unnormalise_obs


def test_normalise_obs_in_bounds():
    arr = np.linspace(0, 255, num=10 * 64 * 64 * 3, endpoint=True)
    arr = arr.reshape(-1, 64, 64, 3)
    arr = arr.astype(np.uint8)
    normalised = normalise_obs(arr)
    assert np.allclose(normalised.min(), 0)
    assert np.allclose(normalised.max(), 1)
    assert normalised.dtype == np.float32


def test_unnormalise_obs_in_bounds():
    arr = np.linspace(0, 1, num=10 * 64 * 64 * 3, endpoint=True)
    arr = arr.reshape(-1, 64, 64, 3)
    arr = arr.astype(np.float32)
    unnormalised = unnormalise_obs(arr)
    assert np.allclose(unnormalised.min(), 0)
    assert np.allclose(unnormalised.max(), 255)
    assert unnormalised.dtype == np.uint8


def test_prep_obs_batched():
    mock_obs = np.zeros(shape=(1, 96, 96, 3))
    assert crop_obs(mock_obs).shape == (1, 64, 64, 3)


def test_prep_obs_unbatched():
    mock_obs = np.zeros(shape=(96, 96, 3))
    assert crop_obs(mock_obs).shape == (64, 64, 3)


def test_prep_obs_raises_wrong_shape():
    with pytest.raises(ValueError):
        # too many
        crop_obs(np.ones(shape=(1, 1, 1, 1, 1)))

    with pytest.raises(ValueError):
        # too few
        crop_obs(np.ones(shape=(1, 1)))
