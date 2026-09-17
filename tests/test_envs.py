import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

from wm.envs import ControllerTrainingCarRacing
from wm.utils import crop_obs


def test_controller_training_renderer_preserves_model_inputs_and_dynamics() -> None:
    rng = np.random.default_rng(7)
    reference = CarRacing()
    training = ControllerTrainingCarRacing()

    try:
        for seed in (0, 17):
            reference_obs, _ = reference.reset(seed=seed)
            training_obs, _ = training.reset(seed=seed)

            np.testing.assert_array_equal(
                crop_obs(reference_obs), crop_obs(training_obs)
            )

            for _ in range(25):
                action = np.array(
                    [
                        rng.uniform(-1, 1),
                        rng.uniform(0, 1),
                        rng.uniform(0, 1),
                    ],
                    dtype=np.float32,
                )
                reference_step = reference.step(action)
                training_step = training.step(action)

                reference_obs, reference_reward, terminated, truncated, _ = (
                    reference_step
                )
                (
                    training_obs,
                    training_reward,
                    training_terminated,
                    training_truncated,
                    _,
                ) = training_step

                assert reference_reward == training_reward
                assert terminated == training_terminated
                assert truncated == training_truncated
                np.testing.assert_array_equal(reference_obs[:84], training_obs[:84])
                np.testing.assert_array_equal(
                    crop_obs(reference_obs), crop_obs(training_obs)
                )
    finally:
        reference.close()
        training.close()
