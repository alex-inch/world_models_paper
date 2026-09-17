## Script for training the Controller model of David Ha's 2018 World Models paper.
#   Uses evosax for a CMA-ES evolutionary strategy to optimise the parameters of a small single-layer network
#   Note: the notebook I originally ran, with outputs, is `notebooks/train_controller.ipynb`

from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from evosax.algorithms import CMA_ES as ES
from flax import nnx
from jaxtyping import Array, Shaped
from rliable import library as rly
from rliable import metrics
from tqdm import trange

from wm.agent import Agent
from wm.controller import Controller
from wm.envs import CONTROLLER_TRAINING_ENV_SPEC
from wm.rnn import MDNRNN
from wm.utils import crop_obs, load_rnn, load_vae, normalise_obs
from wm.vae import VAE

ENV_ID = "CarRacing-v3"

encode = nnx.jit(VAE.encode)
step_rnn = nnx.jit(MDNRNN.step)


def prepare_obs(
    obs: Shaped[np.ndarray, "Batch H W C"],
) -> Shaped[Array, "Batch 64 64 3"]:
    return jnp.asarray(normalise_obs(crop_obs(obs)))


def make_call_controllers(graphdef):
    # Map each population member's controller parameters over the observation
    # and recurrent state belonging to the same environment.
    @nnx.jit
    @nnx.vmap(in_axes=(0, 0))
    def call_controllers(individual, x):
        model: Controller = nnx.merge(graphdef, individual)
        return model(x)

    return call_controllers


def compute_fitness(
    population,
    population_size,
    envs,
    vae,
    rnn,
    call_controllers,
    num_trials: int = 1,
    base_seed: int = 0,
):
    cum_reward = np.zeros(population_size)
    for trial in trange(num_trials, leave=False):
        carries = rnn.initialize_carry(population_size)
        # Every candidate sees the same track within a trial, so CMA-ES ranks the
        # controllers rather than the luck of their randomly generated track.
        trial_seed = base_seed + trial
        obs, _ = envs.reset(seed=[trial_seed] * population_size)
        active = np.ones(population_size, dtype=bool)
        for _ in range(1000):
            obs = prepare_obs(obs)
            _, h = carries
            latents = encode(vae, obs).z
            x = jnp.concatenate([latents, h], axis=-1)
            actions = call_controllers(population, x)  # vmap over the controllers.

            rnn_in = jnp.concatenate([latents, actions], axis=-1)
            carries = step_rnn(rnn, rnn_in, carries)

            obs, reward, terminated, truncated, _ = envs.step(np.array(actions))
            cum_reward += reward * active
            active &= ~(terminated | truncated)
            if not active.any():
                break

    # Evosax minimises the fitness. We want to maximise the reward,
    # so return the negative cumulative reward
    return -cum_reward / num_trials


def main(
    vae_name: str = "beta1.0",
    num_generations: int = 100,
    population_size: int = 64,
    num_evaluation_envs: int = 64,
    seed: int = 0,
) -> None:
    vae, _ = load_vae(f"experiments/vae/{vae_name}")
    rnn = load_rnn(f"experiments/rnn/{vae_name}")

    controller = Controller(rngs=nnx.Rngs(seed))
    graphdef, state = nnx.split(controller)
    call_controllers = make_call_controllers(graphdef)

    # Schedule how many runs are used to compute the fitness at each level. CMA-ES
    # training is slow as hell, so we'd like to speed it up as much as possible. I
    # have no idea how the original paper did 2000 generations of 16 trials each...
    trial_schedule = {0: 2, 15: 4, 25: 8, 50: 16}
    num_trials = trial_schedule[0]

    training_envs = gym.make_vec(
        CONTROLLER_TRAINING_ENV_SPEC,  # Custom renderer for faster training
        num_envs=population_size,
        vectorization_mode="async",
    )

    # Initialise the evolutionary strategy.
    key, init_key = jax.random.split(jax.random.key(seed), 2)
    es = ES(population_size, solution=state)
    es_params = es.default_params
    es_state = es.init(init_key, state, es_params)

    try:
        # This took ~8hrs on a beefy Macbook Pro.
        for generation in range(num_generations):
            num_trials = trial_schedule.get(generation, num_trials)
            key, key_ask, key_tell = jax.random.split(key, 3)
            population, es_state = es.ask(key_ask, es_state, es_params)
            fitness = compute_fitness(
                population,
                population_size,
                training_envs,
                vae,
                rnn,
                call_controllers,
                num_trials,
                base_seed=generation * 10_000,
            )

            es_state, info = es.tell(key_tell, population, fitness, es_state, es_params)

            print(
                f"Step {generation + 1:>2} ({num_trials} trial(s)): "
                f"Best Reward {-info['best_fitness']:.1f}"
            )
    finally:
        training_envs.close()

    # Evaluate the best agent. Evosax returns the parameters as a flat array,
    # which needs to be restored to the shape expected by NNX.
    _, unravel_agent_state = jax.flatten_util.ravel_pytree(state)  # type: ignore
    best_state = unravel_agent_state(es_state.best_solution)

    evaluation_envs = gym.make_vec(
        ENV_ID,
        num_envs=num_evaluation_envs,
        vectorization_mode="async",
    )

    try:
        model = nnx.merge(graphdef, best_state)
        agent = Agent(rngs=nnx.Rngs(key), vae=vae, rnn=rnn, controller=model)
        carry = agent.initialize_carry(num_envs=num_evaluation_envs)
        obs, _ = evaluation_envs.reset()

        cum_reward = np.zeros(num_evaluation_envs)
        active = np.ones(num_evaluation_envs, dtype=bool)

        for _ in trange(1000, leave=False):
            obs = prepare_obs(obs)

            actions, carry = agent(obs, carry)  # type: ignore
            # Technically we should handle resets - but an agent that terminates from
            #  driving off the map will do crappily anyway
            obs, reward, terminated, truncated, _ = evaluation_envs.step(
                np.array(actions)
            )
            cum_reward += reward * active

    finally:
        evaluation_envs.close()

    print(f"  Mean: {cum_reward.mean():.0f}")
    print(f"Median: {np.median(cum_reward):.0f}")
    print(f"   Max: {cum_reward.max():.0f}")
    print(f"   Min: {cum_reward.min():.0f}")

    print("\n All rewards:")
    all_rewards = cum_reward.round(0).astype(np.int64)
    all_rewards.sort()
    print(all_rewards)

    # The mean/median are quite flaky because the policy is fairly high variance
    # given track randomness, so report the interquartile mean as well.
    scores = cum_reward.reshape(-1, 1)
    score_dict = {"agent": scores}

    point_estimates, confidence_intervals = rly.get_interval_estimates(
        score_dict,
        lambda x: np.array([metrics.aggregate_iqm(x)]),
        reps=50_000,
        confidence_interval_size=0.95,
        random_state=np.random.RandomState(seed),
    )

    iqm = point_estimates["agent"][0]
    ci_lower, ci_upper = confidence_intervals["agent"][:, 0]

    print(f"IQM: {iqm:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    ckpt_dir = Path(f"experiments/controller/{vae_name}").resolve()
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(ckpt_dir, best_state)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
