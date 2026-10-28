"""Run model forwards and imagination before computing the losses defined on each class."""

from jax.lax import stop_gradient as sg
from jaxtyping import Array, Float

from dreamer.actor_critic import Actor, Critic
from dreamer.imagine import run_imagination_rollout
from dreamer.wm import DreamerBatch, DreamerWM
from dreamer.wm.rssm import RSSMOut


def wm_objective(
    wm: DreamerWM, batch: DreamerBatch, initial_state: tuple[Array, Array]
):
    preds = wm(batch, initial_state)
    losses = wm.loss_fn(preds, batch, wm.cfg)
    loss = (
        wm.cfg.kl_beta * losses.kl_loss
        + losses.continue_loss
        + losses.reward_loss
        + losses.image_loss
    )

    # Return these to avoid having to recompute them later in training steps.
    model_states = RSSMOut(
        hidden=preds.hidden,
        latent_sample=preds.latent_sample,
        prior=preds.latent_prior,
        posterior=preds.latent_posterior,
    )
    return loss, (losses, model_states)


def actor_objective(
    actor: Actor,
    target_critic: Critic,
    wm: DreamerWM,
    initial_states: RSSMOut,  # [B H], [B Z]
    horizon: int,
    key: Array,
):
    rollout, rewards, conts = run_imagination_rollout(
        wm,
        actor,
        target_critic,
        sg(initial_states),
        horizon=horizon,
        key=key,
    )
    losses, targets, weights = actor.loss_fn(
        actor.cfg, rewards, conts, rollout.value, rollout.action_dist
    )
    # Split out the data for the critic to use in its loss function
    critic_data = (sg(rollout.state.joint_latent[:, :-1]), sg(targets), weights)
    # which contains all the terms for posterity
    return losses.loss, (losses, critic_data)


def critic_objective(
    critic: Critic,
    features: Float[Array, "B T D"],
    targets: Float[Array, "B T"],
    weights: Float[Array, "B T"],
):
    values = critic(features)
    return critic.loss_fn(values, targets, weights)
