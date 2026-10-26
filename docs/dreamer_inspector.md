# Interactive Dreamer inspector

Drive the real CarRacing environment while inspecting a frozen checkpoint:

```bash
uv run scripts/drive_dreamer.py \
  --checkpoint experiments/dreamer/checkpoints/lv7symv9/00016000 \
  --backend cpu
```

The window starts paused. Initial inference compiles JAX functions; subsequent
steps reuse them. `--backend mps` uses the optional Metal installation. Playback
defaults to 15 environment steps per second, adjustable with `--fps`.

The four images show the observation before the action, the prior's forecast,
the actual next observation, and the posterior reconstruction. A staged action
has a forecast but no actual outcome until you step. Reward, continuation and
both critics are evaluated on the predicted and observed next states. Model
inputs use exactly the training crop and normalization.

## Controls

| Control | Action |
| --- | --- |
| Arrows | Steering, throttle and brake; release returns that channel to neutral |
| Click/drag actor axes | Set a persistent action to 0.01 precision |
| Space | Play/pause |
| `.` / `5` | Advance one / five environment steps |
| A | Switch between manual control and sampled actor actions |
| I | Toggle continuous open-loop prediction under the same real actions |
| C | Correct the open-loop state from the current real posterior |
| Timeline / `[` / `]` | Inspect recorded transitions without changing the live environment |
| End | Return to a staged action at the live state |
| B | Bookmark the selected state, or the live state when not browsing |
| Backspace | Restore the selected transition's starting state by action replay |
| T | Compute paired TD(lambda) targets for the selected transition |
| E | Toggle the amplified prediction-error image |
| R / N | Reset the same track / start the next seed |
| S | Save the session |
| Esc | Quit and save |

Actor bars show each bounded action distribution's actual **mean and standard
deviation**, rather than the underlying normal's location and scale. Whiskers
are mean +/- one standard deviation, clipped at the action bounds; they are not
confidence intervals. The orange marker is the proposed action, or the action
actually applied when browsing history. Clicking an axis also switches to
manual control and returns to the live state.

Each latent heatmap has one row per categorical factor and one column per
class. Orange outlines mark that distribution's sampled classes. Click a row
to see its entropy, KL divergence, and the prior probability assigned to the
posterior's sampled class. In open-loop mode, prior and posterior have different
histories; the displayed KL measures that divergence, rather than the training
one-step KL.

## Returns and replay

`T` compares the online critic at the selected starting state with two targets:
a lambda return using recorded rewards and observed-state target values, and a
lambda return from an uninterrupted model rollout under **the same recorded
actions**. Each target is split into reward and critic-bootstrap contributions.
The horizon defaults to 15 (`--horizon`), and gamma and lambda come from the
checkpoint. Targets use up to the collected horizon, bootstrapping at a
nonterminal recording cutoff. Termination and truncation both set continuation
to zero, matching this repository's training buffer.

More future frames can make a previously computed target stale; `T` refreshes
it. Human-controlled targets describe those recorded actions and remain
bootstrapped estimates, rather than ground truth for the actor's value function.

Previews and return branches use separate model RNG streams, and do not alter
the real posterior or the actor's next sampled recommendation. Restore uses the
environment seed plus the recorded actions, and checks observations at every
replayed transition. It does not attempt to clone the Box2D simulator. Restoring
retains the abandoned branch as another episode in the export. Both resetting
and restoring pause playback.

## Recordings and headless checks

Sessions save on exit, including all episodes/branches, to a fresh directory
under `experiments/dreamer/interactive/`. Supply `--output PATH` to choose it.
Existing recordings are not overwritten by a new invocation.

The output uses `metadata.json` and `episode_*.npz`, compatible with
`load_recording`, `view_recording`, and the inspection notebook. The export also
includes action standard deviations, action sources, one-step forecasts,
open-loop starting frames, bookmarks, and lambda targets that were explicitly
computed. Point the notebook's `recording_path` at this directory.

A display-free checkpoint smoke test can also produce a screenshot:

```bash
uv run scripts/drive_dreamer.py \
  --checkpoint experiments/dreamer/checkpoints/lv7symv9/00016000 \
  --backend cpu --headless-steps 60 --horizon 15 \
  --output /tmp/dreamer-inspector-smoke \
  --screenshot /tmp/dreamer-inspector.png
```

Headless mode uses sampled actor actions. Its screenshot selects the beginning
of the last available horizon so the paired return comparison is visible.
