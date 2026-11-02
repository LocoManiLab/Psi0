"""AMO fine-tune trainer.

Identical recipe to SonicTrainer — the flow-matching fine-tune over a LeRobot dataset's
`states`/`action` columns is embodiment-agnostic; the action space is set entirely by
`--model.action-dim` / `--model.odim` (amo: 36 / 32, sonic: 78 / 43).

It exists only so a run collected with the AMO whole-body controller is *named* amo:
`Trainer.instantiate` maps `--train.name=amo` to this class, and that name is also the
run-dir segment (`<output_dir>/amo/<exp>.<ts>`) and the wandb group. Without it, amo runs
would land under `.../sonic/` and be indistinguishable from SONIC latent-token runs.

Put anything genuinely amo-specific (e.g. per-dimension loss weights across the mixed
joint-angle / torso-velocity action vector) here — not in sonic.py.
"""
from .sonic import SonicTrainer


class AmoTrainer(SonicTrainer):
    pass
