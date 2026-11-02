"""HEFT fine-tune trainer.

Sibling of amo.py: the same SonicTrainer recipe (the flow-matching fine-tune is
embodiment-agnostic — the action space comes from `--model.action-dim`/`--model.odim`,
heft: 50 / 29), existing so a run collected with the HEFT motion-tracking controller is
*named* heft: `--train.name=heft` selects this class, and that name is the run-dir
segment (`<output_dir>/heft/<exp>.<ts>`) and the wandb group.

Without it heft runs land under `.../sonic/` next to SONIC latent-token runs — which is
exactly how the 2026-08-05 heft run ended up training on sonic's data.
"""
from .sonic import SonicTrainer


class HeftTrainer(SonicTrainer):
    pass
