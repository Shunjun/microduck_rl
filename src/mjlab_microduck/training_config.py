"""Project defaults applied to every Microduck task before CLI overrides."""

from dataclasses import replace
from typing import Literal


DEFAULT_LOGGER: Literal["tensorboard", "wandb"] = "tensorboard"


def register_microduck_task(*, rl_cfg, **kwargs) -> None:
    from mjlab.tasks.registry import register_mjlab_task

    register_mjlab_task(
        rl_cfg=replace(rl_cfg, logger=DEFAULT_LOGGER),
        **kwargs,
    )
