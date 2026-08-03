"""LIBERO evaluator shim that resets EA-WAM feedback at every episode.

All benchmark logic remains in the original FastWAM evaluator. This file only
adds the episode-boundary reset required by temporal error memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


repo_root = Path(__file__).resolve().parents[2]
libero_dir = repo_root / "experiments" / "libero"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(libero_dir) not in sys.path:
    sys.path.insert(0, str(libero_dir))

from experiments.libero import eval_libero_single as upstream  # noqa: E402


_upstream_run_single_episode = upstream.run_single_episode


def _run_single_episode_with_feedback_reset(*args, **kwargs):
    model = kwargs.get("model")
    if model is None and len(args) >= 4:
        model = args[3]
    if model is not None and hasattr(model, "reset_feedback"):
        model.reset_feedback()
    return _upstream_run_single_episode(*args, **kwargs)


upstream.run_single_episode = _run_single_episode_with_feedback_reset


if __name__ == "__main__":
    # Calling the imported Hydra-decorated entry point makes Hydra search for a
    # Python package named ``configs``. FastWAM intentionally keeps configs as a
    # plain directory, so compose from its absolute path and invoke the original
    # undecorated function instead.
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(repo_root / "configs"),
    ):
        cfg = compose(
            config_name="sim_libero",
            overrides=sys.argv[1:],
        )
    upstream.eval_single_process.__wrapped__(cfg)
