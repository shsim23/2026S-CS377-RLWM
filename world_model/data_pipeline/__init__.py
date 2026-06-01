"""Offline data-collection pipeline for the DreamerV3-style world model.

Produces the fixed, mixed-policy, multi-layout transition dataset described in
`WORLDMODEL_DREAMERV3_SPEC.md` §3. This package is independent of the legacy
(v10c / JEPA) world model and is only used by the `scripts/wm_*` entry points.
"""
from .layouts import maze_dict_to_layout_text, write_layout
from .policies import RandomPolicy, GreedyBFSPolicy, EpsilonGreedy, make_policy_pool

__all__ = [
    "maze_dict_to_layout_text",
    "write_layout",
    "RandomPolicy",
    "GreedyBFSPolicy",
    "EpsilonGreedy",
    "make_policy_pool",
]
