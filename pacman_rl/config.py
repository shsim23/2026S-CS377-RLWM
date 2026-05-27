from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping at top level: {cfg_path}")
    return data


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def make_env_cfg(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    env_cfg = deepcopy(config.get("env", {}))
    for key, value in overrides.items():
        if value is not None:
            env_cfg[key] = value
    if "layout_file" not in env_cfg:
        raise ValueError("Missing env.layout_file in config or --layout override.")
    env_cfg["layout_file"] = str(resolve_path(env_cfg["layout_file"]))
    return env_cfg


def make_train_cfg(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    train_cfg = deepcopy(config.get("train", {}))
    if not train_cfg:
        raise ValueError("Missing train section in config.")
    if run_name is not None:
        train_cfg["run_name"] = run_name
    return train_cfg
