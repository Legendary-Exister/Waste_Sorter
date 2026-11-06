"""Shared configuration and reproducibility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    # ruamel.yaml keeps comments and key order across a load -> edit -> save cycle.
    from ruamel.yaml import YAML as _YAML
    from ruamel.yaml.error import YAMLError as _YAMLError

    _ROUND_TRIP: Any = _YAML(typ="rt")
    _ROUND_TRIP.preserve_quotes = True
except ImportError:
    # Fallback: PyYAML loads fine but cannot preserve comments or key order on save.
    _ROUND_TRIP = None
    import yaml
    from yaml import YAMLError as _YAMLError


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    """Load project settings and validate their ordered class labels."""
    path = ROOT / "config.yaml"
    try:
        if _ROUND_TRIP is not None:
            with path.open(encoding="utf-8") as handle:
                cfg = _ROUND_TRIP.load(handle) or {}
        else:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, _YAMLError) as exc:
        raise RuntimeError(f"Could not load configuration {path}: {exc}") from exc
    if not isinstance(cfg.get("classes"), list) or not cfg["classes"]:
        raise ValueError("config.yaml must define a non-empty classes list")
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    """Write project settings to config.yaml.

    Comments and key order survive a load_config() -> edit -> save_config() cycle
    only when ruamel.yaml is installed; otherwise PyYAML rewrites the file from
    the given dict and those extras are lost. Pass the object returned by
    load_config() to keep them.
    """
    path = ROOT / "config.yaml"
    try:
        if _ROUND_TRIP is not None:
            with path.open("w", encoding="utf-8") as handle:
                _ROUND_TRIP.dump(cfg, handle)
        else:
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except (OSError, _YAMLError) as exc:
        raise RuntimeError(f"Could not save configuration {path}: {exc}") from exc


def get_classes() -> list[str]:
    """Return the configured model output labels in order."""
    return [str(label) for label in load_config()["classes"]]


def set_seed(seed: int) -> None:
    """Set Python and, when installed, PyTorch random seeds."""
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_labels() -> Path:
    """Write the canonical ordered labels file used by model artifacts."""
    path = ROOT / str(load_config()["paths"]["labels"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(get_classes(), indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-labels", action="store_true", help="write models/labels.json")
    args = parser.parse_args()
    try:
        print(write_labels() if args.write_labels else get_classes())
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
