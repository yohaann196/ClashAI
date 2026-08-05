"""Configuration loading with nested-key access, project-relative paths, and PLATFORM RESOLUTION.

THE PROBLEM PLATFORM RESOLUTION SOLVES. Almost every spatial constant in this config -- `window.region`,
the tower anchors in `env.my_towers` / `env.enemy_towers`, `action.arena_box`, the hand slots, the
elixir bar, the state templates -- is calibrated against ONE render at ONE aspect ratio. A Google Play
Games window is ~9:16 (0.566); an iPhone through macOS Mirroring is ~9:19.5 (0.46), and Clash Royale
lays its UI out differently at that shape. The same key genuinely needs a different value per host, and
before this the only way to switch was to overwrite the file and lose the other calibration.

    platform: auto            # auto | windows_gpg | macos_mirror
    platforms:
      windows_gpg: {window: {...}, env: {...}, action: {...}}
      macos_mirror: {window: {...}, env: {...}, action: {...}}

At load time the selected block is DEEP-MERGED over the base config and `platform` is rewritten to the
resolved name. That choice matters: every existing `cfg.get("window", "region")` call site keeps
working untouched and simply sees the right value, instead of ~20 call sites each having to learn about
platforms. `auto` picks from the host OS (Darwin -> macos_mirror, else windows_gpg).

Calibrations therefore live SIDE BY SIDE. Switching hosts is a one-line change and does not destroy the
other machine's numbers -- which is the whole point, since re-deriving them means re-running the
calibration tools.
"""
from __future__ import annotations

import os
import platform as _platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def detect_platform() -> str:
    """The default `platform:` for this host."""
    return "macos_mirror" if _platform.system() == "Darwin" else "windows_gpg"


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay `over` onto `base` (returns a new dict; inputs untouched).

    Nested rather than top-level replacement so a platform block can override `window.region` alone
    without having to restate every other `window.*` key -- a platform block is a DIFF, not a
    replacement config.
    """
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_platform(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the selected `platforms.<name>` block over the base config and pin `platform` to the
    resolved name (so downstream code never has to re-run the auto-detection and possibly disagree)."""
    want = str(data.get("platform") or "auto").strip().lower()
    if want == "auto":
        want = detect_platform()
    blocks = data.get("platforms") or {}
    if blocks and want not in blocks:
        print(f"[config] platform {want!r} has no `platforms.{want}` block "
              f"(have: {', '.join(sorted(blocks))}) -- using the base config unchanged.")
    merged = _deep_merge(data, blocks.get(want, {}))
    merged["platform"] = want
    return merged


@dataclass
class Config:
    data: Dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: Optional[str | os.PathLike] = None) -> "Config":
        # project root is …/icebow  (this file lives at src/clashrl/config.py)
        root = Path(__file__).resolve().parents[2]
        cfg_path = Path(path) if path else root / "config" / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data=resolve_platform(data), root=root)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Fetch a nested value, e.g. cfg.get("record", "fps")."""
        node: Any = self.data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        return self.root.joinpath(*parts)

    @property
    def platform(self) -> str:
        return str(self.data.get("platform") or detect_platform())
