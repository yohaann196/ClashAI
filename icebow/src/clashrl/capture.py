"""Screen capture and normalized <-> screen coordinate mapping.

Captures the Clash Royale render area. WHERE that area is, and how to find it, is the one thing that
differs between setups (Google Play Games on Windows vs iPhone Mirroring on macOS), so it lives behind
a :class:`clashrl.backends.Backend`. This module keeps what is genuinely shared: the mss grab, the
retry-until-the-render-locks behaviour, and the normalized<->screen arithmetic every calibrated
coordinate in the config depends on.

The region is PHYSICAL pixels; mss makes the process DPI-aware so capture and pyautogui coordinates
share one pixel space. (On macOS the backend converts AppleScript's logical points into physical
pixels for the same reason -- see MirrorBackend.)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import mss
import numpy as np

from .backends import Region, make_backend


class WindowCapture:
    def __init__(self, title_contains: Optional[str], region: Optional[List[int]] = None, cfg=None,
                 backend=None):
        """`cfg` selects the platform backend. It stays OPTIONAL because several call sites
        (`record`, `diag`, `monitor`) construct a capture from loose args; without it the backend is
        detected from the host OS, which is the right default rather than a silent Windows assumption.
        """
        self.title_contains = title_contains
        self._sct = mss.mss()
        if backend is not None:
            self.backend = backend
        else:
            self.backend = make_backend(cfg) if cfg is not None else make_backend(_LooseCfg(title_contains))
        if title_contains:
            self.backend.title_contains = title_contains
        if region is not None and (not isinstance(region, (list, tuple)) or len(region) != 4):
            print(f"[capture] window.region must be 4 numbers [left, top, width, height] or null "
                  f"(got {region!r}) -- ignoring it and auto-detecting the window instead.")
            region = None
        self._explicit = region is not None
        self._region: Optional[Region] = Region(*region) if region else None
        self._render_locked = True
        if self._region is None:
            self.refresh_region()

    # -- region ---------------------------------------------------------------------------------
    def _grab_region(self, r: Region) -> Optional[np.ndarray]:
        """Raw BGR grab of an arbitrary rect -- handed to the backend so its render scan can look at
        pixels without either side owning the other's job."""
        try:
            raw = self._sct.grab({"left": r.left, "top": r.top, "width": r.width, "height": r.height})
            return cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR)
        except Exception:  # noqa: BLE001
            return None

    def refresh_region(self) -> Optional[Region]:
        """Re-locate the game render. An EXPLICIT `window.region` always wins -- it is the calibration
        escape hatch for a setup the content scan cannot handle."""
        if self._explicit:
            return self._region
        base = self.backend.find_window()
        if base is None:
            return self._region
        render = self.backend.render_area(base, self._grab_region)
        # False -> grab() keeps re-scanning: the render can be unfindable for a while (a black loading
        # screen, another window overlapping) and giving up would strand the bot on a stale region.
        self._render_locked = render is not None
        self._region = render or base
        return self._region

    @property
    def region(self) -> Optional[Region]:
        return self._region

    def grab(self) -> Optional[np.ndarray]:
        """Return the captured region as a BGR image, or None if unavailable."""
        if self._region is None:
            self.refresh_region()
        elif not self._explicit and not self._render_locked:
            self.refresh_region()
        if self._region is None:
            return None
        return self._grab_region(self._region)

    # -- coordinates ------------------------------------------------------------------------------
    def to_screen(self, nx: float, ny: float) -> Tuple[int, int]:
        r = self._region
        if r is None:
            raise RuntimeError("Capture region unknown; cannot map coordinates.")
        return int(r.left + nx * r.width), int(r.top + ny * r.height)

    def to_norm(self, sx: int, sy: int) -> Tuple[float, float]:
        """Map an absolute screen pixel to normalized [0..1] within the region."""
        r = self._region
        if r is None:
            raise RuntimeError("Capture region unknown; cannot map coordinates.")
        return (sx - r.left) / r.width, (sy - r.top) / r.height


class _LooseCfg:
    """Minimal cfg stand-in for the call sites that pass a bare window title and no Config."""

    def __init__(self, title_contains: Optional[str]):
        self._title = title_contains

    def get(self, *keys, default=None):
        if keys[:2] == ("window", "title_contains"):
            return self._title or default
        return default
