"""Input automation: play a card by tapping the game window.

The SEQUENCE (tap the hand slot, pause, tap the placement) is the same everywhere -- it is how Clash
Royale works, not how any one host works. HOW a tap is delivered is not: a Windows GPG window takes a
plain instant click, while iPhone Mirroring forwards clicks over a link that drops zero-duration ones
and needs the window frontmost first. That difference lives in the backend
(:class:`clashrl.backends.Backend`), so this module is only the sequencing.

Normalized coordinates are resolved through the capture's region, so taps and captured frames share
one pixel space on every platform.
"""
from __future__ import annotations

import time

# pyautogui is imported LAZILY (here and in the backends): it needs a display, so importing it at
# module scope makes `clashrl.controller` unimportable on a headless box -- CI, a remote shell, or a
# machine that only ever runs train-sim. Nothing that merely constructs a Controller needs input.
_PYAUTOGUI = None


def _gui():
    global _PYAUTOGUI
    if _PYAUTOGUI is None:
        import pyautogui
        pyautogui.FAILSAFE = True   # slam mouse into a screen corner to abort
        pyautogui.PAUSE = 0.0
        _PYAUTOGUI = pyautogui
    return _PYAUTOGUI


class Controller:
    def __init__(self, capture, cfg, backend=None):
        self.capture = capture
        self.cfg = cfg
        # Share the capture's backend by default: tapping and capturing MUST agree about which window
        # they are talking to, and two independently-built backends could disagree after a re-detect.
        self.backend = backend or getattr(capture, "backend", None)

    def _tap_screen(self, x: int, y: int) -> None:
        if self.backend is not None:
            self.backend.tap(x, y)
        else:
            _gui().click(x, y)

    def tap(self, nx: float, ny: float) -> None:
        self._tap_screen(*self.capture.to_screen(nx, ny))

    def play_card(self, slot_nx: float, slot_ny: float, target_nx: float, target_ny: float) -> None:
        """Tap-select a hand card, then tap-place it on the target."""
        sx, sy = self.capture.to_screen(slot_nx, slot_ny)
        tx, ty = self.capture.to_screen(target_nx, target_ny)
        self._tap_screen(sx, sy)
        time.sleep(self.cfg.get("play", "select_delay", default=0.08))
        self._tap_screen(tx, ty)
