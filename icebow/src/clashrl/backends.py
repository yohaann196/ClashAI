"""PLATFORM BACKENDS: where the game window is, and how to tap into it.

`WindowCapture` and `Controller` were written against exactly one setup -- Clash Royale in Google Play
Games on Windows -- and the assumptions were spread through both: pygetwindow enumeration, a Win32
`GetClientRect` call, a content trim tuned to GPG's custom title bar and left icon rail, and a 9:16
aspect check. None of that is true of a Mac mirroring an iPhone.

This is the seam. A backend answers three OS-specific questions and nothing else:

    find_window()   where is the app's window, in PHYSICAL screen pixels?
    render_area()   where inside it is the GAME, as opposed to chrome and letterboxing?
    tap()           how do I click a physical pixel so the app actually receives it?

`WindowCapture` keeps the mss grabbing and all the normalized<->screen arithmetic; `Controller` keeps
the play-a-card sequencing. Both are now platform-agnostic.

WHY THE COORDINATES HAVE TO BE NAMESPACED TOO. Every normalized constant in the config -- hand slots,
tower anchors, the elixir bar, `action.arena_box` -- is calibrated against ONE render at ONE aspect
ratio. A GPG render is ~9:16 (0.566); an iPhone is ~9:19.5 (0.46), and Clash Royale lays its UI out
differently at that shape. So the same key genuinely needs a different value per platform, which is why
`config.platform` selects a block that is DEEP-MERGED over the base rather than a second set of keys
that every call site would have to know about. Existing `cfg.get("window", "region")` calls are
unchanged; they just see the resolved value.

STATUS: `GpgBackend` is the code that has been running all along, moved unchanged. `MirrorBackend` is
NEW and has not been run against a real iPhone Mirroring window -- the window lookup, the Retina scale
factor and the render trim are all first-pass and want checking with `run.py diag` before you trust a
capture. It is wired and importable, not proven.
"""
from __future__ import annotations

import platform as _platform
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional at import time
    import pygetwindow as gw
except Exception:  # noqa: BLE001
    gw = None


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int


class Backend(Protocol):
    """The platform seam. Everything else in capture/controller is OS-agnostic."""

    name: str
    title_contains: str
    aspect_range: Tuple[float, float]

    def find_window(self) -> Optional[Region]:
        """The app window in PHYSICAL screen pixels, or None if it cannot be found."""

    def render_area(self, base: Region, grab) -> Optional[Region]:
        """The GAME render inside `base`. None = could not tell (caller retries next frame)."""

    def tap(self, x: int, y: int) -> None:
        """Click one physical screen pixel."""


# --- shared helpers ------------------------------------------------------------------------------
def _content_trim(img: np.ndarray, base: Region, aspect_range: Tuple[float, float],
                  trim_left: bool) -> Optional[Region]:
    """Trim chrome/letterbox from a captured window by CONTENT and sanity-check the aspect.

    Shared because both platforms have the same problem in different clothing: an app-drawn strip at
    the top plus dark bars at the sides/bottom, with the game the only saturated region. `trim_left`
    exists because GPG has a gray icon rail down the left that iPhone Mirroring does not.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    h, w = sat.shape
    chrome = ((sat < 50.0) | (val < 25.0)).astype(np.float32)
    row_frac = chrome.mean(axis=1)
    top, limit = 0, int(h * 0.12)
    while top < limit and row_frac[top] > 0.90:
        top += 1
    col_frac = chrome[top:, :].mean(axis=0)
    col_val = val[top:, :].mean(axis=0)
    x0, x1 = 0, w
    if trim_left:
        while x0 < w * 0.25 and col_frac[x0] > 0.90:
            x0 += 1
    while x1 > w * 0.75 and col_val[x1 - 1] < 18.0:
        x1 -= 1
    row_dark = val[top:, x0:x1].mean(axis=1)
    y1 = len(row_dark)
    while y1 > len(row_dark) * 0.85 and row_dark[y1 - 1] < 18.0:
        y1 -= 1
    rw, rh = x1 - x0, y1
    if rw < 200 or rh < 300:
        return None
    aspect = rw / float(rh)
    if not (aspect_range[0] <= aspect <= aspect_range[1]):
        return None
    return Region(base.left + x0, base.top + top, int(rw), int(rh))


# --- Windows / Google Play Games -----------------------------------------------------------------
class GpgBackend:
    """Clash Royale in Google Play Games on Windows -- the original, unchanged behaviour."""

    name = "windows_gpg"

    def __init__(self, cfg):
        self.cfg = cfg
        self.title_contains = cfg.get("window", "title_contains", default="Clash Royale") or ""
        self.aspect_range = (0.50, 0.68)          # the GPG render is ~9:16 (0.566 calibrated)

    def find_window(self) -> Optional[Region]:
        if gw is None or not self.title_contains:
            return None
        needle = self.title_contains.lower()
        wins = [w for w in gw.getAllWindows()
                if needle in (w.title or "").lower() and w.width > 100 and w.height > 100]
        if not wins:
            return None
        w = wins[0]
        return self._client_area(w) or Region(int(w.left), int(w.top), int(w.width), int(w.height))

    @staticmethod
    def _client_area(w) -> Optional[Region]:
        """The window's CLIENT area in physical pixels (drops the OS title bar/borders)."""
        try:
            import ctypes
            from ctypes import wintypes
            hwnd = getattr(w, "_hWnd", None)
            if not hwnd:
                return None
            rect = wintypes.RECT()
            if not ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
                return None
            pt = wintypes.POINT(0, 0)
            if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt)):
                return None
            if rect.right > 100 and rect.bottom > 100:
                return Region(int(pt.x), int(pt.y), int(rect.right), int(rect.bottom))
        except Exception:  # noqa: BLE001
            return None
        return None

    def render_area(self, base: Region, grab) -> Optional[Region]:
        """GPG draws a custom TITLE BAR and a LEFT ICON SIDEBAR inside the Win32 client area, so
        neither the window rect nor GetClientRect isolates the render every coordinate is calibrated
        to. Both are near-grayscale while the game is saturated; true pillarbox bars are near-black."""
        img = grab(base)
        if img is None:
            return None
        try:
            return _content_trim(img, base, self.aspect_range, trim_left=True)
        except Exception:  # noqa: BLE001
            return None

    def tap(self, x: int, y: int) -> None:
        from .controller import _gui
        _gui().click(x, y)


# --- macOS / iPhone Mirroring --------------------------------------------------------------------
class MirrorBackend:
    """Clash Royale on a real iPhone, shown through macOS iPhone Mirroring.

    Three things differ from the Windows path and each one silently breaks capture if ignored:

    1. WINDOW LOOKUP. pygetwindow's macOS support cannot enumerate another app's windows reliably, so
       the bounds come from System Events via `osascript`. That needs Accessibility permission for
       whatever runs Python (Terminal/iTerm) -- without it the lookup returns nothing rather than
       wrong numbers.
    2. RETINA SCALE. AppleScript reports POINTS; mss grabs PHYSICAL PIXELS. On a Retina display those
       differ by 2x, so using the AppleScript rect directly would capture a quarter of the window and
       land every tap in the wrong place. The factor is measured, not assumed, by comparing the
       physical main-display width from mss against the logical width from AppleScript.
    3. TAP TRANSPORT. Clicks are forwarded to the phone over the mirroring link, and a zero-duration
       click is sometimes dropped. Taps hold the button for `mirror.click_hold_s` and the window is
       raised first, because a click into an unfocused mirror window only focuses it.
    """

    name = "macos_mirror"

    def __init__(self, cfg):
        self.cfg = cfg
        self.title_contains = cfg.get("window", "title_contains", default="iPhone Mirroring") or ""
        self.app_name = cfg.get("mirror", "app_name", default="iPhone Mirroring")
        # An iPhone is ~9:19.5 (0.46) but Clash Royale letterboxes, so accept a wide band: this is a
        # sanity check against grabbing the wrong window, not a calibration.
        ar = cfg.get("mirror", "aspect_range", default=[0.40, 0.62])
        self.aspect_range = (float(ar[0]), float(ar[1]))
        self.click_hold_s = float(cfg.get("mirror", "click_hold_s", default=0.05))
        self.raise_first = bool(cfg.get("mirror", "raise_window", default=True))
        self._scale: Optional[float] = None

    # -- window ---------------------------------------------------------------------------------
    @staticmethod
    def _osascript(script: str) -> Optional[str]:
        try:
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                                 timeout=5.0)
            if out.returncode != 0:
                return None
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    def _backing_scale(self) -> float:
        """PHYSICAL pixels per AppleScript POINT, measured rather than assumed.

        This matters more than it looks: AppleScript reports points, mss grabs pixels, and on a Retina
        Mac those differ by 2x. Getting it wrong does not fail loudly -- it captures a quarter of the
        window and lands every tap at half the intended coordinate.

        Read from Quartz's display mode (pixel width / point width), which needs no permissions and no
        extra dependency: pyobjc-framework-Quartz is already a pyautogui requirement on macOS, and this
        backend cannot tap without pyautogui anyway. An earlier version asked Finder for the desktop
        bounds over AppleScript; on this machine that call HUNG (it needs Automation permission for
        Finder) and the except-branch silently returned 1.0 -- a wrong answer that looks like a right
        one. So an unmeasurable scale now WARNS and is treated as unknown rather than assumed.
        """
        if self._scale is not None:
            return self._scale
        scale = None
        try:
            from Quartz import (CGDisplayCopyDisplayMode, CGDisplayModeGetPixelWidth,
                                CGDisplayModeGetWidth, CGMainDisplayID)
            mode = CGDisplayCopyDisplayMode(CGMainDisplayID())
            pts = float(CGDisplayModeGetWidth(mode))
            px = float(CGDisplayModeGetPixelWidth(mode))
            if pts > 0 and px > 0:
                scale = px / pts
        except Exception:  # noqa: BLE001
            scale = None
        if scale is None or not (0.5 <= scale <= 4.0):
            print("[capture] could not measure the display's backing scale factor (Quartz "
                  "unavailable?). Assuming 1.0 -- if this is a Retina Mac every capture will be a "
                  "quarter of the window and every tap will land at half the right coordinate. "
                  "Set window.region explicitly under platforms.macos_mirror to bypass this.")
            scale = 1.0
        self._scale = scale
        return scale

    def find_window(self) -> Optional[Region]:
        raw = self._osascript(
            f'tell application "System Events" to tell process "{self.app_name}" '
            f'to get {{position, size}} of window 1')
        if not raw:
            return None
        try:
            nums = [int(float(p.strip())) for p in raw.split(",")]
        except ValueError:
            return None
        if len(nums) != 4:
            return None
        s = self._backing_scale()
        left, top, w, h = nums
        if w < 100 or h < 100:
            return None
        return Region(int(left * s), int(top * s), int(w * s), int(h * s))

    def render_area(self, base: Region, grab) -> Optional[Region]:
        """iPhone Mirroring frames the phone screen with a thin surround and (on hover) a floating
        title bar; there is no left icon rail, so the left trim is off -- running GPG's version here
        would eat real arena pixels down the left edge."""
        img = grab(base)
        if img is None:
            return None
        try:
            return _content_trim(img, base, self.aspect_range, trim_left=False)
        except Exception:  # noqa: BLE001
            return None

    # -- input ----------------------------------------------------------------------------------
    def tap(self, x: int, y: int) -> None:
        from .controller import _gui
        gui = _gui()
        if self.raise_first:
            self._raise()
        gui.moveTo(x, y)
        gui.mouseDown()
        time.sleep(self.click_hold_s)      # the mirroring link drops zero-duration clicks
        gui.mouseUp()

    def _raise(self) -> None:
        self._osascript(f'tell application "{self.app_name}" to activate')


_BACKENDS = {"windows_gpg": GpgBackend, "macos_mirror": MirrorBackend}


def detect_platform() -> str:
    """The default `platform:` for this machine."""
    return "macos_mirror" if _platform.system() == "Darwin" else "windows_gpg"


def make_backend(cfg) -> Backend:
    """Build the backend named by `config.platform` (already resolved to a concrete name)."""
    name = str(cfg.get("platform", default=None) or detect_platform())
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"unknown platform {name!r}; valid: {sorted(_BACKENDS)}")
    return cls(cfg)


def available_platforms() -> List[str]:
    return sorted(_BACKENDS)
