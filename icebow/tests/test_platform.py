"""PLATFORM PORTABILITY: device selection (cuda / mps / auto) and the capture/input backends.

Two changes, both about the project having been written for exactly one machine:

  1. `train.device` was cuda-or-cpu in three near-duplicate copies, so a Mac ran the whole pipeline on
     CPU without ever saying an Apple-silicon GPU was idle.
  2. `WindowCapture` and `Controller` hard-coded Windows + Google Play Games -- pygetwindow, a Win32
     GetClientRect, and a content trim tuned to GPG's chrome.

What is testable HERE is the selection logic, the config resolution and the coordinate arithmetic. What
is NOT is `MirrorBackend` against a real iPhone Mirroring window: the window lookup, the Retina scale
factor and the render trim need the actual app in front of a real display. Those are marked as such
rather than faked with mocks that would only assert my own assumptions back at me.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from clashrl import backends, device
from clashrl.backends import GpgBackend, MirrorBackend, Region, make_backend
from clashrl.config import Config, detect_platform, resolve_platform


class _Cfg:
    """Minimal cfg stub -- these units read config, they do not need a whole Config."""

    def __init__(self, data):
        self.data = data

    def get(self, *keys, default=None):
        node = self.data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


# --- device selection ---------------------------------------------------------------------------
@pytest.fixture()
def torch_stub(monkeypatch):
    """A fake torch whose availability and op-success are dialled per test.

    Real hardware cannot exercise the branches that matter (this machine has no CUDA), and the
    interesting cases are precisely the ones where `is_available()` lies.
    """
    class _Backends:
        class mps:
            _ok = False

            @staticmethod
            def is_available():
                return _Backends.mps._ok

    class _Cuda:
        _ok = False

        @staticmethod
        def is_available():
            return _Cuda._ok

        @staticmethod
        def get_device_name(_i):
            return "FakeGPU"

    class _Torch:
        __version__ = "2.9.0"
        backends = _Backends
        cuda = _Cuda
        broken = set()

        @staticmethod
        def zeros(_n, device="cpu"):
            if device in _Torch.broken:
                raise RuntimeError(f"{device} kernel image is invalid for this device")
            return _Fake()

    class _Fake:
        def __add__(self, _o):
            return self

        def item(self):
            return 1.0

    import sys
    monkeypatch.setitem(sys.modules, "torch", _Torch)
    return _Torch


def _dev(want):
    return _Cfg({"train": {"device": want}})


def test_auto_prefers_cuda_then_mps_then_cpu(torch_stub):
    """The preference order is a throughput ordering: a discrete NVIDIA card beats unified-memory
    Metal, and both beat CPU by enough that silently landing on CPU is a real waste."""
    torch_stub.cuda._ok, torch_stub.backends.mps._ok = True, True
    assert device.pick_device(_dev("auto")) == "cuda"

    torch_stub.cuda._ok = False
    assert device.pick_device(_dev("auto")) == "mps"

    torch_stub.backends.mps._ok = False
    assert device.pick_device(_dev("auto")) == "cpu"


def test_auto_is_the_default_when_the_key_is_absent(torch_stub):
    """The old default was 'cuda', which on a Mac meant a silent CPU run every time."""
    torch_stub.backends.mps._ok = True
    assert device.pick_device(_Cfg({})) == "mps"


def test_a_device_that_reports_available_but_cannot_run_is_rejected(torch_stub):
    """`is_available()` returning True does NOT mean the build can run a kernel -- an RTX 50-series
    card with a pre-Blackwell wheel reports available and then throws on the first op. So every
    candidate is PROBED with a real op, not merely queried."""
    torch_stub.cuda._ok, torch_stub.backends.mps._ok = True, True
    torch_stub.broken = {"cuda"}
    assert device.pick_device(_dev("auto")) == "mps", "auto must skip a broken device, not die on it"

    torch_stub.broken = {"cuda", "mps"}
    assert device.pick_device(_dev("auto")) == "cpu"


def test_an_explicit_unavailable_device_falls_back_and_says_why(torch_stub, capsys):
    """An explicit request that cannot be honoured must explain itself -- silently training on CPU
    when you asked for a GPU is the failure that wastes a day."""
    torch_stub.cuda._ok = False
    assert device.pick_device(_dev("cuda"), "t") == "cpu"
    assert "CUDA is not available" in capsys.readouterr().out

    torch_stub.backends.mps._ok = False
    assert device.pick_device(_dev("mps"), "t") == "cpu"
    assert "Metal is not available" in capsys.readouterr().out


def test_mps_missing_op_failure_points_at_the_fallback_env_var(torch_stub, capsys):
    """Metal does not implement every op and torch RAISES rather than falling back, so the one thing
    a user needs to know is the env var that routes those ops to CPU."""
    torch_stub.backends.mps._ok = True
    torch_stub.broken = {"mps"}
    assert device.pick_device(_dev("mps"), "t") == "cpu"
    assert "PYTORCH_ENABLE_MPS_FALLBACK=1" in capsys.readouterr().out


def test_cpu_is_honoured_without_probing_anything(torch_stub):
    torch_stub.cuda._ok = True
    assert device.pick_device(_dev("cpu")) == "cpu"


def test_an_unknown_device_name_falls_back_to_auto(torch_stub, capsys):
    torch_stub.backends.mps._ok = True
    assert device.pick_device(_dev("gpu"), "t") == "mps"
    assert "not one of" in capsys.readouterr().out


def test_every_entry_point_shares_one_implementation():
    """There were three near-copies; a fix applied to one of them was a fix applied to none."""
    from clashrl.play import _pick_device as play_pick
    from clashrl.train_rl import _pick_device as rl_pick
    import inspect
    for fn in (play_pick, rl_pick):
        assert "pick_device" in inspect.getsource(fn), "must delegate to clashrl.device"
    assert "from .device import" in open("src/clashrl/train_bc.py").read()


# --- platform resolution ------------------------------------------------------------------------
@pytest.fixture()
def raw_cfg():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_the_config_declares_both_platforms(raw_cfg):
    assert set(raw_cfg["platforms"]) == {"windows_gpg", "macos_mirror"}


def test_the_selected_platform_block_overrides_the_base(raw_cfg):
    """The point of namespacing: the same key genuinely needs a different value per host, because
    every spatial constant is calibrated against one render at one aspect ratio."""
    win = resolve_platform({**copy.deepcopy(raw_cfg), "platform": "windows_gpg"})
    mac = resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"})
    assert win["window"]["title_contains"] == "Clash Royale"
    assert mac["window"]["title_contains"] == "iPhone Mirroring"
    assert win["templates"]["root"] != mac["templates"]["root"]


def test_both_calibrations_survive_side_by_side(raw_cfg):
    """Before this, switching hosts meant OVERWRITING the other machine's numbers -- and re-deriving
    them means re-running the calibration tools."""
    for name in ("windows_gpg", "macos_mirror"):
        resolve_platform({**copy.deepcopy(raw_cfg), "platform": name})
    assert raw_cfg["platforms"]["windows_gpg"]["window"]["title_contains"] == "Clash Royale"
    assert raw_cfg["platforms"]["macos_mirror"]["window"]["title_contains"] == "iPhone Mirroring"


def test_the_merge_is_deep_not_a_replacement(raw_cfg):
    """A platform block is a DIFF. Overriding `window.region` must not delete `window.title_contains`,
    or every platform would have to restate the entire config."""
    d = copy.deepcopy(raw_cfg)
    d["platform"] = "macos_mirror"
    d["platforms"]["macos_mirror"]["window"] = {"region": [1, 2, 3, 4]}
    r = resolve_platform(d)
    assert r["window"]["region"] == [1, 2, 3, 4]
    assert "title_contains" in r["window"], "the deep merge must preserve sibling keys"


def test_unrelated_config_survives_resolution(raw_cfg):
    """Resolution must not disturb the shared calibration every platform inherits."""
    r = resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"})
    assert r["env"]["my_towers"] == raw_cfg["env"]["my_towers"]
    assert r["action"]["arena_box"] == raw_cfg["action"]["arena_box"]
    assert r["rewards"] == raw_cfg["rewards"]


def test_auto_resolves_to_a_concrete_platform(raw_cfg):
    """`platform` is pinned to the resolved name so downstream code never re-runs the detection and
    possibly disagrees with the merge that already happened."""
    r = resolve_platform({**copy.deepcopy(raw_cfg), "platform": "auto"})
    assert r["platform"] in ("windows_gpg", "macos_mirror")
    assert r["platform"] == detect_platform()


def test_an_unknown_platform_warns_and_leaves_the_base_intact(raw_cfg, capsys):
    """A typo must not silently produce a half-configured host."""
    r = resolve_platform({**copy.deepcopy(raw_cfg), "platform": "linux_wine"})
    assert "no `platforms.linux_wine` block" in capsys.readouterr().out
    assert r["window"]["title_contains"] == raw_cfg["window"]["title_contains"]


def test_loading_the_real_config_resolves_a_platform():
    cfg = Config.load("config/config.yaml")
    assert cfg.platform in ("windows_gpg", "macos_mirror")
    assert cfg.get("templates", "root")


# --- backends -------------------------------------------------------------------------------------
def test_the_backend_follows_the_resolved_platform(raw_cfg):
    assert isinstance(make_backend(_Cfg(resolve_platform(
        {**copy.deepcopy(raw_cfg), "platform": "windows_gpg"}))), GpgBackend)
    assert isinstance(make_backend(_Cfg(resolve_platform(
        {**copy.deepcopy(raw_cfg), "platform": "macos_mirror"}))), MirrorBackend)


def test_an_unknown_backend_is_a_hard_error():
    """Falling back to a default backend would capture the wrong window and read garbage."""
    with pytest.raises(ValueError, match="unknown platform"):
        make_backend(_Cfg({"platform": "playstation"}))


def test_both_backends_satisfy_the_protocol(raw_cfg):
    """The seam is only useful if both sides actually implement it."""
    cfg = _Cfg(resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"}))
    for be in (GpgBackend(cfg), MirrorBackend(cfg)):
        for attr in ("name", "title_contains", "aspect_range", "find_window", "render_area", "tap"):
            assert hasattr(be, attr), f"{type(be).__name__} is missing {attr}"


def test_the_two_backends_expect_different_render_shapes(raw_cfg):
    """The reason a shared aspect check could not work: a GPG render is ~9:16 (0.566) and an iPhone
    is ~9:19.5 (0.46). One range cannot sanity-check both without accepting the wrong window."""
    cfg = _Cfg(resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"}))
    gpg, mirror = GpgBackend(cfg).aspect_range, MirrorBackend(cfg).aspect_range
    assert gpg != mirror
    assert mirror[0] < gpg[0], "the iPhone render is taller/narrower than a GPG window"


def test_content_trim_finds_a_letterboxed_render():
    """The shared trim: a saturated game region inside dark bars, checked against the aspect."""
    import numpy as np
    img = np.zeros((400, 300, 3), np.uint8)
    img[20:380, 40:240] = (30, 200, 40)                 # saturated 200x360 -> aspect 0.555
    got = backends._content_trim(img, Region(0, 0, 300, 400), (0.50, 0.68), trim_left=True)
    assert got is not None
    assert abs(got.width / got.height - 200 / 360) < 0.05


def test_content_trim_rejects_a_wrong_shaped_window():
    """The aspect check is a guard against grabbing the WRONG window entirely -- a browser, the
    desktop -- which would otherwise be captured and silently fed to the policy."""
    import numpy as np
    wide = np.zeros((400, 900, 3), np.uint8)
    wide[10:390, 10:890] = (30, 200, 40)                # far too wide to be a phone/GPG render
    assert backends._content_trim(wide, Region(0, 0, 900, 400), (0.50, 0.68), trim_left=True) is None


def test_gpg_left_trim_is_not_applied_on_the_mirror_backend():
    """GPG has a gray icon rail down the left; iPhone Mirroring does not. Running GPG's trim there
    would eat real arena pixels off the left edge of every frame."""
    import numpy as np
    img = np.zeros((900, 600, 3), np.uint8)             # big enough that BOTH variants clear the
    img[20:880, 0:120] = (10, 10, 10)                   # 200px minimum width guard
    img[20:880, 120:520] = (30, 200, 40)                # dark rail that only GPG should trim
    with_trim = backends._content_trim(img, Region(0, 0, 600, 900), (0.40, 0.70), trim_left=True)
    without = backends._content_trim(img, Region(0, 0, 600, 900), (0.40, 0.70), trim_left=False)
    assert with_trim is not None and without is not None
    assert without.left < with_trim.left, "trim_left=False must keep the left-hand pixels"


# --- capture arithmetic (platform-agnostic, so it IS testable here) -------------------------------
class _FakeBackend:
    name = "fake"
    title_contains = "x"
    aspect_range = (0.4, 0.7)

    def __init__(self, window=None, render=None):
        self._window, self._render = window, render
        self.taps = []

    def find_window(self):
        return self._window

    def render_area(self, base, grab):
        return self._render

    def tap(self, x, y):
        self.taps.append((x, y))


def _capture(backend):
    from clashrl.capture import WindowCapture
    return WindowCapture(None, None, cfg=_Cfg({}), backend=backend)


def test_capture_uses_the_backend_render_area_when_it_locks():
    cap = _capture(_FakeBackend(Region(0, 0, 400, 800), Region(10, 20, 200, 360)))
    assert (cap.region.left, cap.region.top, cap.region.width) == (10, 20, 200)


def test_capture_falls_back_to_the_window_and_keeps_retrying():
    """A render can be unfindable for a while -- a black loading screen, another window overlapping --
    and giving up would strand the bot on a stale region for the rest of the session."""
    cap = _capture(_FakeBackend(Region(0, 0, 400, 800), None))
    assert cap.region == Region(0, 0, 400, 800)
    assert cap._render_locked is False, "an unlocked render must keep re-scanning on each grab"


def test_an_explicit_region_overrides_the_backend():
    """The calibration escape hatch: a setup the content scan cannot handle must still be usable."""
    from clashrl.capture import WindowCapture
    be = _FakeBackend(Region(0, 0, 400, 800), Region(10, 20, 200, 360))
    cap = WindowCapture(None, [5, 6, 100, 200], cfg=_Cfg({}), backend=be)
    assert cap.region == Region(5, 6, 100, 200)
    cap.refresh_region()
    assert cap.region == Region(5, 6, 100, 200), "an explicit region must never be re-detected away"


def test_a_malformed_region_is_ignored_rather_than_crashing(capsys):
    from clashrl.capture import WindowCapture
    cap = WindowCapture(None, [None], cfg=_Cfg({}), backend=_FakeBackend(Region(0, 0, 400, 800)))
    assert "must be 4 numbers" in capsys.readouterr().out
    assert cap.region == Region(0, 0, 400, 800)


def test_normalized_coordinates_round_trip_through_the_region():
    """Every calibrated constant in the config is normalized, so this mapping is what makes one
    calibration meaningful at any window size."""
    cap = _capture(_FakeBackend(Region(100, 50, 400, 800), Region(100, 50, 400, 800)))
    assert cap.to_screen(0.5, 0.25) == (300, 250)
    nx, ny = cap.to_norm(300, 250)
    assert (round(nx, 6), round(ny, 6)) == (0.5, 0.25)


def test_controller_taps_through_the_backend():
    """Tap delivery is the platform-specific half: GPG takes a plain click, the mirroring link needs
    a held one into a raised window."""
    from clashrl.controller import Controller
    be = _FakeBackend(Region(0, 0, 400, 800), Region(0, 0, 400, 800))
    ctl = Controller(_capture(be), _Cfg({}))
    ctl.tap(0.5, 0.5)
    assert be.taps == [(200, 400)]


def test_controller_plays_a_card_as_select_then_place():
    """The SEQUENCE is the game's, not the host's -- it must be identical on every platform."""
    from clashrl.controller import Controller
    be = _FakeBackend(Region(0, 0, 400, 800), Region(0, 0, 400, 800))
    ctl = Controller(_capture(be), _Cfg({"play": {"select_delay": 0.0}}))
    ctl.play_card(0.25, 0.9, 0.5, 0.6)
    assert be.taps == [(100, 720), (200, 480)], "select the slot, then place"


def test_controller_shares_the_captures_backend_by_default():
    """Tapping and capturing must agree about WHICH window they are talking to; two independently
    built backends could disagree after a re-detect."""
    from clashrl.controller import Controller
    be = _FakeBackend(Region(0, 0, 400, 800), Region(0, 0, 400, 800))
    cap = _capture(be)
    assert Controller(cap, _Cfg({})).backend is be


def test_the_retina_scale_factor_is_measured_not_assumed(raw_cfg, capsys):
    """AppleScript reports POINTS, mss grabs PIXELS, and on a Retina Mac those differ by 2x. Getting
    it wrong does not fail loudly -- it captures a quarter of the window and lands every tap at half
    the intended coordinate.

    REGRESSION GUARD: the first version asked Finder for the desktop bounds over AppleScript. On a real
    Retina Mac that call HUNG (it needs Automation permission for Finder) and the except-branch
    silently returned 1.0 -- a wrong answer that looks like a right one. An unmeasurable scale must now
    WARN rather than quietly assume.
    """
    cfg = _Cfg(resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"}))
    be = MirrorBackend(cfg)
    scale = be._backing_scale()
    assert 0.5 <= scale <= 4.0
    if scale == 1.0 and "could not measure" in capsys.readouterr().out:
        pytest.skip("Quartz unavailable here; the warning path fired, which is the point")
    assert be._backing_scale() == scale, "the measurement must be cached, not re-derived per tap"


def test_an_unmeasurable_scale_warns_instead_of_silently_assuming(raw_cfg, monkeypatch, capsys):
    """Silence here is the dangerous outcome: 1.0 on a Retina Mac breaks capture AND input while
    looking like a normal startup."""
    import sys
    monkeypatch.setitem(sys.modules, "Quartz", None)     # force the import to fail
    cfg = _Cfg(resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"}))
    be = MirrorBackend(cfg)
    assert be._backing_scale() == 1.0
    out = capsys.readouterr().out
    assert "backing scale" in out and "Retina" in out


def test_the_window_lookup_degrades_when_the_app_is_not_running(raw_cfg):
    """iPhone Mirroring not being open is the normal case most of the time; it must return None so the
    capture keeps retrying, not raise into the middle of a match loop."""
    cfg = _Cfg(resolve_platform({**copy.deepcopy(raw_cfg), "platform": "macos_mirror"}))
    be = MirrorBackend(cfg)
    be.app_name = "NoSuchApplicationHopefully"
    assert be.find_window() is None


# --- what is NOT covered here ---------------------------------------------------------------------
@pytest.mark.skip(reason="needs a real iPhone Mirroring window on a real display: the osascript "
                         "window lookup, the measured Retina scale factor and the render trim cannot "
                         "be exercised without the app running and Accessibility granted")
def test_mirror_backend_against_a_live_window():
    """Placeholder that records the gap rather than pretending mocks covered it.

    Run `run.py diag` on a Mac with iPhone Mirroring open to check these by eye: the reported region
    should hug the phone screen, and the tap markers should land where you expect.
    """
