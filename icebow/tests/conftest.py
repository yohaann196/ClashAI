"""Shared fixtures for the sim-engine suite.

Two rules the whole suite follows:

GOLDEN VALUES COME FROM `config/cards_stats.json`. That file is the imported Fandom-wiki stat dump
(level-11 vardefines) and is the closest thing this project has to ground truth for Clash Royale
numbers. Tests assert against it in two complementary ways -- recomputing the engine's contract from
the published fields (catches drift in `build_spec`) AND pinning a handful of literal constants
(catches silent drift in the stat file itself). One without the other would let a regression through.

THE ENGINE IS MADE DETERMINISTIC. `SimEngine.reset()` rolls the opponent's tower troop and level from
weighted distributions, so an unconstrained engine has different enemy tower HP every run. The `cfg`
fixture pins that roll to princess/L15 -- identical to the player's side -- so tower arithmetic is
symmetric and reproducible. Tests that care about the roll itself should build their own config.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ICEBOW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ICEBOW / "src"))

from clashrl.cards import CardDB              # noqa: E402
from clashrl.config import Config             # noqa: E402
from clashrl.sim.engine import SimEngine, build_spec  # noqa: E402

# Physics sub-tick the env actually drives the engine with (`sim.sub_dt`). Tests step at this
# granularity so timing behaviour matches training rather than some idealised dt.
DT = 0.1


@pytest.fixture(scope="session")
def stats() -> dict:
    """The GOLDEN card stats: `config/cards_stats.json` -> {card: {hitpoints, damage, hit_speed, dps...}}."""
    raw = json.loads((ICEBOW / "config" / "cards_stats.json").read_text(encoding="utf-8"))
    return raw["cards"]


@pytest.fixture(scope="session")
def stats_meta() -> dict:
    raw = json.loads((ICEBOW / "config" / "cards_stats.json").read_text(encoding="utf-8"))
    return raw["meta"]


@pytest.fixture(scope="session")
def db() -> CardDB:
    return CardDB(path=ICEBOW / "config" / "cards.yaml")


@pytest.fixture()
def cfg() -> Config:
    """Project config with the opponent's tower roll PINNED so both sides run princess @ L15.

    Without this the enemy tower troop (princess / dagger duchess / cannoneer / royal chef) and its
    level are sampled per match, so max HP, hit damage and cadence differ run to run.
    """
    c = Config.load(str(ICEBOW / "config" / "config.yaml"))
    c.data.setdefault("sim", {})
    c.data["sim"]["opponent_tower_weights"] = {"princess": 1}
    c.data["sim"]["enemy_levels"] = [15]
    c.data["sim"]["enemy_level_weights"] = [1]
    return c


@pytest.fixture()
def engine(cfg, db) -> SimEngine:
    return SimEngine(cfg, db, random.Random(0))


@pytest.fixture()
def spec(db):
    """`spec("knight", level=11)` -> CardSpec, the engine's stat resolution for one card."""
    def _spec(key: str, level: int = 11):
        return build_spec(db, key, level)
    return _spec


# --- helpers ---------------------------------------------------------------------------------
def place(engine, team: int, spec, x: float, y: float, ready: bool = True) -> list:
    """Deploy `spec` for `team` at (x, y), bypassing the elixir check, and return the new units.

    `ready=True` clears the ~1s deploy delay so a test can exercise combat/movement without first
    burning a second of simulation. Tests that are ABOUT the deploy delay pass ready=False.
    """
    engine.elixir[team] = 10.0
    n0 = len(engine.units)
    assert engine.deploy(team, spec, x, y), "deploy rejected (affordability/`done` guard)"
    fresh = engine.units[n0:]
    if ready:
        for u in fresh:
            u.deploy_left = 0.0
    return fresh


def place_one(engine, team: int, spec, x: float, y: float, ready: bool = True):
    units = place(engine, team, spec, x, y, ready)
    assert len(units) == 1, f"{spec.key} spawns {len(units)} units; use place() instead"
    return units[0]


def silence_towers(engine) -> None:
    """Zero every tower's damage.

    Combat tests need the unit under test to survive long enough to measure a cadence, and crown
    towers out-damage most single troops. Silencing them isolates the mechanic being pinned instead
    of racing tower DPS. Range/targeting/wake behaviour is untouched, so tower tests still work.
    """
    for team in (0, 1):
        for tw in engine.towers[team]:
            tw.hit_dmg = 0.0


def freeze(unit) -> None:
    """Immobilise a unit via a huge DEPLOY delay -- it cannot act, is not separated by collision, and
    is NOT TARGETABLE BY TOWERS (deploy immunity, pinned in test_towers.py).

    Use for spell/vortex geometry, where the measurement must not be polluted by the unit walking,
    fighting back, or being shoved by collision. Do NOT use when a tower needs to shoot it -- use
    `park()` instead.
    """
    unit.deploy_left = 1e9


def park(unit) -> None:
    """Immobilise a unit via a huge STUN -- it cannot act, but it remains a legal TOWER TARGET.

    The counterpart to `freeze()`: use whenever a test needs a stationary victim that towers will
    actually fire at.
    """
    unit.stun_left = 1e9


def run(engine, seconds: float, dt: float = DT) -> None:
    """Advance the engine `seconds` in `dt` sub-ticks (the cadence training uses)."""
    for _ in range(int(round(seconds / dt))):
        engine.advance(dt)


def total_damage_to(unit, spec) -> float:
    """Damage a unit has taken, counting the shield pool it started with."""
    return (spec.hp + spec.shield_hp) - (unit.hp + unit.shield_left)
