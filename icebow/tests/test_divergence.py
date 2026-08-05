"""SIM-VS-REALITY DIVERGENCE HARNESS (clashrl/divergence.py).

The harness cannot be tested against a real recording here -- that needs a session and a trained
detector. But its two load-bearing claims can be tested without either, by using the ENGINE ITSELF as
the "reality" being matched:

  1. SELF-CONSISTENCY -- replaying a script against a trajectory the same engine produced from the
     same script must score ~0. A harness that reports drift where there is none cannot be trusted to
     report drift where there is.
  2. INJECTED-DRIFT ATTRIBUTION -- generate "reality" with ONE mechanic deliberately broken, then ask
     the harness which mechanic explains the drift. It must name the one that was broken. This is the
     end-to-end test of the ranking: a known cause goes in, the right answer must come out.

That leaves exactly one thing untested here: the OBSERVATION half (video -> trajectory), which needs a
real recording. Stated plainly rather than papered over.
"""
from __future__ import annotations

import pytest

from clashrl.divergence import (MECHANICS, Play, Sample, Trajectory, _charge_gap, replay_engine,
                                score)


def _script(n=8):
    """A deterministic two-sided script that EXERCISES EVERY RANKED MECHANIC.

    Built by role rather than by cycling a card list, because an ablation only produces a signal if
    the match actually used the mechanic:
      * ground troops spawned MID-ARENA (x=0.42/0.58, off the bridges at 0.25/0.75) so bridge routing
        really engages -> `pathing`, and so they collide on the way -> `body_block`
      * a building-targeting troop (Hog Rider) -> `building_pull`
      * ranged troops that must acquire targets -> `aggro`
      * a siege building (X-Bow) -> `siege_sight`
      * a spell aimed at an ENEMY TOWER, not at my own half where it would hit nothing -> `spell_delay`
    """
    plays = []
    for i in range(n):
        t = 4.0 + i * 7.0
        lane = 0.42 if i % 2 == 0 else 0.58
        mine = ["knight", "skeletons", "ice_wizard", "x_bow"][i % 4]
        y = 0.58 if mine == "x_bow" else 0.62
        plays.append(Play(t=t, team=0, card=mine, x=lane, y=y))
        plays.append(Play(t=t + 2.0, team=1,
                          card=["hog_rider", "musketeer", "knight", "archers"][i % 4],
                          x=lane, y=0.38))
        if i % 3 == 2:                                  # rocket the enemy princess tower
            plays.append(Play(t=t + 4.0, team=0, card="rocket", x=0.25, y=0.205))
    return sorted(plays, key=lambda p: p.t)


@pytest.fixture()
def script():
    return _script()


# --- 1. self-consistency ------------------------------------------------------------------------
def test_replaying_the_engine_against_itself_scores_zero_divergence(cfg, db, script):
    """A harness that reports drift where there is none cannot be trusted where there is."""
    truth = replay_engine(cfg, db, script, duration=120.0)
    again = replay_engine(cfg, db, script, duration=120.0)
    sc = score(truth, again, cfg)

    assert sc["tower_hp"] == pytest.approx(0.0, abs=1e-9)
    assert sc["units"] == pytest.approx(0.0, abs=1e-9)
    assert sc["first_tower"] == pytest.approx(0.0, abs=1e-9)
    assert sc["outcome"] == pytest.approx(0.0)
    assert sc["divergence"] == pytest.approx(0.0, abs=1e-9)


def test_the_replay_is_deterministic(cfg, db, script):
    """Attribution compares replays against each other, so any nondeterminism would show up as
    phantom mechanic sensitivity."""
    a = replay_engine(cfg, db, script, duration=90.0)
    b = replay_engine(cfg, db, script, duration=90.0)
    assert [s.my_hp for s in a.samples] == [s.my_hp for s in b.samples]
    assert [s.enemy_units for s in a.samples] == [s.enemy_units for s in b.samples]
    assert a.outcome == b.outcome and a.first_tower_t == b.first_tower_t


def test_the_script_is_actually_executed(cfg, db, script):
    """If the engine silently refused the scripted plays the harness would be measuring an empty
    match. Deploys are forced past the engine's own elixir accounting for exactly this reason."""
    traj = replay_engine(cfg, db, script, duration=120.0)
    assert any(s.my_units and s.my_units > 0 for s in traj.samples), "no friendly units ever existed"
    assert any(s.enemy_units and s.enemy_units > 0 for s in traj.samples), "no enemy units ever existed"
    assert any(s.enemy_hp[0] < 1.0 or s.enemy_hp[1] < 1.0 for s in traj.samples), \
        "nothing ever damaged a tower -- the script did nothing"


# --- 2. injected-drift attribution --------------------------------------------------------------
@pytest.mark.parametrize("broken", ["body_block", "aggro", "building_pull", "pathing"])
def test_injected_drift_is_attributed_to_the_mechanic_that_caused_it(cfg, db, script, broken):
    """THE END-TO-END TEST OF THE RANKING. Build "reality" with ONE mechanic ablated, then score the
    normal engine against it and rank every mechanic by how much ablating it moves the divergence.

    The mechanic that was actually broken must come out on top: ablating it in the replay too makes
    the replay match "reality" again, so its delta is the most negative.
    """
    spec = MECHANICS[broken]
    truth = replay_engine(cfg, db, script, duration=150.0,
                          spec_patch=spec.get("spec"), engine_patch=spec.get("engine"))

    base = score(truth, replay_engine(cfg, db, script, duration=150.0), cfg)
    if base["divergence"] == pytest.approx(0.0, abs=1e-9):
        pytest.skip(f"ablating {broken} changed nothing in this scripted match -- nothing to attribute")

    deltas = {}
    for name, m in MECHANICS.items():
        sim = replay_engine(cfg, db, script, duration=150.0,
                            spec_patch=m.get("spec"), engine_patch=m.get("engine"))
        deltas[name] = score(truth, sim, cfg)["divergence"] - base["divergence"]

    best = min(deltas, key=lambda k: deltas[k])
    assert deltas[broken] < 0, (
        f"ablating the broken mechanic ({broken}) should REDUCE divergence toward reality; "
        f"got {deltas[broken]:+.4f}. deltas={ {k: round(v, 4) for k, v in deltas.items()} }")
    assert best == broken, (
        f"attribution named {best!r} but {broken!r} was the injected cause. "
        f"deltas={ {k: round(v, 4) for k, v in deltas.items()} }")


def test_ablating_the_broken_mechanic_recovers_a_near_perfect_match(cfg, db, script):
    """The strong form: reproducing reality's ablation in the replay should drive divergence back to
    ~0, since the two engines are then identical."""
    spec = MECHANICS["body_block"]
    truth = replay_engine(cfg, db, script, duration=150.0, engine_patch=spec["engine"])
    matched = replay_engine(cfg, db, script, duration=150.0, engine_patch=spec["engine"])
    assert score(truth, matched, cfg)["divergence"] == pytest.approx(0.0, abs=1e-9)


def test_every_mechanic_ablation_actually_changes_the_simulation(cfg, db, script):
    """A mechanic whose ablation changes nothing is either not wired up or not exercised by the
    script -- either way its ranking row would be meaningless, so surface it here.

    Checked at PHYSICS resolution (0.1s), not the 1s reporting stride: a 0.4s spell-delay change falls
    entirely between 1s samples, so at the coarse stride it would look 'unexercised' when it is simply
    sub-stride. The harness makes the same distinction for its [NOT EXERCISED] tag.
    """
    base = replay_engine(cfg, db, script, duration=150.0, stride_s=0.1)
    inert = []
    for name, m in MECHANICS.items():
        sim = replay_engine(cfg, db, script, duration=150.0, stride_s=0.1,
                            spec_patch=m.get("spec"), engine_patch=m.get("engine"))
        if score(base, sim, cfg)["divergence"] == pytest.approx(0.0, abs=1e-9):
            inert.append(name)
    assert not inert, f"these ablations had no effect on the scripted match: {inert}"


def test_sub_stride_drift_is_invisible_to_the_score_but_not_to_the_exercised_check(cfg, db, script):
    """Pins the resolution caveat that cost a debugging round: a spell landing 0.4s earlier changes
    nothing the 1s sampling grid can see, so the SCORE is legitimately 0 -- but the mechanic was very
    much used, which is why the 'exercised' check runs at physics resolution instead."""
    m = MECHANICS["spell_delay"]
    coarse_a = replay_engine(cfg, db, script, duration=150.0, stride_s=1.0)
    coarse_b = replay_engine(cfg, db, script, duration=150.0, stride_s=1.0, spec_patch=m["spec"])
    assert score(coarse_a, coarse_b, cfg)["divergence"] == pytest.approx(0.0, abs=1e-9)

    fine_a = replay_engine(cfg, db, script, duration=150.0, stride_s=0.1)
    fine_b = replay_engine(cfg, db, script, duration=150.0, stride_s=0.1, spec_patch=m["spec"])
    assert score(fine_a, fine_b, cfg)["divergence"] > 0.0


# --- 3. the score itself ------------------------------------------------------------------------
def _traj(hp_mine, hp_theirs, units=(0, 0), outcome="draw", first=None, dur=10.0):
    return Trajectory(
        samples=[Sample(t=float(i), my_hp=list(hp_mine), enemy_hp=list(hp_theirs),
                        my_units=units[0], enemy_units=units[1]) for i in range(int(dur))],
        outcome=outcome, first_tower_t=first, duration=dur)


def test_tower_hp_component_is_mean_absolute_fraction_error():
    """Tower HP is the backbone signal: it integrates almost every mechanic (who reached what, when)."""
    a = _traj([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    b = _traj([0.8, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert score(a, b)["tower_hp"] == pytest.approx(0.2 / 6)


def test_outcome_component_is_binary():
    """Coarse on purpose -- it is the one thing that actually matters about a match."""
    a = _traj([1.0] * 3, [1.0] * 3, outcome="win")
    assert score(a, _traj([1.0] * 3, [1.0] * 3, outcome="win"))["outcome"] == 0.0
    assert score(a, _traj([1.0] * 3, [1.0] * 3, outcome="loss"))["outcome"] == 1.0


def test_first_tower_component_is_normalized_by_match_length():
    """10s of error in a 3-minute match is not the same as 10s in a 20-second one."""
    a = _traj([1.0] * 3, [1.0] * 3, first=20.0, dur=100.0)
    b = _traj([1.0] * 3, [1.0] * 3, first=30.0, dur=100.0)
    assert score(a, b)["first_tower"] == pytest.approx(0.1)


def test_one_side_never_losing_a_tower_is_maximum_first_tower_divergence():
    """'A tower fell' vs 'no tower ever fell' is a total disagreement, not a small time error."""
    a = _traj([1.0] * 3, [1.0] * 3, first=20.0, dur=100.0)
    b = _traj([1.0] * 3, [1.0] * 3, first=None, dur=100.0)
    assert score(a, b)["first_tower"] == pytest.approx(1.0)


def test_unavailable_components_are_skipped_and_the_composite_renormalized():
    """Without a detector there are no unit counts. The composite must renormalize over the metrics
    that WERE measured rather than silently scoring the missing one as a perfect match."""
    a = _traj([1.0] * 3, [1.0] * 3)
    b = _traj([1.0] * 3, [1.0] * 3)
    for s in list(a.samples) + list(b.samples):
        s.my_units = s.enemy_units = None
    sc = score(a, b)
    assert sc["units"] != sc["units"], "units should be NaN when unavailable"
    assert sc["_weight_covered"] < 1.0
    assert sc["divergence"] == pytest.approx(0.0)


def test_a_totally_wrong_simulation_scores_near_one():
    """The score has to have a usable top end, or 'bad' and 'catastrophic' look the same."""
    real = _traj([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], units=(8, 8), outcome="win", first=10.0, dur=60.0)
    sim = _traj([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], units=(0, 0), outcome="loss", first=None, dur=60.0)
    assert score(real, sim)["divergence"] > 0.8


# --- 4. the unmodelled-charge gap ---------------------------------------------------------------
def test_charge_cards_in_the_script_are_reported_as_an_unmodelled_gap(db):
    """`charge` cannot be ablated because the engine does not model it -- sim/engine.py says so in its
    own docstring. It has to be reported as a structural gap, not silently missing from the ranking."""
    played = [Play(t=1.0, team=1, card="prince", x=0.5, y=0.4),
              Play(t=2.0, team=0, card="knight", x=0.5, y=0.6)]
    gap = _charge_gap(played, db)
    assert "prince" in gap
    assert "knight" not in gap, "Knight has no charge mechanic"


def test_charge_is_not_offered_as_an_ablatable_mechanic():
    """Offering a `charge` ablation would imply the engine models it. It does not."""
    assert "charge" not in MECHANICS


def test_a_match_with_no_charge_cards_reports_an_empty_gap(db):
    played = [Play(t=1.0, team=0, card="knight", x=0.5, y=0.6),
              Play(t=2.0, team=0, card="skeletons", x=0.5, y=0.6)]
    assert _charge_gap(played, db) == []
