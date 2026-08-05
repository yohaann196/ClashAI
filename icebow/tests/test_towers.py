"""CROWN TOWERS: destruction, crown accounting, and the KING ACTIVATION rules."""
from __future__ import annotations

import pytest

from conftest import park, place_one, run, silence_towers

LEFT, RIGHT, KING = 0, 1, 2


def test_king_tower_starts_asleep_and_princesses_start_awake(engine):
    """CR mechanic: the KING TOWER STARTS DORMANT. It does not fire until it is activated, while both
    princess towers defend from the opening second."""
    for team in (0, 1):
        assert engine.towers[team][KING].king is True
        assert engine.towers[team][KING].active is False, "the king must start asleep"
        assert engine.towers[team][LEFT].active is True
        assert engine.towers[team][RIGHT].active is True


def test_a_sleeping_king_does_not_shoot(engine, spec):
    """CR mechanic: a DORMANT KING is inert -- troops can stand in its range untouched, which is what
    makes an un-activated king safe to walk past."""
    king = engine.towers[1][KING]
    for tw in engine.towers[1][:2]:
        tw.alive = False                            # remove princess interference
    victim_spec = spec("musketeer")
    u = place_one(engine, 0, victim_spec, king.x, king.y + 0.05)   # well inside king_range
    park(u)                                         # stationary, but still a legal tower target

    run(engine, 5.0)
    assert u.hp == pytest.approx(victim_spec.hp), "an asleep king must deal no damage"

    king.active = True
    run(engine, 5.0)
    assert u.hp < victim_spec.hp, "once activated the king must fire"


def test_any_damage_to_the_king_wakes_it(engine, spec):
    """CR mechanic: KING ACTIVATION BY DAMAGE. Any hit on the king tower -- a spell, a Miner, chip from
    anything -- wakes it for the rest of the match. This is why chipping the king is a real cost."""
    king = engine.towers[1][KING]
    assert king.active is False

    engine._damage_tower(king, 1.0, by_team=0)
    assert king.active is True, "one point of damage must activate the king"
    assert king.hp == pytest.approx(king.max_hp - 1.0)


def test_losing_a_princess_activates_that_side_s_king(engine):
    """CR mechanic: KING ACTIVATION BY TOWER LOSS. When a princess tower falls, the DEFENDER's king
    wakes up -- the attacker's king is unaffected."""
    defender, attacker = 1, 0
    princess = engine.towers[defender][LEFT]
    assert engine.towers[defender][KING].active is False
    assert engine.towers[attacker][KING].active is False

    engine._damage_tower(princess, princess.hp, by_team=attacker)

    assert princess.alive is False
    assert engine.towers[defender][KING].active is True, "the DEFENDER's king wakes"
    assert engine.towers[attacker][KING].active is False, "the attacker's king must NOT wake"


def test_tower_destruction_clamps_hp_and_flips_alive(engine):
    """CR mechanic: a tower is DESTROYED at 0 HP -- it cannot go negative and stops existing as a
    target. (Until it hits 0 it works at full strength, which is why partial chip is worth so little.)"""
    tower = engine.towers[1][LEFT]
    engine._damage_tower(tower, tower.max_hp * 10, by_team=0)

    assert tower.hp == 0.0, "HP must clamp at zero, not go negative"
    assert tower.alive is False


def test_overkill_damage_is_not_credited_as_chip(engine):
    """CR mechanic: you cannot deal MORE damage than a tower has left. Overkill is wasted, so the chip
    accounting the reward reads must record only the HP actually removed."""
    tower = engine.towers[1][LEFT]
    tower.hp = 100.0
    engine.chip = {0: 0.0, 1: 0.0}

    engine._damage_tower(tower, 5000.0, by_team=0)
    assert engine.chip[0] == pytest.approx(100.0), "only the remaining 100 HP counts"


def test_a_destroyed_tower_takes_no_further_damage(engine):
    """CR mechanic: a felled tower is GONE -- later splash or spells cannot 'hit' it again."""
    tower = engine.towers[1][LEFT]
    engine._damage_tower(tower, tower.max_hp, by_team=0)
    engine.chip = {0: 0.0, 1: 0.0}

    engine._damage_tower(tower, 500.0, by_team=0)
    assert engine.chip[0] == pytest.approx(0.0)
    assert tower.hp == 0.0


def test_crowns_count_destroyed_enemy_towers(engine):
    """CR mechanic: a CROWN is scored per enemy tower destroyed -- the match score."""
    assert engine.crowns(0) == 0
    engine._damage_tower(engine.towers[1][LEFT], 1e9, by_team=0)
    assert engine.crowns(0) == 1
    engine._damage_tower(engine.towers[1][RIGHT], 1e9, by_team=0)
    assert engine.crowns(0) == 2
    assert engine.crowns(1) == 0, "the opponent has taken nothing"


def test_felling_the_king_ends_the_match_immediately(engine):
    """CR mechanic: a KING TOWER KILL is an instant three-crown victory -- the match stops there."""
    engine._damage_tower(engine.towers[1][KING], 1e9, by_team=0)
    engine.advance(0.1)

    assert engine.done is True
    assert engine.outcome == "win"


def test_losing_your_own_king_is_a_loss(engine):
    """CR mechanic: the same rule from the other side -- your king falling ends the match as a loss."""
    engine._damage_tower(engine.towers[0][KING], 1e9, by_team=1)
    engine.advance(0.1)

    assert engine.done is True
    assert engine.outcome == "loss"


def test_king_tower_has_more_hitpoints_than_a_princess(engine):
    """CR mechanic: the KING TOWER IS THE TOUGHEST -- more HP than either princess tower."""
    king = engine.towers[0][KING]
    princess = engine.towers[0][LEFT]
    assert king.max_hp > princess.max_hp


def test_king_tower_range_is_shorter_than_a_princess_tower(engine):
    """CR mechanic: the KING covers LESS ground than a princess tower (7 tiles vs 7.5), which is why
    troops can siege from just outside its reach."""
    assert engine.king_range < engine.tower_range


def test_tied_crowns_are_broken_by_the_least_healthy_tower(engine):
    """CR mechanic: the OVERTIME TIEBREAK. With crowns level, the side whose least-healthy crown tower
    is worse off loses. Compared as HP FRACTIONS so differing tower levels stay fair."""
    engine.towers[0][LEFT].hp = engine.towers[0][LEFT].max_hp * 0.90
    engine.towers[1][LEFT].hp = engine.towers[1][LEFT].max_hp * 0.40
    assert engine._score_outcome() == "win", "the enemy's weakest tower is worse -> we win"

    engine.towers[0][LEFT].hp = engine.towers[0][LEFT].max_hp * 0.20
    assert engine._score_outcome() == "loss"


def test_equal_damage_is_a_draw(engine):
    """CR mechanic: a genuine DRAW -- crowns level and both sides equally damaged."""
    assert engine._score_outcome() == "draw"


def test_tower_hit_damage_scales_with_tower_level(cfg, db):
    """CR mechanic: TOWER LEVEL SCALING -- crown towers gain HP and damage at CR's 1.1x per level, the
    same curve cards follow, so a higher-level opponent's towers hit harder and last longer."""
    import random

    from clashrl.sim.engine import SimEngine

    cfg.data["sim"]["enemy_levels"] = [16]           # one level above our reference 15
    cfg.data["sim"]["enemy_level_weights"] = [1]
    e = SimEngine(cfg, db, random.Random(0))

    mine, theirs = e.towers[0][LEFT], e.towers[1][LEFT]
    assert theirs.max_hp == pytest.approx(mine.max_hp * 1.1)
    assert theirs.hit_dmg == pytest.approx(mine.hit_dmg * 1.1)
    assert theirs.hit_speed == pytest.approx(mine.hit_speed), "level never changes hit SPEED"


def test_towers_ignore_units_that_are_still_deploying(engine, spec):
    """CR mechanic: DEPLOY IMMUNITY -- a troop still materialising is not yet a target, so towers do
    not begin firing during the spawn animation."""
    silence_towers(engine)                            # (we assert on targeting, not on damage)
    tower = engine.towers[1][0]
    tower.hit_dmg = 500.0                             # re-arm just this tower
    victim_spec = spec("musketeer")
    u = place_one(engine, 0, victim_spec, tower.x, tower.y + 0.05, ready=False)
    assert u.deploy_left > 0

    run(engine, 0.5)                                  # still inside the deploy window
    assert u.hp == pytest.approx(victim_spec.hp), "a spawning troop must not be shot"
