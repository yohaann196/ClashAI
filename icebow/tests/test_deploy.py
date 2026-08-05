"""DEPLOYMENT: the ~1s spawn delay, spells landing on their own timer, and multi-unit placement."""
from __future__ import annotations

import math

import pytest

from conftest import place, place_one, run, silence_towers


def test_troops_take_about_a_second_to_appear(spec):
    """CR mechanic: DEPLOY TIME. A troop cannot act for ~1s after being placed -- the reason you
    cannot instantly block a push that is already at your tower."""
    for card in ("knight", "musketeer", "miner"):
        assert spec(card).deploy_time == pytest.approx(1.0)


def test_spells_have_no_deploy_delay(spec):
    """CR mechanic: SPELLS ARE NOT TROOPS -- they have no spawn animation; they use a travel/cast
    delay instead (see test_spells.py)."""
    for card in ("rocket", "the_log", "tornado"):
        assert spec(card).deploy_time == pytest.approx(0.0)


def test_a_deploying_troop_cannot_move(engine, spec):
    """CR mechanic: DEPLOY DELAY freezes the troop in place -- it does not begin walking until it has
    finished spawning."""
    silence_towers(engine)
    knight = spec("knight")
    u = place_one(engine, 0, knight, 0.30, 0.60, ready=False)
    start = (u.x, u.y)

    run(engine, 0.5)                                   # still spawning
    assert (u.x, u.y) == pytest.approx(start), "must not move during the deploy delay"
    assert u.deploy_left > 0

    run(engine, 2.0)                                   # well past the delay
    assert math.dist((u.x, u.y), start) > 0.0, "must start advancing once deployed"


def test_a_deploying_troop_cannot_attack(engine, spec):
    """CR mechanic: a troop mid-spawn deals NO damage, even standing on top of its target."""
    silence_towers(engine)
    tower = engine.towers[1][0]
    knight = spec("knight")
    place_one(engine, 0, knight, tower.x, tower.y + knight.reach, ready=False)

    run(engine, 0.5)
    assert tower.hp == pytest.approx(tower.max_hp), "no damage during the deploy delay"

    run(engine, 2.0)
    assert tower.hp < tower.max_hp, "it must attack once deployed"


def test_deploy_delay_expires_after_its_full_duration(engine, spec):
    """CR mechanic: the delay is a FIXED ~1s, not a per-tick approximation."""
    knight = spec("knight")
    u = place_one(engine, 0, knight, 0.30, 0.60, ready=False)
    assert u.deploy_left == pytest.approx(knight.deploy_time)

    run(engine, 0.9)
    assert u.deploy_left > 0.0, "still spawning just before 1s"

    run(engine, 0.3)
    assert u.deploy_left <= 0.0, "spawned by ~1.1s"


def test_multi_unit_card_spawns_a_cluster_at_the_placement_point(engine, spec):
    """CR mechanic: SWARM PLACEMENT. A multi-unit card (Skeletons, Minions, Royal Recruits) drops all
    of its units as a tight cluster CENTRED ON WHERE YOU PLACED IT -- not at some other spot.

    REGRESSION GUARD: the spawn offsets used to re-include the placement coordinate (`ox = x + ...`)
    and were then added to `x` again, resolving every unit to `2x + offset`. For any normal placement
    that clamped into the arena corner, so every swarm card in the sim -- Skeletons in this very deck --
    spawned at (0.97, 0.97) stacked on each other instead of where they were played.
    """
    skeletons = spec("skeletons")
    assert skeletons.count >= 3, "this test needs a genuine multi-unit card"

    px, py = 0.50, 0.60
    units = place(engine, 0, skeletons, px, py)
    assert len(units) == skeletons.count

    for u in units:
        assert math.dist((u.x, u.y), (px, py)) < 0.05, (
            f"unit spawned at ({u.x:.3f}, {u.y:.3f}), far from the placement point ({px}, {py})")

    xs = {round(u.x, 4) for u in units}
    assert len(xs) > 1, "the cluster must be spread out, not stacked on one point"


def test_single_unit_card_spawns_exactly_where_placed(engine, spec):
    """CR mechanic: a one-unit card lands EXACTLY on the placement point."""
    px, py = 0.42, 0.63
    u = place_one(engine, 0, spec("knight"), px, py)
    assert (u.x, u.y) == pytest.approx((px, py))


def test_deployment_is_clamped_inside_the_arena(engine, spec):
    """CR mechanic: units exist ON the board -- a placement at the very edge is pulled inside it."""
    u = place_one(engine, 0, spec("knight"), 0.0, 1.0)
    assert 0.03 <= u.x <= 0.97
    assert 0.03 <= u.y <= 0.97


def test_buildings_never_move(engine, spec):
    """CR mechanic: BUILDINGS ARE STATIONARY. An X-Bow shoots from where it was placed and never
    advances, which is what makes siege placement decisive."""
    silence_towers(engine)
    xbow = spec("x_bow")
    assert xbow.kind == "building"
    u = place_one(engine, 0, xbow, 0.48, 0.55)
    start = (u.x, u.y)

    run(engine, 5.0)
    assert (u.x, u.y) == pytest.approx(start)


def test_buildings_expire_after_their_lifetime(engine, spec):
    """CR mechanic: BUILDING LIFETIME -- a building decays and disappears on its own timer even if it
    is never attacked."""
    silence_towers(engine)
    xbow = spec("x_bow")
    assert xbow.lifetime is not None
    place_one(engine, 0, xbow, 0.48, 0.55)
    assert any(u.spec.base == "x_bow" for u in engine.units)

    run(engine, xbow.lifetime + 1.0)
    assert not any(u.spec.base == "x_bow" for u in engine.units), "the building must expire"


def test_deploy_is_refused_once_the_match_is_over(engine, spec):
    """CR mechanic: no plays after the final whistle."""
    engine.done = True
    engine.elixir[0] = 10.0
    assert engine.deploy(0, spec("knight"), 0.5, 0.6) is False
