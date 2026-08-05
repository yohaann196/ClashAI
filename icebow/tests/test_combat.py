"""COMBAT MATH: discrete hits at a fixed cadence, DPS<->hit-speed consistency, and the CROWN-TOWER
DAMAGE REDUCTION that makes `tower_hit_dmg` diverge from `hit_dmg`."""
from __future__ import annotations

import pytest

from conftest import DT, freeze, park, place_one, run, silence_towers

# Cards whose published stats every derived number below is checked against.
TROOPS_WITH_ATTACKS = ["knight", "musketeer", "miner", "royal_recruits", "archers", "valkyrie"]


@pytest.mark.parametrize("card", TROOPS_WITH_ATTACKS)
def test_hit_damage_is_dps_times_hit_speed(spec, stats, card):
    """CR mechanic: DPS IS DERIVED, NOT PRIMARY -- a troop lands one discrete hit every `hit_speed`
    seconds, and its damage-per-second is that hit divided by the interval. The engine stores the
    per-hit figure, so `hit_dmg == dps * hit_speed` must hold against the published stats."""
    published = stats[card]
    s = spec(card)
    assert s.hit_speed == pytest.approx(published["hit_speed"])
    assert s.hit_dmg == pytest.approx(published["dps"] * published["hit_speed"])


@pytest.mark.parametrize("card", TROOPS_WITH_ATTACKS)
def test_hitpoints_match_the_published_level_11_stats(spec, stats, card):
    """CR mechanic: a card's HITPOINTS at its reference level (the stat dump is level-11)."""
    assert spec(card, 11).hp == pytest.approx(stats[card]["hitpoints"])


def test_knight_golden_stats(spec, stats):
    """CR mechanic: KNIGHT'S STAT LINE. His level-11 numbers, pinned as literals -- this also catches
    silent drift in cards_stats.json itself, which the recompute-from-published tests cannot see."""
    assert stats["knight"]["hitpoints"] == 1766
    assert stats["knight"]["damage"] == 202
    assert stats["knight"]["hit_speed"] == 1.2
    assert stats["knight"]["dps"] == 168

    s = spec("knight", 11)
    assert s.hp == pytest.approx(1766.0)
    assert s.hit_speed == pytest.approx(1.2)
    assert s.hit_dmg == pytest.approx(168 * 1.2)      # = 201.6


def test_published_damage_and_dps_times_hit_speed_agree_to_rounding(stats):
    """GOLDEN DATA (not a mechanic): the stat dump carries BOTH `damage` and `dps`, and `dps` is
    rounded to a whole number, so
    `dps * hit_speed` reconstructs `damage` only to within that rounding (Knight: 201.6 vs 202).

    Pinned because the engine builds hits from `dps` while a reader naturally expects `damage` --
    if the two ever diverge by more than rounding, the stat import has broken.
    """
    for card in TROOPS_WITH_ATTACKS:
        p = stats[card]
        reconstructed = p["dps"] * p["hit_speed"]
        assert reconstructed == pytest.approx(p["damage"], abs=p["hit_speed"] + 1e-9), card


def test_attacks_are_discrete_hits_not_continuous_damage(engine, spec):
    """CR mechanic: DISCRETE HITS. Damage lands in whole `hit_dmg` chunks on the `hit_speed` cadence;
    a target is never chipped by a fractional per-tick amount."""
    silence_towers(engine)
    musketeer = spec("musketeer")                      # 1.0s hit speed, long range
    tower = engine.towers[1][0]
    place_one(engine, 0, musketeer, tower.x, tower.y + musketeer.reach)

    run(engine, 5.0)
    dealt = tower.max_hp - tower.hp
    assert dealt > 0, "the musketeer never engaged the tower"
    hits = dealt / musketeer.tower_hit_dmg
    assert hits == pytest.approx(round(hits)), "damage must be a whole number of discrete hits"


def test_hit_cadence_matches_hit_speed(engine, spec):
    """CR mechanic: HIT SPEED. Over N seconds a troop lands about N/hit_speed hits -- Musketeer's
    1.0s cadence gives ~1 hit per second."""
    silence_towers(engine)
    musketeer = spec("musketeer")
    tower = engine.towers[1][0]
    place_one(engine, 0, musketeer, tower.x, tower.y + musketeer.reach)

    seconds = 10.0
    run(engine, seconds)
    hits = round((tower.max_hp - tower.hp) / musketeer.tower_hit_dmg)
    expected = seconds / musketeer.hit_speed
    assert abs(hits - expected) <= 1, f"{hits} hits in {seconds}s, expected ~{expected}"


def test_slow_status_halves_attack_cadence(engine, spec):
    """CR mechanic: SLOW (Ice Wizard) cuts both movement AND attack speed -- the engine divides the
    post-hit cooldown by the slow factor, so a slowed troop attacks half as often."""
    silence_towers(engine)
    musketeer = spec("musketeer")
    tower = engine.towers[1][0]
    u = place_one(engine, 0, musketeer, tower.x, tower.y + musketeer.reach)

    # keep the slow topped up so the whole window is slowed
    seconds = 10.0
    for _ in range(int(seconds / DT)):
        u.slow_left = 5.0
        engine.advance(DT)

    hits = round((tower.max_hp - tower.hp) / musketeer.tower_hit_dmg)
    expected_unslowed = seconds / musketeer.hit_speed
    assert hits == pytest.approx(expected_unslowed * engine.slow_factor, abs=1.5)


# --- tower_hit_dmg vs hit_dmg -----------------------------------------------------------------
def test_miner_deals_reduced_damage_to_crown_towers(spec, db, stats):
    """CR mechanic: CROWN TOWER DAMAGE REDUCTION. Some cards -- Miner most famously -- hit crown
    towers for far less than they hit troops. This is Miner's signature nerf and the reason an
    unanswered Miner is a chip threat, not a win condition on its own."""
    miner = spec("miner")
    published = stats["miner"]

    assert miner.hit_dmg == pytest.approx(published["dps"] * published["hit_speed"])   # 149 * 1.3
    assert miner.tower_hit_dmg == pytest.approx(78.0), "Miner's curated crown-tower damage"
    assert db.crown_tower_damage("miner") == 78
    assert miner.tower_hit_dmg < miner.hit_dmg / 2, "the reduction should be drastic, not cosmetic"


@pytest.mark.parametrize("card", ["knight", "musketeer", "valkyrie"])
def test_ordinary_troops_hit_towers_for_full_damage(spec, db, card):
    """CR mechanic: MOST troops have NO crown-tower reduction -- they hit towers exactly as hard as
    they hit troops. The reduction is the exception, so `tower_hit_dmg == hit_dmg` by default."""
    assert db.crown_tower_damage(card) is None
    s = spec(card)
    assert s.tower_hit_dmg == pytest.approx(s.hit_dmg)


def test_crown_tower_reduction_applies_to_towers_only_not_to_troops(engine, spec):
    """CR mechanic: the reduction is TARGET-DEPENDENT. The same Miner swing removes `tower_hit_dmg`
    from a crown tower but a full `hit_dmg` from a troop."""
    silence_towers(engine)
    miner = spec("miner")

    # A window SHORTER than Miner's 1.3s hit speed isolates exactly one swing in each case.
    one_swing = 1.0
    assert one_swing < miner.hit_speed

    # (a) vs a TOWER -- reduced
    tower = engine.towers[1][0]
    place_one(engine, 0, miner, tower.x, tower.y + miner.reach)
    run(engine, one_swing)
    assert tower.max_hp - tower.hp == pytest.approx(miner.tower_hit_dmg), "one reduced hit on the tower"

    # (b) vs a TROOP -- full damage
    engine.reset()
    silence_towers(engine)
    victim_spec = spec("musketeer")
    victim = place_one(engine, 1, victim_spec, 0.5, 0.5)
    freeze(victim)                                 # cannot fight back or flee; just absorbs
    attacker = place_one(engine, 0, miner, 0.5, 0.5 + miner.reach)
    run(engine, one_swing)
    assert victim_spec.hp - victim.hp == pytest.approx(miner.hit_dmg), "one FULL hit on the troop"
    assert attacker.hp > 0


def test_spell_crown_tower_damage_is_also_reduced(spec, stats):
    """CR mechanic: damage SPELLS carry their own reduced crown-tower value -- a Rocket that deletes a
    Musketeer only chips a tower. Pinned as literal golden values from the stat dump."""
    assert stats["rocket"]["damage"] == 1484
    assert stats["rocket"]["crown_tower_damage"] == 371
    rocket = spec("rocket")
    assert rocket.spell_dmg == pytest.approx(1484.0)
    assert rocket.spell_tower_dmg == pytest.approx(371.0)
    assert rocket.spell_tower_dmg == pytest.approx(rocket.spell_dmg * 0.25, rel=0.01), "Rocket is ~1/4"

    log = spec("the_log")
    assert stats["the_log"]["damage"] == 266
    assert stats["the_log"]["crown_tower_damage"] == 40
    assert log.spell_dmg == pytest.approx(266.0)
    assert log.spell_tower_dmg == pytest.approx(40.0)


def test_tower_shoots_on_its_own_discrete_cadence(engine, spec):
    """CR mechanic: CROWN TOWERS attack with discrete single-target shots, after a first-shot delay,
    and are SINGLE-TARGET (no splash) -- one troop of a pair is focused down at a time."""
    victim_spec = spec("musketeer")
    tower = engine.towers[1][0]
    # clearly different distances so the tower's nearest-target choice is unambiguous, and far enough
    # apart that soft collision never shuffles them
    a = place_one(engine, 0, victim_spec, tower.x, tower.y + 0.05)
    b = place_one(engine, 0, victim_spec, tower.x, tower.y + 0.13)
    for u in (a, b):
        park(u)                                    # stationary but still a legal tower target

    run(engine, 3.0)
    damaged = [u for u in (a, b) if u.hp < victim_spec.hp]
    assert len(damaged) == 1, "a tower focuses ONE target; splash would have hit both"

    dealt = victim_spec.hp - damaged[0].hp
    hits = dealt / tower.hit_dmg
    assert hits == pytest.approx(round(hits)), "tower damage arrives in discrete shots"
    assert hits >= 2
