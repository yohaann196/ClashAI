"""CARD LEVELS and the two damage-mitigation mechanics: SHIELD POOLS and Evo Knight's DAMAGE REDUCTION.

The two mitigations are deliberately different and were once conflated (5440e7b modelled Evo Knight as
a shield pool before it was corrected to damage reduction), so each is pinned separately here.
"""
from __future__ import annotations

import pytest

from conftest import place, place_one, run, silence_towers

SCALED_CARDS = ["knight", "musketeer", "miner", "royal_recruits"]
REFERENCE_LEVEL = 11        # cards_stats.json is a level-11 dump


# --- level scaling ----------------------------------------------------------------------------
def test_stat_dump_is_a_level_eleven_reference(stats_meta):
    """GOLDEN DATA (not a mechanic): the stat dump is LEVEL 11 -- every scaling assertion below is
    relative to that reference level."""
    assert stats_meta["level"] == REFERENCE_LEVEL


@pytest.mark.parametrize("card", SCALED_CARDS)
@pytest.mark.parametrize("level", [9, 10, 11, 12, 13, 14, 15, 16])
def test_hitpoints_scale_by_1_1_per_level(spec, stats, card, level):
    """CR mechanic: LEVEL SCALING. Each card level multiplies HITPOINTS by 1.1, compounding from the
    card's reference level -- so a level-14 card has 1.1^3 the HP of the same card at 11."""
    expected = stats[card]["hitpoints"] * (1.1 ** (level - REFERENCE_LEVEL))
    assert spec(card, level).hp == pytest.approx(expected)


@pytest.mark.parametrize("card", SCALED_CARDS)
@pytest.mark.parametrize("level", [9, 11, 14, 16])
def test_hit_damage_scales_by_1_1_per_level(spec, stats, card, level):
    """CR mechanic: LEVEL SCALING applies to DAMAGE on the same 1.1x curve as hitpoints."""
    expected = stats[card]["dps"] * stats[card]["hit_speed"] * (1.1 ** (level - REFERENCE_LEVEL))
    assert spec(card, level).hit_dmg == pytest.approx(expected)


@pytest.mark.parametrize("card", SCALED_CARDS)
@pytest.mark.parametrize("level", [9, 14, 16])
def test_hit_speed_and_movement_never_scale_with_level(spec, card, level):
    """CR mechanic: LEVEL AFFECTS HP AND DAMAGE ONLY. Hit speed, movement speed, range and elixir cost
    are identical at every level -- a maxed card is not a faster card."""
    base = spec(card, REFERENCE_LEVEL)
    lifted = spec(card, level)
    assert lifted.hit_speed == pytest.approx(base.hit_speed)
    assert lifted.speed == pytest.approx(base.speed)
    assert lifted.reach == pytest.approx(base.reach)
    assert lifted.elixir == base.elixir


def test_one_level_is_exactly_ten_percent(spec):
    """CR mechanic: the scaling step is exactly 1.1x per level -- pinned as a literal."""
    a, b = spec("knight", 11), spec("knight", 12)
    assert b.hp / a.hp == pytest.approx(1.1)
    assert b.hit_dmg / a.hit_dmg == pytest.approx(1.1)


def test_spell_damage_scales_with_level(spec, stats):
    """CR mechanic: SPELLS SCALE TOO -- both their troop damage and their reduced crown-tower damage."""
    lvl = 14
    sc = 1.1 ** (lvl - REFERENCE_LEVEL)
    rocket = spec("rocket", lvl)
    assert rocket.spell_dmg == pytest.approx(stats["rocket"]["damage"] * sc)
    assert rocket.spell_tower_dmg == pytest.approx(stats["rocket"]["crown_tower_damage"] * sc)


def test_crown_tower_damage_scales_with_level(spec):
    """CR mechanic: a troop's reduced crown-tower damage scales on the SAME curve as its normal hit."""
    a, b = spec("miner", 11), spec("miner", 13)
    assert b.tower_hit_dmg / a.tower_hit_dmg == pytest.approx(1.1 ** 2)


def test_a_higher_level_card_beats_the_same_card_at_a_lower_level(engine, spec):
    """CR mechanic: level advantage is REAL -- the same card at a higher level wins the mirror fight."""
    silence_towers(engine)
    strong = spec("knight", 14)
    weak = spec("knight", 11)
    a = place_one(engine, 0, strong, 0.50, 0.50)
    b = place_one(engine, 1, weak, 0.50, 0.50 + strong.reach)

    run(engine, 30.0)
    assert b.hp <= 0 or b not in engine.units, "the lower-level Knight should lose the mirror"
    assert a.hp > 0


# --- shield pools -----------------------------------------------------------------------------
def test_royal_recruits_carry_a_shield_pool(spec, stats):
    """CR mechanic: SHIELDS. Royal Recruits (and Guards) spawn behind a shield that must be broken
    before their body takes damage."""
    rr = spec("royal_recruits")
    assert rr.shield_hp > 0.0
    assert rr.shield_hp == pytest.approx(stats["royal_recruits"]["hitpoints"] * 0.5)


def test_a_unit_starts_with_its_shield_full(engine, spec):
    """CR mechanic: the shield is present ON SPAWN, not earned."""
    rr = spec("royal_recruits")
    u = place(engine, 0, rr, 0.50, 0.60)[0]
    assert u.shield_left == pytest.approx(rr.shield_hp)
    assert u.hp == pytest.approx(rr.hp)


def test_shield_absorbs_damage_before_hitpoints(engine, spec):
    """CR mechanic: the SHIELD SOAKS FIRST. Body HP is untouched while the shield holds."""
    rr = spec("royal_recruits")
    u = place(engine, 0, rr, 0.5, 0.6)[0]

    engine._hurt(u, 50.0)
    assert u.hp == pytest.approx(rr.hp), "body HP must be untouched while the shield holds"
    assert u.shield_left == pytest.approx(rr.shield_hp - 50.0)


def test_breaking_a_shield_discards_the_overflow(engine, spec):
    """CR mechanic: SHIELD OVERFLOW IS LOST. A single huge hit only STRIPS the shield -- the excess does
    not carry into body HP. This is why a cheap chip spell can pop a shield as effectively as a Rocket,
    and why a shielded troop survives one oversized hit."""
    rr = spec("royal_recruits")
    u = place(engine, 0, rr, 0.5, 0.6)[0]

    engine._hurt(u, 1_000_000.0)
    assert u.shield_left == pytest.approx(0.0), "the shield is stripped"
    assert u.hp == pytest.approx(rr.hp), "the overflow must NOT carry into hitpoints"

    engine._hurt(u, 100.0)                                   # the next hit lands on the body
    assert u.hp == pytest.approx(rr.hp - 100.0)


def test_unshielded_units_take_damage_straight_to_hitpoints(engine, spec):
    """CR mechanic: most cards have NO shield -- damage goes straight to HP."""
    knight = spec("knight")
    u = place_one(engine, 0, knight, 0.5, 0.6)
    assert u.shield_left == pytest.approx(0.0)

    engine._hurt(u, 100.0)
    assert u.hp == pytest.approx(knight.hp - 100.0)


# --- Evo Knight damage reduction --------------------------------------------------------------
def test_evo_knight_has_damage_reduction_not_a_shield(spec):
    """CR mechanic: EVO KNIGHT'S EVOLUTION IS DAMAGE REDUCTION (60% less from ALL sources while not
    attacking) -- explicitly NOT a numerical shield pool.

    REGRESSION GUARD (5440e7b): it was first modelled as a ~0.5x HP shield, which is a different
    mechanic with different counterplay, and had to be reverted.
    """
    evo = spec("knight_evo")
    assert evo.damage_reduction == pytest.approx(0.60)
    assert evo.shield_hp == pytest.approx(0.0), "damage reduction is not a shield pool"

    plain = spec("knight")
    assert plain.damage_reduction == pytest.approx(0.0)
    assert plain.hp == pytest.approx(evo.hp), "the evolution grants no extra hitpoints"


def test_evo_knight_takes_reduced_damage_while_not_attacking(engine, spec):
    """CR mechanic: the reduction is ACTIVE WHILE MOVING/APPROACHING -- 60% less damage, so an Evo
    Knight walking into a defence is far harder to kill than a plain one."""
    evo = spec("knight_evo")
    u = place_one(engine, 0, evo, 0.5, 0.6)
    u.attacking = False

    engine._hurt(u, 100.0)
    assert evo.hp - u.hp == pytest.approx(40.0), "60% reduction -> 40 of a 100 hit lands"


def test_evo_knight_takes_full_damage_while_attacking(engine, spec):
    """CR mechanic: the reduction DROPS THE INSTANT IT ENGAGES. While swinging it takes full damage --
    the window that makes it killable."""
    evo = spec("knight_evo")
    u = place_one(engine, 0, evo, 0.5, 0.6)
    u.attacking = True

    engine._hurt(u, 100.0)
    assert evo.hp - u.hp == pytest.approx(100.0), "no reduction while attacking"


def test_plain_knight_always_takes_full_damage(engine, spec):
    """CR mechanic: the NON-evolved Knight has no reduction in either state -- the control case."""
    plain = spec("knight")
    for attacking in (False, True):
        engine.reset()
        u = place_one(engine, 0, plain, 0.5, 0.6)
        u.attacking = attacking
        engine._hurt(u, 100.0)
        assert plain.hp - u.hp == pytest.approx(100.0)


def test_the_attacking_flag_turns_on_when_a_target_comes_into_reach(engine, spec):
    """CR mechanic: the reduction toggles off ENGAGEMENT, driven by the simulation rather than set by
    hand -- a marching Evo Knight is protected, one in melee is not."""
    silence_towers(engine)
    evo = spec("knight_evo")

    # (a) marching with nothing in reach -> protected
    walker = place_one(engine, 0, evo, 0.50, 0.60)
    run(engine, 0.5)
    assert walker.attacking is False, "no target in reach -> reduction ON"

    # (b) toe to toe with an enemy -> engaged
    engine.reset()
    silence_towers(engine)
    fighter = place_one(engine, 0, evo, 0.50, 0.50)
    foe = place_one(engine, 1, spec("musketeer"), 0.50, 0.50 + evo.reach)
    foe.hp = 1e9                                    # keep the fight going so the flag stays set
    run(engine, 0.5)
    assert fighter.attacking is True, "target in reach -> reduction OFF"


def test_evo_knight_survives_longer_than_a_plain_knight_under_fire(engine, spec):
    """CR mechanic: the END-TO-END payoff -- with identical HP, the evolution's reduction means it
    absorbs materially more damage before dying while approaching."""
    evo, plain = spec("knight_evo"), spec("knight")
    assert evo.hp == pytest.approx(plain.hp)

    hit, dealt = 100.0, 0.0
    e = place_one(engine, 0, evo, 0.5, 0.6)
    e.attacking = False
    while e.hp > 0 and dealt < 1e6:
        engine._hurt(e, hit)
        dealt += hit
    evo_absorbed = dealt

    engine.reset()
    p = place_one(engine, 0, plain, 0.5, 0.6)
    p.attacking = False
    dealt = 0.0
    while p.hp > 0 and dealt < 1e6:
        engine._hurt(p, hit)
        dealt += hit

    assert evo_absorbed == pytest.approx(dealt / 0.4, rel=0.02), "it should soak ~2.5x the damage"
