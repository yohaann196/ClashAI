"""SPELLS: the three distinct delivery shapes the engine models -- Rocket's POINT BLAST, The Log's
ROLLING CORRIDOR, and Tornado's PULLING VORTEX. Confusing any two of them changes what the deck can do
(87db21c: Rocket was misclassified as a roll, which gave it the Log's narrow corridor)."""
from __future__ import annotations

import math

import pytest

from conftest import DT, freeze, place_one, run, silence_towers, total_damage_to

TILE = 0.16 / 5.5      # normalized units per board tile (the engine's `long` reach is 5.5 tiles)


def cast(engine, team, spec, x, y):
    """Cast a spell and advance just past its delay so it resolves."""
    engine.elixir[team] = 10.0
    assert engine.deploy(team, spec, x, y)
    run(engine, spec.spell_delay + DT)


# --- ROCKET: a radial point blast --------------------------------------------------------------
def test_rocket_is_a_point_blast_not_a_rolling_spell(spec):
    """CR mechanic: ROCKET IS A POINT BLAST. It detonates in a circle at the target.

    REGRESSION GUARD (87db21c): Rocket carries a `knockback` flag like The Log, and classifying rolls
    on knockback ALONE made Rocket roll -- a forward corridor with the Log's narrow half-width instead
    of its blast. Rolling additionally requires being ground-only; Rocket hits air, so it cannot roll.
    """
    rocket = spec("rocket")
    assert rocket.rolls is False, "Rocket must not be a rolling spell"
    assert rocket.ground_only is False, "Rocket hits air, which is what disqualifies it from rolling"
    assert rocket.roll_len == pytest.approx(0.0)
    assert rocket.knockback == pytest.approx(0.0)


def test_rocket_blast_radius_is_two_tiles(spec):
    """CR mechanic: ROCKET'S RADIUS IS 2 TILES. Pinned because it was once a flat 0.09 (~3 tiles),
    which over-rewarded it and would over-pay the Tornado->Rocket clump combo beyond the real game."""
    assert spec("rocket").spell_radius == pytest.approx(2.0 * TILE)


def test_rocket_damages_everything_inside_its_radius_in_every_direction(engine, spec):
    """CR mechanic: a POINT BLAST is RADIAL -- it hits units behind and beside the impact point, not
    only ahead of it. This is the property that distinguishes it from a corridor spell."""
    silence_towers(engine)
    rocket = spec("rocket")
    victim = spec("musketeer")
    cx, cy = 0.50, 0.30
    r = rocket.spell_radius

    ahead = place_one(engine, 1, victim, cx, cy - r * 0.6)
    behind = place_one(engine, 1, victim, cx, cy + r * 0.6)
    beside = place_one(engine, 1, victim, cx + r * 0.6, cy)
    outside = place_one(engine, 1, victim, cx, cy + r * 2.0)
    for u in (ahead, behind, beside, outside):
        freeze(u)

    cast(engine, 0, rocket, cx, cy)

    for u, label in ((ahead, "ahead"), (behind, "behind"), (beside, "beside")):
        assert u.hp < victim.hp, f"the blast must reach the unit {label} of the impact point"
    assert outside.hp == pytest.approx(victim.hp), "outside the radius must be untouched"


def test_rocket_hits_air_units(engine, spec):
    """CR mechanic: ROCKET HITS AIR. A corridor spell like The Log cannot."""
    silence_towers(engine)
    rocket = spec("rocket")
    flyer = spec("baby_dragon")
    assert flyer.flying is True

    u = place_one(engine, 1, flyer, 0.50, 0.30)
    freeze(u)
    cast(engine, 0, rocket, 0.50, 0.30)
    assert u.hp < flyer.hp


def test_rocket_deals_its_full_damage_to_troops(engine, spec, stats):
    """CR mechanic: a spell deals its FULL published damage to troops (the reduction is towers-only)."""
    silence_towers(engine)
    rocket = spec("rocket")
    victim = spec("musketeer")
    u = place_one(engine, 1, victim, 0.50, 0.30)
    freeze(u)
    u.hp = 5000.0                                      # survive a 1484 rocket so we can measure it

    cast(engine, 0, rocket, 0.50, 0.30)
    assert 5000.0 - u.hp == pytest.approx(stats["rocket"]["damage"])


# --- THE LOG: a forward rolling corridor -------------------------------------------------------
def test_the_log_is_a_rolling_ground_only_corridor(spec):
    """CR mechanic: THE LOG ROLLS. It sweeps a forward corridor along the ground, knocking troops back
    -- it is not a circular blast and it cannot touch air."""
    log = spec("the_log")
    assert log.rolls is True
    assert log.ground_only is True
    assert log.roll_len > 0.0
    assert log.knockback > 0.0
    assert log.spell_radius == pytest.approx(0.07), "for a roll, spell_radius is the corridor HALF-WIDTH"


def test_the_log_only_hits_forward_of_the_cast_point(engine, spec):
    """CR mechanic: a ROLL TRAVELS FORWARD (toward the enemy). Troops behind the cast point are missed,
    which is why Log placement relative to the push matters."""
    silence_towers(engine)
    log = spec("the_log")
    victim = spec("knight")
    cx, cy = 0.50, 0.60

    # team 0 rolls toward the enemy = DECREASING y
    forward = place_one(engine, 1, victim, cx, cy - log.roll_len * 0.5)
    backward = place_one(engine, 1, victim, cx, cy + 0.15)
    too_far = place_one(engine, 1, victim, cx, cy - log.roll_len * 2.0)
    for u in (forward, backward, too_far):
        freeze(u)

    cast(engine, 0, log, cx, cy)

    assert forward.hp < victim.hp, "a troop in the corridor must be hit"
    assert backward.hp == pytest.approx(victim.hp), "a troop BEHIND the roll must be missed"
    assert too_far.hp == pytest.approx(victim.hp), "beyond the roll length must be missed"


def test_the_log_corridor_has_a_narrow_half_width(engine, spec):
    """CR mechanic: the roll is a LANE, not a circle -- a troop off to the side is untouched."""
    silence_towers(engine)
    log = spec("the_log")
    victim = spec("knight")
    cx, cy = 0.50, 0.60

    inside = place_one(engine, 1, victim, cx + log.spell_radius * 0.5, cy - 0.10)
    outside = place_one(engine, 1, victim, cx + log.spell_radius * 2.0, cy - 0.10)
    freeze(inside)
    freeze(outside)

    cast(engine, 0, log, cx, cy)
    assert inside.hp < victim.hp
    assert outside.hp == pytest.approx(victim.hp), "outside the corridor half-width must be missed"


def test_the_log_cannot_hit_air(engine, spec):
    """CR mechanic: THE LOG ROLLS ALONG THE GROUND -- Minions and Baby Dragon fly straight over it."""
    silence_towers(engine)
    log = spec("the_log")
    flyer = spec("baby_dragon")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, flyer, cx, cy - 0.10)
    freeze(u)
    cast(engine, 0, log, cx, cy)
    assert u.hp == pytest.approx(flyer.hp), "a rolling spell must not damage air units"


def test_the_log_knocks_ground_troops_backward_along_the_roll(engine, spec):
    """CR mechanic: LOG KNOCKBACK. Struck ground troops are shoved in the roll direction -- for a
    defensive Log that pushes the enemy push AWAY from your tower, buying time."""
    silence_towers(engine)
    log = spec("the_log")
    victim = spec("knight")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, victim, cx, cy - 0.10)
    freeze(u)
    y_before = u.y

    cast(engine, 0, log, cx, cy)
    assert u.y == pytest.approx(y_before - log.knockback), "team 0's roll pushes toward decreasing y"


def test_the_log_rolls_the_other_way_for_the_opposing_team(engine, spec):
    """CR mechanic: 'FORWARD' IS PER-SIDE. Team 1 attacks downward, so its Log rolls toward increasing
    y -- the mirror of team 0's."""
    silence_towers(engine)
    log = spec("the_log")
    victim = spec("knight")
    cx, cy = 0.50, 0.40

    u = place_one(engine, 0, victim, cx, cy + 0.10)
    freeze(u)
    y_before = u.y

    cast(engine, 1, log, cx, cy)
    assert u.hp < victim.hp
    assert u.y == pytest.approx(y_before + log.knockback)


# --- TORNADO: an active pulling vortex ---------------------------------------------------------
def test_tornado_is_a_pulling_vortex_not_an_instant_blast(spec):
    """CR mechanic: TORNADO PULLS. It is an active area that drags enemies to its centre over its
    duration -- the clump enabler this deck is built around (clump -> Ice Wizard splash / centre
    Rocket). An instant blast could never produce that."""
    nado = spec("tornado")
    assert nado.pulls is True
    assert nado.rolls is False


def test_tornado_radius_is_five_and_a_half_tiles(spec):
    """CR mechanic: TORNADO'S PULL RADIUS is 5.5 tiles -- far wider than a damage spell, which is what
    lets one cast gather a whole push."""
    nado = spec("tornado")
    assert nado.pull_radius == pytest.approx(5.5 * TILE)
    assert nado.pull_radius == pytest.approx(0.16)
    assert nado.pull_radius > spec("rocket").spell_radius * 2


def test_tornado_vortex_lasts_about_one_second(engine, spec):
    """CR mechanic: TORNADO DURATION (~1.05s). The vortex is active for a fixed window and then gone."""
    nado = spec("tornado")
    assert nado.pull_duration == pytest.approx(1.05)

    cast(engine, 0, nado, 0.50, 0.60)
    assert len(engine.vortices) == 1, "the vortex must be live right after the spell lands"

    run(engine, nado.pull_duration + 0.2)
    assert engine.vortices == [], "the vortex must expire after its duration"


def test_tornado_drags_enemies_toward_its_centre(engine, spec):
    """CR mechanic: THE PULL. Enemies inside the radius are dragged toward the vortex centre; the drag
    rate is ~0.35 normalized units/second (~12 tiles/s), fast enough to gather a push within the
    duration."""
    silence_towers(engine)
    nado = spec("tornado")
    victim = spec("musketeer")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, victim, cx, cy + 0.15)     # 0.15 away, inside the 0.16 radius
    freeze(u)                                            # so only the vortex moves it
    d_before = math.dist((u.x, u.y), (cx, cy))

    engine.elixir[0] = 10.0
    engine.deploy(0, nado, cx, cy)
    run(engine, nado.spell_delay + DT)                   # land it + exactly one vortex tick
    d_after = math.dist((u.x, u.y), (cx, cy))

    assert d_after < d_before, "the vortex must pull the enemy inward"
    assert d_before - d_after == pytest.approx(0.35 * DT, rel=0.02), "~0.35/s drag rate"


def test_tornado_drags_heavy_tanks_at_half_speed(engine, spec):
    """CR mechanic: HEAVY UNITS RESIST THE PULL. A Golem/Giant-class tank is dragged at half rate, so
    a Tornado cannot yank a tank across the arena the way it repositions a swarm."""
    silence_towers(engine)
    nado = spec("tornado")
    tank = spec("giant")
    assert tank.radius >= 0.03, "this test needs a card the engine treats as heavy"
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, tank, cx, cy + 0.15)
    freeze(u)
    d_before = math.dist((u.x, u.y), (cx, cy))

    engine.elixir[0] = 10.0
    engine.deploy(0, nado, cx, cy)
    run(engine, nado.spell_delay + DT)
    moved = d_before - math.dist((u.x, u.y), (cx, cy))

    assert moved == pytest.approx(0.35 * DT * 0.5, rel=0.02), "tanks are pulled at half rate"


def test_tornado_leaves_units_outside_its_radius_alone(engine, spec):
    """CR mechanic: the pull has a HARD EDGE -- outside the radius there is no drag and no damage."""
    silence_towers(engine)
    nado = spec("tornado")
    victim = spec("musketeer")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, victim, cx, cy + nado.pull_radius * 2)
    freeze(u)
    before = (u.x, u.y, u.hp)

    cast(engine, 0, nado, cx, cy)
    run(engine, nado.pull_duration + 0.2)
    assert (u.x, u.y, u.hp) == pytest.approx(before)


def test_tornado_pulls_air_units_too(engine, spec):
    """CR mechanic: TORNADO AFFECTS AIR AND GROUND -- unlike The Log, flyers are pulled as well."""
    silence_towers(engine)
    nado = spec("tornado")
    flyer = spec("baby_dragon")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, flyer, cx, cy + 0.12)
    freeze(u)
    d_before = math.dist((u.x, u.y), (cx, cy))

    cast(engine, 0, nado, cx, cy)
    assert math.dist((u.x, u.y), (cx, cy)) < d_before


def test_tornado_total_damage_equals_its_spell_damage(engine, spec, stats):
    """CR mechanic: TORNADO'S DAMAGE IS SPREAD OVER ITS DURATION (damage-over-time), and the total
    across the whole vortex equals the card's published damage -- no more, no less.

    The final tick is PRO-RATED against the remaining duration precisely so the total lands exactly on
    the published figure instead of overshooting by a partial tick.
    """
    silence_towers(engine)
    nado = spec("tornado")
    victim = spec("musketeer")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, victim, cx, cy + 0.05)
    freeze(u)

    cast(engine, 0, nado, cx, cy)
    run(engine, nado.pull_duration + 0.5)               # well past expiry

    assert engine.vortices == []
    assert total_damage_to(u, victim) == pytest.approx(stats["tornado"]["damage"], rel=1e-6)


def test_tornado_damage_arrives_gradually_not_all_at_once(engine, spec, stats):
    """CR mechanic: DAMAGE OVER TIME. Partway through the vortex a caught unit has taken only part of
    the total -- which is why Tornado alone rarely kills, and needs the splash follow-up."""
    silence_towers(engine)
    nado = spec("tornado")
    victim = spec("musketeer")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 1, victim, cx, cy + 0.05)
    freeze(u)

    cast(engine, 0, nado, cx, cy)                       # lands + one tick
    partial = total_damage_to(u, victim)
    full = stats["tornado"]["damage"]
    assert 0 < partial < full, "one tick must deal a fraction of the total"
    assert partial == pytest.approx(full * DT / nado.pull_duration, rel=1e-6)


def test_tornado_only_chips_a_tower_when_cast_directly_on_it(engine, spec, stats):
    """CR mechanic: TORNADO IS NOT A CHIP SPELL. Its crown-tower damage is negligible and only applies
    when the cast point overlaps the tower -- its value is repositioning UNITS, not tower damage."""
    silence_towers(engine)
    nado = spec("tornado")
    tower = engine.towers[1][0]

    cast(engine, 0, nado, tower.x, tower.y)
    on_tower = tower.max_hp - tower.hp
    assert on_tower == pytest.approx(stats["tornado"]["crown_tower_damage"])
    assert on_tower < 50, "the crown chip must stay negligible"

    engine.reset()
    silence_towers(engine)
    tower = engine.towers[1][0]
    cast(engine, 0, nado, tower.x, tower.y + 0.20)      # a normal defensive cast, away from the tower
    assert tower.hp == pytest.approx(tower.max_hp), "a cast away from the tower must not chip it"


def test_tornado_does_not_pull_your_own_troops(engine, spec):
    """CR mechanic: a spell only affects ENEMIES -- your own defenders are not dragged into your own
    Tornado."""
    silence_towers(engine)
    nado = spec("tornado")
    friendly = spec("musketeer")
    cx, cy = 0.50, 0.60

    u = place_one(engine, 0, friendly, cx, cy + 0.10)   # same team as the caster
    freeze(u)
    before = (u.x, u.y, u.hp)

    cast(engine, 0, nado, cx, cy)
    run(engine, nado.pull_duration + 0.2)
    assert (u.x, u.y, u.hp) == pytest.approx(before)


def test_tornado_gathers_a_spread_out_push_into_a_clump(engine, spec):
    """CR mechanic: THE CLUMP. The whole point of the card -- several separated enemies end up bunched
    at the vortex centre, which is what turns Ice Wizard's splash into an everything-hitter and lets a
    centre Rocket catch the entire push."""
    silence_towers(engine)
    nado = spec("tornado")
    victim = spec("musketeer")
    cx, cy = 0.50, 0.60

    spread = [
        place_one(engine, 1, victim, cx - 0.10, cy),
        place_one(engine, 1, victim, cx + 0.10, cy),
        place_one(engine, 1, victim, cx, cy + 0.12),
    ]
    for u in spread:
        freeze(u)

    def max_separation():
        return max(math.dist((a.x, a.y), (b.x, b.y)) for a in spread for b in spread)

    before = max_separation()
    cast(engine, 0, nado, cx, cy)
    run(engine, nado.pull_duration)
    after = max_separation()

    assert after < before / 2, f"the push should be clumped: spread {before:.3f} -> {after:.3f}"


def test_spells_land_after_their_cast_delay(engine, spec):
    """CR mechanic: SPELL TRAVEL TIME. A spell does not resolve the instant you tap -- there is a
    short delay, which is what makes predictive casting a skill."""
    silence_towers(engine)
    rocket = spec("rocket")
    victim = spec("musketeer")
    u = place_one(engine, 1, victim, 0.50, 0.30)
    freeze(u)

    engine.elixir[0] = 10.0
    engine.deploy(0, rocket, 0.50, 0.30)
    assert len(engine.spells) == 1, "the spell must be in flight, not resolved"

    run(engine, rocket.spell_delay - DT)
    assert u.hp == pytest.approx(victim.hp), "no damage before the spell lands"

    run(engine, 2 * DT)
    assert u.hp < victim.hp, "damage lands once the delay elapses"
    assert engine.spells == []
