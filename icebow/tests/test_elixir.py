"""ELIXIR ECONOMY: regeneration rate, the 1x/2x/3x tiers, and the overtime transitions."""
from __future__ import annotations

import pytest

from conftest import run

SINGLE_S = 2.8      # seconds per elixir in regulation
DOUBLE_S = 1.4      # ...from 60s remaining
TRIPLE_S = 0.93     # ...in overtime


def test_single_elixir_is_one_per_2_8_seconds(engine):
    """CR mechanic: SINGLE ELIXIR. For the first two minutes one elixir accrues every 2.8s."""
    engine.t = 0.0
    assert engine.elixir_rate() == pytest.approx(1.0 / SINGLE_S)

    engine.elixir[0] = 0.0
    engine.advance(SINGLE_S)
    assert engine.elixir[0] == pytest.approx(1.0)


def test_double_elixir_starts_at_sixty_seconds_remaining(engine):
    """CR mechanic: DOUBLE ELIXIR. With 60s left in regulation the rate doubles to one per 1.4s."""
    engine.t = engine.regulation - 60.0
    assert engine.elixir_rate() == pytest.approx(1.0 / DOUBLE_S)
    assert engine.elixir_rate() == pytest.approx(2.0 * (1.0 / SINGLE_S)), "double must be exactly 2x single"

    engine.elixir[0] = 0.0
    engine.advance(DOUBLE_S)
    assert engine.elixir[0] == pytest.approx(1.0)


def test_triple_elixir_in_overtime(engine):
    """CR mechanic: TRIPLE ELIXIR. Once regulation expires and overtime begins, one elixir per 0.93s.

    Note 0.93 is the figure the game quotes, not an exact third of 2.8 (which would be 0.9333) -- so
    triple elixir is fractionally FASTER than exactly 3x. Pinned so nobody 'tidies' it to 2.8/3.
    """
    engine.t = engine.regulation
    assert engine.elixir_rate() == pytest.approx(1.0 / TRIPLE_S)
    assert engine.elixir_rate() > 3.0 * (1.0 / SINGLE_S)
    assert engine.elixir_rate() == pytest.approx(3.0 * (1.0 / SINGLE_S), rel=0.005)

    engine.elixir[0] = 0.0
    engine.advance(TRIPLE_S)
    assert engine.elixir[0] == pytest.approx(1.0)


@pytest.mark.parametrize("t,expected_s,tier", [
    (0.0, SINGLE_S, "single"),
    (119.9, SINGLE_S, "single"),
    (120.0, DOUBLE_S, "double"),     # regulation(180) - 60 -> the double-elixir boundary
    (179.9, DOUBLE_S, "double"),
    (180.0, TRIPLE_S, "triple"),     # regulation expires -> overtime
    (239.9, TRIPLE_S, "triple"),
])
def test_elixir_tier_transitions_are_inclusive_at_the_boundary(engine, t, expected_s, tier):
    """CR mechanic: the ELIXIR TIER BOUNDARIES. Each tier starts exactly ON its boundary second."""
    engine.t = t
    assert engine.elixir_rate() == pytest.approx(1.0 / expected_s), f"t={t} should be {tier}"


def test_elixir_is_capped_at_ten(engine):
    """CR mechanic: the ELIXIR BAR CAPS AT 10 -- surplus regeneration is lost, not banked."""
    engine.elixir[0] = 9.9
    run(engine, 30.0)
    assert engine.elixir[0] == pytest.approx(10.0)


def test_both_teams_regenerate_identically(engine):
    """CR mechanic: elixir regeneration is SYMMETRIC -- neither side has an economy edge."""
    engine.elixir[0] = engine.elixir[1] = 0.0
    run(engine, 10.0)
    assert engine.elixir[0] == pytest.approx(engine.elixir[1])
    assert engine.elixir[0] > 0.0


def test_both_sides_start_at_five_elixir(engine):
    """CR mechanic: a match OPENS AT 5 ELIXIR for both players."""
    assert engine.elixir[0] == pytest.approx(5.0)
    assert engine.elixir[1] == pytest.approx(5.0)


def test_deploy_debits_the_card_cost_and_is_refused_when_unaffordable(engine, spec):
    """CR mechanic: a card costs its ELIXIR to play, and cannot be played without the elixir."""
    rocket = spec("rocket")            # 6 elixir
    engine.elixir[0] = 5.9
    assert engine.deploy(0, rocket, 0.5, 0.3) is False, "must refuse below the card cost"
    assert engine.elixir[0] == pytest.approx(5.9), "a refused deploy must not debit elixir"

    engine.elixir[0] = 6.0
    assert engine.deploy(0, rocket, 0.5, 0.3) is True
    assert engine.elixir[0] == pytest.approx(0.0)


def test_match_ends_after_regulation_plus_overtime(engine):
    """CR mechanic: a match runs REGULATION then OVERTIME, then ends and scores on crowns."""
    engine.t = engine.regulation + engine.overtime - 0.05
    engine.advance(0.1)
    assert engine.done is True
    assert engine.outcome in {"win", "loss", "draw"}


def test_elixir_stops_accruing_once_the_match_is_over(engine):
    """CR mechanic: the match CLOCK STOPS at the end -- no post-match economy."""
    engine.t = engine.regulation + engine.overtime
    engine.advance(0.1)
    assert engine.done is True

    engine.elixir[0] = 3.0
    run(engine, 5.0)
    assert engine.elixir[0] == pytest.approx(3.0), "advance() must be a no-op once done"
