"""OPPONENT ELIXIR + CYCLE estimation (clashrl/opp_elixir.py).

Two groups of tests, and the second group is the important one:

  1. the estimator's own arithmetic -- integration, the deploy-delay correction, the cap re-sync, how
     the interval widens with detector recall;
  2. SIM PARITY -- that the simulator runs this estimator on CORRUPTED inputs rather than being handed
     engine ground truth. That is the whole point of the design: a policy trained on exact opponent
     elixir would lean on precision it does not have live, and the failure would only appear on the
     ladder. Also that the corruption's private RNG leaves seeded eval streams bit-identical.
"""
from __future__ import annotations

import pytest

from clashrl.cycle import cycle_vector
from clashrl.opp_elixir import (OPP_ELIXIR_DIM, DeployObserver, EnemyCycleTracker, OpponentElixir,
                                features)
from clashrl.sim.env import SimMatchEnv

SINGLE_RATE = 1.0 / 2.8


@pytest.fixture()
def est(cfg, db) -> OpponentElixir:
    cfg.data["observation"].setdefault("opponent_elixir", {}).update(
        {"assumed_recall": 1.0, "blind_spend_frac": 0.0})   # a perfect, fully-covering detector
    return OpponentElixir(cfg, db)


@pytest.fixture()
def noisy_est(cfg, db) -> OpponentElixir:
    """An estimator sized for a REALISTIC detector (recall 0.55), where the interval is wide."""
    cfg.data["observation"].setdefault("opponent_elixir", {}).update(
        {"assumed_recall": 0.55, "blind_spend_frac": 0.0})   # isolate the RECALL term in these tests
    return OpponentElixir(cfg, db)


# --- 1. the arithmetic --------------------------------------------------------------------------
def test_a_match_starts_at_five_elixir_known_exactly(est):
    """CR: both players open at 5. That is the one moment the estimate is certain."""
    assert est.lo == pytest.approx(5.0)
    assert est.hi == pytest.approx(5.0)
    assert est.confidence == pytest.approx(1.0)


def test_elixir_integrates_at_the_known_rate(est):
    """The dynamics are not guessed: rate is identical for both players and already modelled."""
    est.regen(2.8, SINGLE_RATE)
    assert est.mid == pytest.approx(6.0)
    est.regen(2.8, SINGLE_RATE)
    assert est.mid == pytest.approx(7.0)


def test_an_observed_deploy_subtracts_its_cost(est, db):
    """The only unknown in e(t) is which cards they played; observing one removes that unknown."""
    est.regen(2.8, SINGLE_RATE)                     # -> 6.0
    est.observe_deploy("hog_rider", SINGLE_RATE)    # Hog Rider costs 4
    assert est.mid == pytest.approx(6.0 - float(db.elixir("hog_rider")))


def test_the_estimate_is_clamped_to_the_bar(est):
    """CR: the bar holds 10 and cannot go negative -- overflow is genuinely lost."""
    for _ in range(200):
        est.regen(0.1, SINGLE_RATE)
    assert est.hi == pytest.approx(10.0)

    est.observe_deploy("rocket", SINGLE_RATE)       # 6
    est.observe_deploy("rocket", SINGLE_RATE)       # another 6 -> would go negative
    assert est.lo >= 0.0 and est.hi >= 0.0


def test_deploy_delay_is_corrected_at_the_cap(cfg, db):
    """A troop is visible ~1s AFTER the elixir left the bar. Applying the cost at DETECTION time while
    the estimate rides the 10-clamp would let the clamp eat elixir that was really spent, so the
    estimator rewinds the regen, applies the cost there, then rolls forward again.

    Opponent capped at 10, spends 4: truth is 6 + one second of regen. Naively clamping first would
    give exactly 6 and quietly lose the regen.
    """
    cfg.data["observation"].setdefault("opponent_elixir", {}).update(
        {"assumed_recall": 1.0, "deploy_delay_s": 1.0, "blind_spend_frac": 0.0})
    e = OpponentElixir(cfg, db)
    for _ in range(300):
        e.regen(0.1, SINGLE_RATE, quiet=True)         # an empty board: genuinely idle at the cap
    assert e.hi == pytest.approx(10.0)

    e.observe_deploy("hog_rider", SINGLE_RATE)      # cost 4
    # spend happened a second ago at the cap: 10 - 4, plus the second of regen since
    assert e.mid == pytest.approx(10.0 - 4.0 + SINGLE_RATE * 1.0, abs=1e-6)


def test_sustained_idle_at_the_cap_collapses_the_interval(noisy_est, cfg):
    """SELF-CORRECTION. The 10-elixir cap is an ABSORBING STATE that erases history -- after enough
    idle time they are at 10 no matter what we missed earlier. Without this, missed deploys would
    accumulate into permanent uncertainty over a 3-minute match."""
    noisy_est.observe_deploy("rocket", SINGLE_RATE)
    assert noisy_est.confidence < 1.0, "a spend under imperfect recall must widen the interval"

    for _ in range(400):                             # ~40s with an EMPTY board -> they are at 10
        noisy_est.regen(0.1, SINGLE_RATE, quiet=True)
    assert noisy_est.lo == pytest.approx(10.0)
    assert noisy_est.hi == pytest.approx(10.0)
    assert noisy_est.confidence == pytest.approx(1.0)


def test_the_cap_resync_needs_an_empty_board_not_merely_no_detections(noisy_est):
    """The distinction that makes the cap re-sync sound rather than dangerous.

    "No deploy OBSERVED" is not "no deploy HAPPENED". With a whitelist covering a slice of the meta the
    estimator can observe nothing for an entire match while the opponent plays continuously -- measured:
    0 of 33 deploys observed on a sampled sim rollout. Treating that silence as idleness would snap the
    interval confidently to 10 exactly when it is most wrong, so the re-sync requires EVIDENCE of quiet.
    """
    noisy_est.observe_deploy("rocket", SINGLE_RATE)
    for _ in range(400):                              # enemy units on the board the whole time
        noisy_est.regen(0.1, SINGLE_RATE, quiet=False)
    assert noisy_est.lo < 10.0 - 1e-6, "a busy board must NOT be read as idle-at-the-cap"

    for _ in range(400):                              # ...now the board actually clears
        noisy_est.regen(0.1, SINGLE_RATE, quiet=True)
    assert noisy_est.lo == pytest.approx(10.0)


def test_the_interval_widens_with_worse_detector_recall(cfg, db):
    """The interval is not a vibe -- its width is sized from the detector's MEASURED recall, so a
    weaker detector produces a visibly less confident estimate rather than a confidently wrong one."""
    widths = {}
    for recall in (1.0, 0.8, 0.5):
        cfg.data["observation"].setdefault("opponent_elixir", {}).update(
            {"assumed_recall": recall, "blind_spend_frac": 0.0})
        e = OpponentElixir(cfg, db)
        e.regen(20.0, SINGLE_RATE)
        e.observe_deploy("hog_rider", SINGLE_RATE)
        widths[recall] = e.hi - e.lo
    assert widths[1.0] == pytest.approx(0.0), "a perfect detector leaves no room for missed spends"
    assert widths[0.5] > widths[0.8] > widths[1.0]


def test_perfect_recall_gives_a_point_estimate(est):
    """At recall 1.0 there is nothing to have missed, so the interval degenerates to a point."""
    est.regen(10.0, SINGLE_RATE)
    est.observe_deploy("knight", SINGLE_RATE)
    assert est.lo == pytest.approx(est.hi)
    assert est.confidence == pytest.approx(1.0)


def test_an_unnameable_spawn_still_counts_as_a_spend(est):
    """Seeing a spawn we cannot NAME still tells us elixir left their bar. Ignoring it would leave the
    estimate confidently too high; the honest response is to widen, not to pretend."""
    est.regen(20.0, SINGLE_RATE)
    before = est.hi
    est.observe_deploy(None, SINGLE_RATE)
    assert est.hi < before, "an unnamed spawn must still reduce the upper bound"
    assert est.lo < est.hi, "...and must widen the interval, since the cost is unknown"


def test_can_afford_is_graded_off_the_interval(cfg, db):
    """'Can they pay for this' has to be answered conservatively: certainly-yes, certainly-no, and an
    honest fraction in between rather than a coin flip on the midpoint."""
    cfg.data["observation"].setdefault("opponent_elixir", {}).update(
        {"assumed_recall": 1.0, "blind_spend_frac": 0.0})
    e = OpponentElixir(cfg, db)
    e.regen(20.0, SINGLE_RATE)                        # pinned at 10
    assert e.can_afford(6.0) == pytest.approx(1.0)

    e._lo, e._hi = 0.0, 2.0                           # certainly cannot afford a Rocket
    assert e.can_afford(6.0) == pytest.approx(0.0)

    e._lo, e._hi = 2.0, 8.0                           # genuinely uncertain
    assert 0.0 < e.can_afford(6.0) < 1.0


# --- the enemy cycle ----------------------------------------------------------------------------
def test_enemy_cycle_bootstraps_from_a_completely_unknown_deck(db):
    """Unlike your own queue, NOTHING about the opponent's hand is observable -- so it starts as all
    `-1` unknowns, which `cycle_vector` already tolerates."""
    c = EnemyCycleTracker(db)
    assert c.known_cards == []
    assert c.coverage == pytest.approx(0.0)
    assert all(v == -1 for v in c._queue)
    assert cycle_vector(c._queue, 8).sum() == pytest.approx(0.0), "unknown slots contribute nothing"


def test_seeing_cards_played_fills_in_their_deck(db):
    """The only observable is what they play, so the deck is learned one card at a time."""
    c = EnemyCycleTracker(db)
    for card in ("hog_rider", "musketeer", "knight", "archers"):
        c.record_play(card)
    assert set(c.known_cards) == {"hog_rider", "musketeer", "knight", "archers"}
    assert c.coverage == pytest.approx(0.5)


def test_a_played_card_rotates_to_the_back_of_their_queue(db):
    """CR: playing a card sends it to the BACK of the 8-card queue. That is what makes the cycle
    predictable at all."""
    c = EnemyCycleTracker(db)
    c.record_play("hog_rider")
    assert c._queue[-1] == c._cards.index("hog_rider")

    c.record_play("musketeer")
    assert c._queue[-1] == c._cards.index("musketeer")
    assert c._queue[-2] == c._cards.index("hog_rider"), "the earlier play sits just ahead of it"


def test_a_just_played_card_is_furthest_from_coming_back(db):
    """Soonness is the useful read: a card just played is 7 plays away, so it is NOT a threat now."""
    c = EnemyCycleTracker(db)
    for card in ("hog_rider", "musketeer", "knight", "archers", "fireball"):
        c.record_play(card)
    assert c.soonness("fireball") < c.soonness("hog_rider"), \
        "the most recently played card must be the furthest away"


def test_wincon_soonness_needs_their_wincon_to_have_been_seen(db):
    """Until they show a win condition we know nothing about it -- reporting 0 is honest, not a guess."""
    c = EnemyCycleTracker(db)
    assert c.wincon() is None and c.wincon_soonness() == pytest.approx(0.0)

    c.record_play("hog_rider")                       # a win condition
    assert c.wincon() == "hog_rider"


def test_more_than_eight_distinct_cards_is_rejected_as_detector_noise(db):
    """A deck has 8 cards. A ninth means the detector misread something, and accepting it would corrupt
    the cycle model permanently."""
    c = EnemyCycleTracker(db)
    for card in ("hog_rider", "musketeer", "knight", "archers", "fireball", "zap", "cannon", "ice_spirit"):
        c.record_play(card)
    assert c.coverage == pytest.approx(1.0)
    c.record_play("golem")
    assert "golem" not in c.known_cards
    assert len(c.known_cards) == 8


# --- the feature block --------------------------------------------------------------------------
def test_feature_block_has_the_declared_width(est, db):
    v = features(est, EnemyCycleTracker(db), my_elixir=5.0)
    assert v.shape == (OPP_ELIXIR_DIM,)
    assert ((v >= 0.0) & (v <= 1.0)).all(), "every feature must be normalized for the net"


def test_elixir_advantage_is_centred_and_signed(est, db):
    """The headline derived feature: 0.5 = level, >0.5 = you are ahead. Centring matters because the
    policy has to read 'I am up 4 elixir, commit' off it."""
    c = EnemyCycleTracker(db)
    assert features(est, c, my_elixir=5.0)[3] == pytest.approx(0.5)      # both at 5
    assert features(est, c, my_elixir=10.0)[3] > 0.5
    assert features(est, c, my_elixir=0.0)[3] < 0.5


def test_can_afford_wincon_feature_tracks_the_estimate(cfg, db):
    """The other derived feature the policy needs: is their answer/wincon actually payable right now."""
    cfg.data["observation"].setdefault("opponent_elixir", {}).update(
        {"assumed_recall": 1.0, "blind_spend_frac": 0.0})
    e = OpponentElixir(cfg, db)
    c = EnemyCycleTracker(db)
    c.record_play("hog_rider")                        # their wincon costs 4
    e._lo = e._hi = 0.5
    assert features(e, c, 5.0)[4] == pytest.approx(0.0)
    e._lo = e._hi = 9.0
    assert features(e, c, 5.0)[4] == pytest.approx(1.0)


def test_at_cap_feature_flags_the_punish_window(est, db):
    """Them sitting at 10 is the moment to commit -- it has to be legible as its own signal."""
    c = EnemyCycleTracker(db)
    assert features(est, c, 5.0)[6] == pytest.approx(0.0)
    est.regen(60.0, SINGLE_RATE)
    assert features(est, c, 5.0)[6] == pytest.approx(1.0)


# --- the deploy observer ------------------------------------------------------------------------
class _Det:
    def __init__(self, base, gy):
        self.base, self.gy = base, gy


def test_deploy_observer_reports_only_newly_appeared_enemy_cards():
    """The detector reports PRESENCE every frame; a deploy is presence that was not there before."""
    obs = DeployObserver()
    assert obs.step([_Det("hog_rider", 0.3)]) == ["hog_rider"]
    assert obs.step([_Det("hog_rider", 0.3)]) == [], "a surviving unit is not a new deploy"
    assert obs.step([_Det("hog_rider", 0.3), _Det("musketeer", 0.3)]) == ["musketeer"]


def test_deploy_observer_ignores_units_that_walk_into_view_on_your_half():
    """A troop crossing the river is not a fresh play -- only spawns on THEIR half count."""
    obs = DeployObserver()
    assert obs.step([_Det("hog_rider", 0.8)]) == []


# --- 2. SIM PARITY: the estimator must NOT see ground truth --------------------------------------
def _sim(cfg, **obs):
    cfg.data["observation"]["use_opponent_elixir"] = True
    cfg.data["observation"].update(obs)
    return SimMatchEnv(cfg, seed=5)


def test_the_sim_estimator_misses_deploys_it_is_not_fed_ground_truth(cfg):
    """THE CENTRAL REQUIREMENT. The engine knows exactly what the opponent played; the live policy will
    not. If the sim fed the estimator ground truth, the policy would learn to trust a precision it does
    not have and the failure would only appear on the ladder.

    At the measured recall the estimator must therefore MISS deploys -- its interval must be wide and
    its estimate must differ from the engine's true opponent elixir.
    """
    env = _sim(cfg, sim_detector_recall=0.55, sim_detector_precision=0.91)
    env.reset()
    gaps, widths = [], []
    for i in range(120):
        hand = [c for c in env._hand_ids() if env.specs[c].elixir <= env.elixir]
        env.step((1, hand[0], 0) if hand and i % 3 == 0 else (0, 0, 0))
        gaps.append(abs(env.opp_elixir.mid - env.eng.elixir[1]))
        widths.append(env.opp_elixir.hi - env.opp_elixir.lo)

    assert max(gaps) > 0.5, "the estimate never diverged from truth -- it is being fed ground truth"
    assert max(widths) > 0.0, "the interval never widened -- no detector noise is being applied"


def test_the_upper_bound_is_sound_under_recall_error(cfg):
    """`hi` is a genuine UPPER BOUND against MISSED detections: a spend we did not see can only make
    the truth lower. Verified over a full rollout with precision pinned at 1.0 to isolate recall --
    it is deliberately NOT claimed under precision error, where misreading a cheap card as an
    expensive one subtracts too much and can push `hi` below the truth."""
    env = _sim(cfg, sim_detector_recall=0.5, sim_detector_precision=1.0)
    env.reset()
    for i in range(150):
        hand = [c for c in env._hand_ids() if env.specs[c].elixir <= env.elixir]
        env.step((1, hand[0], 0) if hand and i % 3 == 0 else (0, 0, 0))
        assert env.opp_elixir.hi >= env.eng.elixir[1] - 1e-6, (
            f"upper bound {env.opp_elixir.hi:.3f} fell below true opponent elixir "
            f"{env.eng.elixir[1]:.3f} at step {i}")


def test_the_estimate_brackets_the_truth_most_of_the_time(cfg):
    """The interval is only useful if the truth is usually inside it. Not asserted at 100%: precision
    error can push a bound past the truth, and claiming otherwise would be overselling it."""
    env = _sim(cfg, sim_detector_recall=0.55, sim_detector_precision=1.0)
    env.reset()
    inside = total = 0
    for i in range(150):
        hand = [c for c in env._hand_ids() if env.specs[c].elixir <= env.elixir]
        env.step((1, hand[0], 0) if hand and i % 3 == 0 else (0, 0, 0))
        truth = env.eng.elixir[1]
        total += 1
        inside += env.opp_elixir.lo - 1e-6 <= truth <= env.opp_elixir.hi + 1e-6
    assert inside / total > 0.8, f"truth was inside the interval only {inside}/{total} of the time"


def test_the_private_rng_leaves_seeded_eval_streams_bit_identical(cfg, db):
    """The 5cdf867 DomainRand rule: the corruption must draw from a PRIVATE rng so the 777-seeded eval
    benchmark is unchanged. The seed is drawn UNCONDITIONALLY -- drawing it only when the feature is on
    would itself desync `env.rng` between the two configurations, which is the exact bug this guards."""
    def stream(flag, n=10):
        c = cfg.__class__(data=__import__("copy").deepcopy(cfg.data), root=cfg.root)
        c.data["observation"]["use_opponent_elixir"] = flag
        e = SimMatchEnv(c, seed=777)
        out = []
        for _ in range(n):
            e.reset()
            out.append((tuple(e.cycle), tuple(getattr(e.opponent, "cards", ()) or ()),
                        getattr(e.opponent, "style", None), e.eng.tower_setup[1]))
            for _ in range(20):
                _o, _r, done, _i = e.step((0, 0, 0))
                if done:
                    break
        return out

    assert stream(True) == stream(False)


def test_the_corruption_is_reproducible_for_a_given_env_seed(cfg):
    """Two envs at the same seed must corrupt identically, or training runs stop being reproducible."""
    a, b = _sim(cfg), _sim(cfg)
    a.reset(); b.reset()
    for i in range(60):
        for env in (a, b):
            hand = [c for c in env._hand_ids() if env.specs[c].elixir <= env.elixir]
            env.step((1, hand[0], 0) if hand and i % 3 == 0 else (0, 0, 0))
        assert a.opp_elixir.lo == pytest.approx(b.opp_elixir.lo)
        assert a.opp_elixir.hi == pytest.approx(b.opp_elixir.hi)


def test_the_block_is_appended_to_the_threat_vector_at_the_declared_width(cfg):
    """The policy only benefits if the features actually reach it, at the width the net was built for."""
    off = SimMatchEnv(cfg, seed=1)
    cfg2 = cfg.__class__(data=__import__("copy").deepcopy(cfg.data), root=cfg.root)
    cfg2.data["observation"]["use_opponent_elixir"] = False
    narrow = SimMatchEnv(cfg2, seed=1)
    assert off.threat_dim - narrow.threat_dim == OPP_ELIXIR_DIM
    off.reset()
    assert off.threat_vec.shape[0] == off.threat_dim
