"""ACTION SPACE: per-card named anchors (clashrl/actions.py), which replaced the 18x24 placement grid.

The grid could not express the tiles this deck is decided by -- an X-Bow's 4-tile vs 5-tile placement --
and 87db21c's adaptive split-push opponents punish exactly that imprecision. These tests pin the
properties the replacement has to hold: every deck card is playable, every anchor is a real position,
the head mask matches the anchors, and a checkpoint trained on different anchors is REFUSED rather than
silently placing cards on the wrong tiles.
"""
from __future__ import annotations

import math

import pytest

from clashrl.actions import AnchorSpace, signature_diff, signatures_match


@pytest.fixture()
def aspace(cfg, db) -> AnchorSpace:
    return AnchorSpace(cfg, db)


# --- coverage ---------------------------------------------------------------------------------
def test_every_deck_card_has_at_least_one_anchor(aspace, db):
    """A card with no anchors is a card the policy CANNOT PLAY -- the action space would silently
    remove it from the deck."""
    for card_id, key in enumerate(aspace.deck_keys):
        assert aspace.count(card_id) >= 1, f"{key} has no anchors"
    assert len(aspace.deck_keys) == len(db.deck_identities())


def test_the_deck_s_signature_cards_have_their_named_placements(aspace):
    """The placements this deck is built on must exist BY NAME -- they are the reason the grid was
    replaced, and the reward/verify tooling refers to them by name."""
    x_bow = aspace.card_index("x_bow")
    for name in ("4tile_left", "4tile_right", "5tile_left", "5tile_right", "defensive_centre"):
        assert aspace.index_of(x_bow, name) >= 0, f"x_bow is missing the {name} anchor"

    for card, names in (("tesla", ["centre_pull"]),
                        ("knight", ["bridge_left", "bridge_right", "defend_left", "defend_right"]),
                        ("ice_wizard", ["behind_left_tower", "behind_right_tower"]),
                        ("tornado", ["king_activate"]),
                        ("skeletons", ["split_left", "split_right"])):
        cid = aspace.card_index(card)
        assert cid >= 0, f"{card} is not in the deck"
        for n in names:
            assert aspace.index_of(cid, n) >= 0, f"{card} is missing the {n} anchor"


def test_evolved_identities_inherit_their_base_card_s_anchors(aspace):
    """An EVOLUTION changes a card's stats, not the tiles it wants to stand on -- so `knight_evo`
    plays the same placements as `knight` unless it defines its own."""
    for evo, base in (("knight_evo", "knight"), ("tesla_evo", "tesla")):
        e, b = aspace.card_index(evo), aspace.card_index(base)
        if e < 0 or b < 0:
            continue
        assert [a.name for a in aspace.anchors_for(e)] == [a.name for a in aspace.anchors_for(b)]
        assert [(a.x, a.y) for a in aspace.anchors_for(e)] == [(a.x, a.y) for a in aspace.anchors_for(b)]


def test_the_action_space_is_orders_of_magnitude_smaller_than_the_grid(aspace):
    """The point of the change: ~4300 grid actions (10 cards x 432 cells) collapse to a few dozen
    deliberate, named placements."""
    total = sum(aspace.count(c) for c in range(aspace.n_cards))
    assert total < 100, f"{total} placements -- the anchor set has grown back into a grid"
    assert total < aspace.n_cards * 432 / 10


# --- resolved geometry -------------------------------------------------------------------------
def test_every_anchor_resolves_inside_the_arena(aspace):
    """An anchor is a TAP POINT: off-board coordinates would tap the HUD or miss the window."""
    for card_id in range(aspace.n_cards):
        for a in aspace.anchors_for(card_id):
            assert 0.0 < a.x < 1.0, f"{a.name} x={a.x}"
            assert aspace.a_top <= a.y <= aspace.a_bot, f"{a.name} y={a.y} outside the arena band"


def test_anchors_within_a_card_are_distinct_positions(aspace):
    """Two anchors resolving to the same tile are two actions that do the same thing -- wasted head
    width and a split gradient."""
    for card_id, key in enumerate(aspace.deck_keys):
        pts = [(round(a.x, 4), round(a.y, 4)) for a in aspace.anchors_for(card_id)]
        assert len(set(pts)) == len(pts), f"{key} has duplicate anchor positions: {pts}"


def test_anchor_names_are_unique_within_a_card(aspace):
    """Names are the interface -- `verify --anchors`, the reward cross-check and the checkpoint
    signature all key off them."""
    for card_id, key in enumerate(aspace.deck_keys):
        names = [a.name for a in aspace.anchors_for(card_id)]
        assert len(set(names)) == len(names), f"{key} has duplicate anchor names"


def test_offsets_are_measured_in_true_board_tiles(aspace):
    """An offset of N tiles must move the anchor exactly N tile-widths from its landmark -- that is
    what makes '4-tile' and '5-tile' mean what they say."""
    x_bow = aspace.card_index("x_bow")
    four = aspace.anchor(x_bow, aspace.index_of(x_bow, "4tile_left"))
    five = aspace.anchor(x_bow, aspace.index_of(x_bow, "5tile_left"))
    assert five.y - four.y == pytest.approx(aspace.tile_h), "5-tile must sit exactly one tile behind 4-tile"
    assert five.x == pytest.approx(four.x), "both are the same lane"


def test_anchors_resolve_relative_to_calibrated_landmarks(aspace, cfg):
    """Landmarks come STRAIGHT from the coordinates config already calibrates (env.my_towers /
    env.enemy_towers), so an anchor described as 'at the enemy left tower' is at the tower every other
    component believes in -- no second, drifting copy of the board geometry."""
    enemy = cfg.get("env", "enemy_towers")
    rocket = aspace.card_index("rocket")
    a = aspace.anchor(rocket, aspace.index_of(rocket, "enemy_left_tower"))
    assert (a.x, a.y) == pytest.approx((enemy[0][0], enemy[0][1]))

    mine = cfg.get("env", "my_towers")
    assert aspace.landmarks["my_king"] == pytest.approx(tuple(mine[2]))


def test_your_own_cards_are_anchored_on_your_own_half(aspace):
    """CR rule: everything except a spell deploys on YOUR half. The old grid needed a deployable mask
    and a clamp to enforce that; anchors must satisfy it BY CONSTRUCTION."""
    river = 0.5
    for card_id, key in enumerate(aspace.deck_keys):
        base = key[:-4] if key.endswith("_evo") else key
        if base in ("rocket", "tornado", "the_log"):
            continue                       # spells may target anywhere
        for a in aspace.anchors_for(card_id):
            assert a.y > river, f"{key}.{a.name} at y={a.y:.3f} is on the ENEMY half"


def test_rocket_can_target_both_halves(aspace):
    """CR rule: a Rocket goes ANYWHERE -- its target set must span enemy towers AND your own defensive
    tiles (rocketing a push that already crossed)."""
    rocket = aspace.card_index("rocket")
    ys = [a.y for a in aspace.anchors_for(rocket)]
    assert min(ys) < 0.5, "no enemy-half rocket target"
    assert max(ys) > 0.5, "no own-half rocket target"


# --- the head mask ------------------------------------------------------------------------------
def test_the_placement_head_is_as_wide_as_the_widest_card(aspace):
    """The head is a single Linear layer shared by every card, so it must be wide enough for the card
    with the most anchors -- and no wider."""
    assert aspace.n_anchors == max(aspace.count(c) for c in range(aspace.n_cards))


def test_the_mask_exposes_exactly_that_card_s_anchors(aspace):
    """Masking is what makes one shared head safe: a card must never be able to select a slot that is
    not one of ITS anchors (that slot means a different tile, or nothing at all)."""
    for card_id in range(aspace.n_cards):
        mask = aspace.mask(card_id)
        assert len(mask) == aspace.n_anchors
        assert sum(mask) == aspace.count(card_id)
        assert all(mask[:aspace.count(card_id)]), "a card's own anchors must all be selectable"
        assert not any(mask[aspace.count(card_id):]), "unused slots must be masked off"


def test_mask_table_matches_the_per_card_masks(aspace):
    """The trainers build one [n_cards, n_anchors] tensor from this -- it must agree with mask()."""
    table = aspace.mask_table()
    assert len(table) == aspace.n_cards
    for card_id in range(aspace.n_cards):
        assert table[card_id] == aspace.mask(card_id)


def test_every_unmasked_slot_resolves_to_that_anchor_s_point(aspace):
    """The mask and the coordinate lookup must agree, or the policy selects one tile and taps another."""
    for card_id in range(aspace.n_cards):
        for idx, ok in enumerate(aspace.mask(card_id)):
            if not ok:
                continue
            a = aspace.anchor(card_id, idx)
            assert a is not None
            assert aspace.point(card_id, idx) == pytest.approx((a.x, a.y))


def test_an_out_of_range_index_falls_back_rather_than_crashing(aspace):
    """The head is masked so this is unreachable -- but a live match is the wrong place to raise, so a
    bad index resolves to the card's first anchor instead."""
    pt = aspace.point(0, 999)
    first = aspace.anchors_for(0)[0]
    assert pt == pytest.approx((first.x, first.y))


# --- quantizing a human play (behaviour cloning) -------------------------------------------------
def test_nearest_snaps_a_recorded_click_onto_the_card_s_own_anchors(aspace):
    """Behaviour cloning can only imitate actions the policy can TAKE, so a recorded human click has
    to be quantized onto the played card's anchor list."""
    x_bow = aspace.card_index("x_bow")
    for target in aspace.anchors_for(x_bow):
        idx = aspace.nearest(x_bow, target.x + 0.002, target.y + 0.002)
        assert aspace.name(x_bow, idx) == target.name

    nudged = aspace.nearest(x_bow, 0.9, 0.9)          # nowhere near any anchor
    assert 0 <= nudged < aspace.count(x_bow), "an off-anchor click still snaps to a valid anchor"


def test_nearest_only_ever_returns_an_index_that_card_can_play(aspace):
    """A quantized label outside the card's own anchors would train the head to select a masked slot."""
    for card_id in range(aspace.n_cards):
        for x in (0.05, 0.5, 0.95):
            for y in (0.15, 0.5, 0.85):
                idx = aspace.nearest(card_id, x, y)
                assert aspace.mask(card_id)[idx], "nearest() returned a masked slot"


# --- checkpoint migration guard (the 377b5b7 fail-fast pattern) -----------------------------------
def test_signature_is_stable_across_rebuilds(cfg, db):
    """The signature is stored in every checkpoint, so identical config must produce an identical
    signature -- otherwise every reload would be a false mismatch."""
    a, b = AnchorSpace(cfg, db), AnchorSpace(cfg, db)
    assert a.signature() == b.signature()
    assert signatures_match(a.signature(), b.signature())


def test_signature_changes_when_an_anchor_is_added_or_renamed(cfg, db):
    """A policy's placement head is indexed by the anchor LIST, so any change to that list must
    invalidate the checkpoint -- index 2 would otherwise silently become a different tile."""
    base = AnchorSpace(cfg, db).signature()

    renamed = dict(db.anchors)
    renamed["x_bow"] = [dict(e) for e in renamed["x_bow"]]
    renamed["x_bow"][0]["name"] = "4tile_left_v2"
    db2 = _db_with_anchors(db, renamed)
    assert not signatures_match(AnchorSpace(cfg, db2).signature(), base)

    added = dict(db.anchors)
    added["tesla"] = list(added["tesla"]) + [{"name": "extra", "from": "my_king", "offset": [0, -4]}]
    db3 = _db_with_anchors(db, added)
    assert not signatures_match(AnchorSpace(cfg, db3).signature(), base)


def test_signature_ignores_a_pure_coordinate_nudge(cfg, db):
    """Deliberate: the signature keys off the anchor LIST (names/order), which is what the head indexes.
    Re-calibrating WHERE an anchor sits -- the whole point of `verify --anchors` -- keeps a checkpoint
    usable, because slot 2 still means '4tile_left'. Changing WHICH anchors exist does not."""
    base = AnchorSpace(cfg, db).signature()
    moved = dict(db.anchors)
    moved["x_bow"] = [dict(e) for e in moved["x_bow"]]
    moved["x_bow"][0]["offset"] = [0, 6]
    db2 = _db_with_anchors(db, moved)
    assert signatures_match(AnchorSpace(cfg, db2).signature(), base)


def test_a_grid_era_checkpoint_is_rejected_with_an_explanation(cfg, db):
    """The migration case: a checkpoint from before named anchors has no signature at all, and must be
    refused with a message that says WHY rather than loading and placing cards on wrong tiles."""
    current = AnchorSpace(cfg, db).signature()
    assert not signatures_match(None, current)
    reasons = signature_diff(None, current)
    assert reasons and "GRID" in reasons[0].upper()


def test_signature_diff_names_the_cards_that_changed(cfg, db):
    """The guard prints this, so it has to identify WHICH card moved -- 'signatures differ' is useless
    when you are trying to work out whether to retrain or revert."""
    base = AnchorSpace(cfg, db).signature()
    changed = dict(db.anchors)
    changed["tornado"] = [dict(e) for e in changed["tornado"]][:2]
    db2 = _db_with_anchors(db, changed)
    reasons = signature_diff(base, AnchorSpace(cfg, db2).signature())
    assert any("tornado" in r for r in reasons), reasons


def _db_with_anchors(db, anchors):
    """A shallow stand-in for CardDB carrying a different anchor block."""
    class _DB:
        def __init__(self, inner, anc):
            self._inner, self.anchors = inner, anc

        def __getattr__(self, item):
            return getattr(self._inner, item)

    return _DB(db, anchors)


# --- misconfiguration is loud --------------------------------------------------------------------
def test_a_card_with_no_anchor_block_is_a_hard_error(cfg, db):
    """Silently dropping a card from the action space would be a policy that can never play it."""
    missing = {k: v for k, v in db.anchors.items() if k != "x_bow"}
    with pytest.raises(ValueError, match="no anchors defined"):
        AnchorSpace(cfg, _db_with_anchors(db, missing))


def test_an_unknown_landmark_is_a_hard_error(cfg, db):
    """A typo'd landmark would otherwise resolve to some default position -- an anchor that is silently
    on the wrong tile is exactly the failure this whole change exists to remove."""
    bad = dict(db.anchors)
    bad["tesla"] = [{"name": "oops", "from": "not_a_landmark", "offset": [0, 0]}]
    with pytest.raises(ValueError, match="unknown landmark"):
        AnchorSpace(cfg, _db_with_anchors(db, bad))


def test_duplicate_anchor_names_are_a_hard_error(cfg, db):
    """Two anchors with one name make `index_of` ambiguous and the signature misleading."""
    bad = dict(db.anchors)
    bad["tesla"] = [{"name": "dup", "from": "my_king", "offset": [0, -1]},
                    {"name": "dup", "from": "my_king", "offset": [0, -2]}]
    with pytest.raises(ValueError, match="duplicate anchor name"):
        AnchorSpace(cfg, _db_with_anchors(db, bad))


# --- integration with the reward geometry --------------------------------------------------------
def test_xbow_named_placements_satisfy_the_offensive_reward(aspace, cfg):
    """The anchors and the reward that scores them are configured separately, so they can disagree --
    and did: at env.xbow_range 0.36 the canonical 4-tile and 5-tile anchors were scored as MISPLACES,
    training the policy against the very placements the anchors exist to express.

    `verify --anchors` cross-checks this; this test keeps it from regressing.
    """
    _, enemy, _ = __import__("clashrl.reward", fromlist=["_anchors"])._anchors(cfg)
    xbow_range = float(cfg.get("env", "xbow_range"))
    x_bow = aspace.card_index("x_bow")
    for a in aspace.anchors_for(x_bow):
        if not a.name.startswith(("4tile", "5tile")):
            continue
        d = min(math.hypot(a.x - ax, a.y - ay) for ax, ay in enemy[:2])
        assert d <= xbow_range, (
            f"x_bow.{a.name} is {d:.3f} from the nearest enemy princess but env.xbow_range is "
            f"{xbow_range:.3f} -> the offensive reward would score it a misplace")


def test_xbow_defensive_anchor_sits_in_the_rewarded_band(aspace, cfg):
    """The defensive-phase reward pays for a back-centre X-Bow inside a specific band; the anchor named
    for that phase has to be in it."""
    front = float(cfg.get("env", "xbow_defense_front"))
    back = float(cfg.get("env", "xbow_defense_back"))
    x_bow = aspace.card_index("x_bow")
    a = aspace.anchor(x_bow, aspace.index_of(x_bow, "defensive_centre"))
    assert abs(a.x - 0.48) <= 0.18, "must be central"
    assert front <= a.y <= back, f"y={a.y:.3f} outside the defensive band [{front}, {back}]"
