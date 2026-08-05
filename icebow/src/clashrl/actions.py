"""ACTION SPACE: per-card NAMED ANCHORS at exact tile positions.

WHY THIS REPLACED THE GRID. The action space used to be an 18x24 placement grid -- 432 cells x 10 card
identities ~= 4300 actions -- and it could not express the placements this deck is actually built on.
An X-Bow's value is decided by WHICH TILE it sits on: the 4-tile and 5-tile placements reach the enemy
tower, one row further back does not, and a coarsened 24-row grid does not even have a cell boundary at
the right place. The policy was being asked to find a handful of correct tiles by exploring thousands of
near-identical wrong ones, and the grid could not represent some of the right answers at all.

87db21c made that worse on purpose: the adaptive split-push opponents punish imprecise timing and
placement specifically. A grid that cannot express the exact answer cannot learn to beat them.

WHAT REPLACED IT. Each card gets a small set of NAMED anchors -- `x_bow.4tile_left`,
`tornado.king_activate`, `knight.bridge_right` -- defined in `config/cards.yaml` under `anchors:`. The
action is `(card identity, anchor index)`; there are no illegal placements to mask out and no clamping,
because every anchor is a legal, deliberate, human-named position. The whole space is ~50 actions
instead of ~4300, and every one of them is a play a person could describe out loud.

HOW AN ANCHOR IS DEFINED -- and why it is not a raw tile index. Each anchor is a TILE OFFSET FROM A
NAMED LANDMARK:

    {name: 4tile_left, from: bridge_left, offset: [0, 4]}   # 4 tiles behind the left bridge

Landmarks (tower centres, bridges, the river) come straight from the coordinates `config.yaml` already
calibrates -- `env.my_towers`, `env.enemy_towers` -- so a landmark is exactly where the rest of the
pipeline believes it is, with no re-derivation. Only the OFFSET is scaled by the tile lattice. That
matters: `action.arena_box` is documented as a close approximation of the tile grid rather than a
pixel-exact homography, so hanging anchors off landmarks confines that approximation to a few tiles of
offset instead of letting it displace an absolute tile index across the whole board.

Offsets are in TRUE BOARD TILES on Clash Royale's 18x32 lattice: `+dcol` is right, `+drow` is toward
YOUR baseline (down-screen). The policy always acts from team 0's point of view -- the sim mirrors the
board for team 1 -- so "toward your baseline" is unambiguous.

VERIFY BEFORE YOU TRAIN. Tile-exactness is the entire point, so the coordinates below are a starting
set derived from the calibrated landmarks, NOT gospel. `run.py verify --anchors` overlays every anchor
on real in-match frames; check they land on the intended tiles and adjust `config/cards.yaml` (or
`action.arena_box`, which sets the tile size) until they do. An anchor that lands one tile off is worse
than a grid cell, because the policy will trust it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, NamedTuple, Optional, Tuple

# Clash Royale's board. Anchor offsets are in these tiles, NOT in `action.grid` (which is gone).
TILES_W, TILES_H = 18, 32
_RIVER_Y = 0.5           # the river, in the normalized frame coords the whole pipeline shares
_BRIDGES_X = (0.25, 0.75)


class Anchor(NamedTuple):
    """One named placement for one card."""
    name: str            # e.g. "4tile_left" -- unique within the card
    landmark: str        # the reference point the offset hangs off
    dcol: float          # tiles right of the landmark
    drow: float          # tiles toward YOUR baseline from the landmark
    x: float             # resolved normalized frame coordinate
    y: float

    @property
    def label(self) -> str:
        return f"{self.landmark}{self.dcol:+g},{self.drow:+g}"


class AnchorSpace:
    """The action space: per-card named anchors resolved to normalized frame coordinates.

    `n_anchors` is the WIDEST card's anchor count -- the policy's placement head is that wide and is
    masked per card, exactly as the card head is masked to the hand. Cards with fewer anchors simply
    have the tail of the head masked off.
    """

    def __init__(self, cfg, db=None):
        from .cards import CardDB

        self.cfg = cfg
        self.db = db if db is not None else CardDB(cfg)
        self.deck_keys: List[str] = list(self.db.deck_identities())
        self.n_cards = max(1, len(self.deck_keys))

        self.slots = cfg.get("hand", "slots", default=[])
        self.n_slots = len(self.slots)
        self.a_top = float(cfg.get("label", "arena_top", default=0.10))
        self.a_bot = float(cfg.get("label", "arena_bottom", default=0.86))
        self.chat_box = cfg.get("buttons", "chat_avoid_box", default=None)

        bx = cfg.get("action", "arena_box", default=None) or \
            cfg.get("env", "arena_region", default=[0.03, 0.10, 0.97, 0.86])
        self.bx0, self.by0, self.bx1, self.by1 = (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))
        # ONE TILE, in normalized frame units. Only anchor OFFSETS are scaled by this; landmarks come
        # from their own calibrated coordinates, so a slightly-off arena_box shifts an anchor by a
        # fraction of a tile rather than relocating it across the board.
        self.tile_w = (self.bx1 - self.bx0) / TILES_W
        self.tile_h = (self.by1 - self.by0) / TILES_H

        self.landmarks: Dict[str, Tuple[float, float]] = self._build_landmarks(cfg)
        self._anchors: List[List[Anchor]] = [self._resolve_card(k) for k in self.deck_keys]
        self.n_anchors = max((len(a) for a in self._anchors), default=1)
        self._index: List[Dict[str, int]] = [
            {a.name: i for i, a in enumerate(card)} for card in self._anchors]

    # -- landmarks -------------------------------------------------------------------------------
    def _build_landmarks(self, cfg) -> Dict[str, Tuple[float, float]]:
        """Reference points anchors hang off, in normalized frame coordinates.

        Tower positions are taken verbatim from `env.my_towers` / `env.enemy_towers` -- the same
        constants the sim engine seeds its towers from and the live reward reads -- so an anchor
        described as 'behind the left tower' is behind the tower every other component believes in.
        """
        mine = cfg.get("env", "my_towers", default=[[0.245, 0.615], [0.745, 0.615], [0.48, 0.72]])
        enemy = cfg.get("env", "enemy_towers", default=[[0.25, 0.205], [0.745, 0.205], [0.48, 0.11]])
        lm = {
            "my_left_tower": tuple(mine[0]),
            "my_right_tower": tuple(mine[1]),
            "my_king": tuple(mine[2]),
            "enemy_left_tower": tuple(enemy[0]),
            "enemy_right_tower": tuple(enemy[1]),
            "enemy_king": tuple(enemy[2]),
            "bridge_left": (_BRIDGES_X[0], _RIVER_Y),
            "bridge_right": (_BRIDGES_X[1], _RIVER_Y),
            "river_centre": (0.5, _RIVER_Y),
        }
        # the centre line midway between the river and your king -- the classic siege/defence band
        lm["my_centre"] = (0.5, (_RIVER_Y + float(mine[2][1])) / 2.0)
        extra = cfg.get("anchors", "landmarks", default=None) or {}
        for name, xy in extra.items():                 # config may add or override landmarks
            lm[str(name)] = (float(xy[0]), float(xy[1]))
        return lm

    # -- anchor resolution -----------------------------------------------------------------------
    def _anchor_config(self) -> dict:
        """`anchors.cards` from the card KB (config/cards.yaml)."""
        raw = getattr(self.db, "anchors", None)
        return dict(raw or {})

    def _resolve_card(self, key: str) -> List[Anchor]:
        """Anchors for one deck identity.

        An EVOLVED identity (`knight_evo`) falls back to its base card's anchors unless it defines its
        own: an evolution changes a card's stats, not the tiles it wants to stand on.
        """
        cards = self._anchor_config()
        base = key[:-4] if key.endswith("_evo") else key
        spec = cards.get(key) or cards.get(base)
        if not spec:
            raise ValueError(
                f"no anchors defined for deck card '{key}' -- add an `anchors.cards.{base}` block to "
                f"config/cards.yaml. Every deck card needs at least one anchor; the action space is "
                f"built from them.")
        out: List[Anchor] = []
        seen = set()
        for entry in spec:
            name = str(entry["name"])
            if name in seen:
                raise ValueError(f"duplicate anchor name '{name}' for card '{key}'")
            seen.add(name)
            landmark = str(entry["from"])
            if landmark not in self.landmarks:
                raise ValueError(
                    f"anchor '{key}.{name}' references unknown landmark '{landmark}'. "
                    f"Known: {', '.join(sorted(self.landmarks))}")
            dcol, drow = (float(entry.get("offset", [0, 0])[0]), float(entry.get("offset", [0, 0])[1]))
            lx, ly = self.landmarks[landmark]
            x, y = self._clamp(lx + dcol * self.tile_w, ly + drow * self.tile_h)
            out.append(Anchor(name, landmark, dcol, drow, x, y))
        return out

    def _clamp(self, nx: float, ny: float) -> Tuple[float, float]:
        """Keep a resolved anchor on the board and off the card tray / chat icon."""
        nx = min(max(nx, 0.02), 0.98)
        ny = min(max(ny, self.a_top), self.a_bot)
        if self.chat_box:
            x0, y0, x1, y1 = self.chat_box
            if x0 <= nx <= x1 and y0 <= ny <= y1:
                ny = max(self.a_top, y0 - 0.01)
        return nx, ny

    # -- lookups ---------------------------------------------------------------------------------
    def anchors_for(self, card_id: int) -> List[Anchor]:
        if not (0 <= card_id < self.n_cards):
            return []
        return self._anchors[card_id]

    def count(self, card_id: int) -> int:
        return len(self.anchors_for(card_id))

    def anchor(self, card_id: int, idx: int) -> Optional[Anchor]:
        card = self.anchors_for(card_id)
        return card[idx] if 0 <= idx < len(card) else None

    def point(self, card_id: int, idx: int) -> Tuple[float, float]:
        """Normalized (nx, ny) for one action. Falls back to the card's FIRST anchor on a bad index --
        a policy head is masked so this should be unreachable, and silently placing at a sane spot beats
        raising inside a live match."""
        a = self.anchor(card_id, idx)
        if a is None:
            card = self.anchors_for(card_id)
            if not card:
                return 0.5, 0.6
            a = card[0]
        return a.x, a.y

    def name(self, card_id: int, idx: int) -> str:
        a = self.anchor(card_id, idx)
        return a.name if a is not None else "?"

    def index_of(self, card_id: int, name: str) -> int:
        """Anchor index by name, or -1. Lets rewards/tests refer to `4tile_left` rather than an int."""
        if not (0 <= card_id < self.n_cards):
            return -1
        return self._index[card_id].get(name, -1)

    def card_index(self, key: str) -> int:
        return self.deck_keys.index(key) if key in self.deck_keys else -1

    def mask(self, card_id: int) -> List[bool]:
        """Per-card validity over the placement head (length `n_anchors`).

        Replaces the grid's `deployable_mask`: with anchors there is no such thing as an undeployable
        position, only a head slot this card does not use.
        """
        n = self.count(card_id)
        return [i < n for i in range(self.n_anchors)]

    def mask_table(self) -> List[List[bool]]:
        """`[n_cards][n_anchors]` validity -- the trainers build one tensor from this."""
        return [self.mask(c) for c in range(self.n_cards)]

    def nearest(self, card_id: int, nx: float, ny: float) -> int:
        """The anchor closest to a raw normalized point -- how a RECORDED human play (an arbitrary
        pixel) is quantized onto the action space for behaviour cloning."""
        card = self.anchors_for(card_id)
        if not card:
            return 0
        return min(range(len(card)), key=lambda i: (card[i].x - nx) ** 2 + (card[i].y - ny) ** 2)

    def decode(self, slot: int, card_id: int, idx: int) -> Tuple[float, float, float, float]:
        """(slot_nx, slot_ny, target_nx, target_ny) normalized taps for the controller."""
        snx, sny = self.slots[slot]
        tnx, tny = self.point(card_id, idx)
        return snx, sny, tnx, tny

    # -- checkpoint identity ---------------------------------------------------------------------
    def signature(self) -> dict:
        """A compact, comparable description of the action space, stored in every checkpoint.

        A policy's placement head is meaningless under a different anchor set -- index 2 silently
        becomes a different tile -- so `train-rl` and `play` compare this and refuse a mismatch rather
        than play nonsense. Mirrors the deck guard added in 377b5b7.
        """
        detail = {k: [a.name for a in card] for k, card in zip(self.deck_keys, self._anchors)}
        blob = json.dumps(detail, sort_keys=True).encode("utf-8")
        return {"kind": "anchors", "n_cards": self.n_cards, "n_anchors": self.n_anchors,
                "cards": detail, "hash": hashlib.sha256(blob).hexdigest()[:16]}

    def describe(self) -> str:
        lines = [f"action space: ANCHORS -- {self.n_cards} cards x up to {self.n_anchors} anchors "
                 f"= {sum(len(a) for a in self._anchors)} placements"]
        for key, card in zip(self.deck_keys, self._anchors):
            names = ", ".join(f"{a.name}({a.x:.3f},{a.y:.3f})" for a in card)
            lines.append(f"  {key:12} {len(card)}: {names}")
        return "\n".join(lines)


def signatures_match(a: Optional[dict], b: Optional[dict]) -> bool:
    """True when two action-space signatures describe the same head layout AND the same anchors."""
    if not a or not b:
        return False
    return (a.get("kind") == b.get("kind") and a.get("hash") == b.get("hash")
            and a.get("n_cards") == b.get("n_cards") and a.get("n_anchors") == b.get("n_anchors"))


def signature_diff(ckpt: Optional[dict], current: dict) -> List[str]:
    """Human-readable reasons two signatures differ -- printed by the fail-fast guards."""
    if not ckpt:
        return ["the checkpoint predates named anchors (it was trained on the 18x24 placement GRID)"]
    out: List[str] = []
    if ckpt.get("kind") != current.get("kind"):
        out.append(f"action-space kind {ckpt.get('kind')!r} -> {current.get('kind')!r}")
    if ckpt.get("n_cards") != current.get("n_cards"):
        out.append(f"card count {ckpt.get('n_cards')} -> {current.get('n_cards')}")
    if ckpt.get("n_anchors") != current.get("n_anchors"):
        out.append(f"placement-head width {ckpt.get('n_anchors')} -> {current.get('n_anchors')}")
    ck_cards, cur_cards = ckpt.get("cards") or {}, current.get("cards") or {}
    for key in sorted(set(ck_cards) | set(cur_cards)):
        old, new = ck_cards.get(key), cur_cards.get(key)
        if old is None:
            out.append(f"card {key}: not in the checkpoint (added)")
        elif new is None:
            out.append(f"card {key}: gone from the config (removed)")
        elif old != new:
            out.append(f"card {key}: {old} -> {new}")
    return out or ["signatures differ"]
