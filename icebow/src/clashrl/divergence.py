"""SIM-VS-REALITY DIVERGENCE: how far does sim/engine.py drift from a real match?

Everything the policy learns, it learns inside `sim/engine.py`. If the engine's model of Clash Royale
is wrong, the policy optimises a game that does not exist and the error is invisible -- the win-rate
curve rises happily against a simulator that is wrong in the same way it always was. This harness makes
the error measurable: take a REAL recorded match, replay the SAME action sequence through the engine
from a matched initial state, step both in lockstep, and score how far apart they end up.

WHAT IS ACTUALLY RECOVERABLE FROM A RECORDING -- the constraint that shapes everything below:

    tower HP .......... YES, per tower per frame (TowerHpTracker's digit CNN)
    tower destroyed ... YES (TowerTracker)
    YOUR plays ........ YES, exactly: events.jsonl clicks paired with hand recognition
    OPPONENT's plays .. NO. Nothing records them. They must be INFERRED from the detector as
                        "a new enemy unit appeared here, so a card was played here" -- which is why
                        a trained board detector is REQUIRED for a real session, and why opponent-play
                        inference quality is an upper bound on the whole measurement.
    unit counts ....... only via the detector, and only as counts (no identity correspondence)
    outcome ........... YES (results scoreboard, cross-checked against towers felled)

So the replay is: your real plays at their real times and positions, plus the opponent's INFERRED
plays, driven into the engine from a matched start. Anything the engine gets wrong -- pathing, aggro,
body-blocking -- compounds from there, which is exactly the signal we want.

THE DIVERGENCE SCORE combines four components, each normalized to roughly [0, 1] so they are
commensurable, then weighted (`divergence.weights`):

RESOLUTION CAVEAT: `--stride` sets the lockstep sampling period (default 1s), and anything shorter is
invisible to the score -- a 0.4s spell-delay error simply falls between samples. The score measures
drift AT THE SAMPLING RESOLUTION; it is not a claim about sub-second fidelity. (The "was this mechanic
exercised at all?" check deliberately runs at physics resolution instead, so a sub-stride mechanic is
not mislabelled as unused.)

    tower_hp    mean |HP-fraction error| across all six towers over time -- the backbone signal,
                since tower HP integrates almost every mechanic (who reached what, and when)
    units       mean per-side unit-count error, normalized -- catches "the sim kills things too fast"
    first_tower |difference in time-to-first-tower|, normalized by match length
    outcome     0 if the same result, 1 if not -- coarse but the one that actually matters

ATTRIBUTION -- and what it does and does not prove. To rank mechanics, each is ABLATED in the engine
and the replay re-scored; the ranking is by how much the divergence MOVES. This is a SENSITIVITY
ranking, not a fault-finding: a large delta means the trajectory depends heavily on that mechanic, so
its fidelity is where accuracy is worth buying. A NEGATIVE delta is the sharper result -- turning the
mechanic OFF made the sim match reality BETTER, which means the current model of it is actively worse
than not modelling it at all.

`charge` cannot be ablated because it is NOT MODELLED -- engine.py says so in its own docstring
("card-specific quirks (charge / ramp-up) are still out of scope"). An unmodelled mechanic has no knob
to turn, so it is reported separately as a structural gap, listing the cards in the match that have it.

Usage, from icebow/:
    python run.py divergence                       # the latest session
    python run.py divergence --session <path> --all-mechanics
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Cards whose real behaviour includes a CHARGE (damage/speed ramp on an uninterrupted run). The engine
# models none of it; used to report the gap rather than to ablate anything.
CHARGE_CARDS = {"prince", "dark_prince", "battle_ram", "ram_rider", "sparky", "electro_giant",
                "hog_rider", "royal_hogs", "elite_barbarians", "bandit", "goblin_giant"}


@dataclass
class Sample:
    """One lockstep observation of a match, real or simulated."""
    t: float
    my_hp: List[float]          # HP FRACTION per tower [left, right, king]
    enemy_hp: List[float]
    my_units: Optional[int] = None
    enemy_units: Optional[int] = None


@dataclass
class Play:
    t: float
    team: int                   # 0 = you, 1 = opponent
    card: str
    x: float
    y: float


@dataclass
class Trajectory:
    samples: List[Sample] = field(default_factory=list)
    plays: List[Play] = field(default_factory=list)
    outcome: Optional[str] = None
    first_tower_t: Optional[float] = None      # when the FIRST tower on either side fell
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def at(self, t: float) -> Optional[Sample]:
        """The sample nearest `t` (trajectories are sampled on their own grids)."""
        if not self.samples:
            return None
        return min(self.samples, key=lambda s: abs(s.t - t))


# --- OBSERVE: pull the real trajectory out of a recording -------------------------------------
def observe_session(cfg, session: Path, stride_s: float = 1.0, conf: Optional[float] = None,
                    verbose: bool = False) -> Optional[Trajectory]:
    """Extract the real match trajectory + both sides' play scripts from a recorded session."""
    import cv2

    from .cards import CardDB
    from .label import _extract_plays
    from .outcome import read_scoreboard
    from .reward import TowerTracker
    from .states import GameState
    from .tower_hp import TowerHpTracker
    from .vision import Vision

    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    region, fps = meta["region"], float(meta.get("fps", 12))
    video = next((session / n for n in ("video.mp4", "video.avi") if (session / n).exists()), None)
    if video is None:
        print(f"[divergence] no video in {session}")
        return None

    vision = Vision(cfg)
    db = CardDB(cfg)
    tower = TowerTracker(cfg)
    hp = TowerHpTracker(cfg)
    traj = Trajectory()

    # --- the opponent's plays need the detector; without it there is nothing to replay them from ---
    detector = None
    team_tracker = None
    try:
        from .replay_mine import TeamTracker, load_detector
        det = load_detector(cfg)
        detector = det if det.available else None
        team_tracker = TeamTracker()
    except Exception:                                     # noqa: BLE001
        detector = None
    det_conf = float(conf if conf is not None else cfg.get("observation", "detector_conf", default=0.5))
    if detector is None:
        traj.notes.append(
            "NO DETECTOR: the opponent's plays cannot be recovered (nothing records them) and unit "
            "counts are unavailable. The replay will contain YOUR plays only, so the divergence is "
            "dominated by 'the opponent never played' and is NOT a measurement of engine fidelity. "
            "Train the detector (tools/detect/train.py) before trusting any number from this run.")

    # --- YOUR plays: exact, from the click log paired with hand recognition ---
    events = [json.loads(line) for line in
              (session / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    clicks = [e for e in events if e.get("type") == "click" and e.get("pressed")]
    slots = cfg.get("hand", "slots", default=[])
    my_plays = _extract_plays(clicks, region,
                              slots,
                              float(cfg.get("label", "click_radius", default=0.06)),
                              float(cfg.get("label", "pair_timeout", default=2.0)),
                              float(cfg.get("label", "arena_top", default=0.10)),
                              float(cfg.get("label", "arena_bottom", default=0.86)))

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(stride_s * fps)))
    t0 = None                                   # video time of the first IN_MATCH frame
    prev_enemy: Dict[str, Tuple[float, float]] = {}
    first_tower_t = None
    my_alive_prev = [True, True, True]
    en_alive_prev = [True, True, True]
    seen_scoreboard = False

    for fi in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        vt = fi / fps
        state = vision.detect_state(frame)
        if state != GameState.IN_MATCH:
            if t0 is not None and not seen_scoreboard:
                sb = read_scoreboard(frame, cfg)
                if sb.present:
                    seen_scoreboard = True
                    thr = float(cfg.get("outcome", "gold_frac", default=0.10))
                    blue = sum(f >= thr for f in sb.blue_fracs)
                    red = sum(f >= thr for f in sb.red_fracs)
                    traj.outcome = "win" if blue > red else "loss" if red > blue else "draw"
            continue
        if t0 is None:
            t0 = vt
            tower.reset()
            hp.reset()
        t = vt - t0
        tower.step(frame)
        hp.step(frame)

        s = Sample(t=t,
                   my_hp=[min(1.0, max(0.0, v / hp.my_full)) for v in list(hp.my_hp)[:3]],
                   enemy_hp=[min(1.0, max(0.0, v / hp.full)) for v in list(hp.enemy_hp)[:3]])
        while len(s.my_hp) < 3:
            s.my_hp.append(1.0)
        while len(s.enemy_hp) < 3:
            s.enemy_hp.append(1.0)

        if detector is not None:
            dets = detector.detect(frame, conf=det_conf)
            team_tracker.tag(dets, t)
            mine = [d for d in dets if d.team == "mine"]
            enemy = [d for d in dets if d.team == "enemy"]
            s.my_units, s.enemy_units = len(mine), len(enemy)
            # OPPONENT PLAYS, inferred: an enemy class that was NOT on the board last sample and is on
            # THEIR half now == a card just played. Coarse -- it misses a card replayed while a
            # previous copy is still alive, and mistimes anything that walks into view -- but nothing
            # in the recording records the opponent's taps, so this is the only source there is.
            now_keys = {}
            for d in enemy:
                now_keys[d.base] = (d.cx, d.gy)
                if d.base not in prev_enemy and d.gy < 0.5:
                    traj.plays.append(Play(t=t, team=1, card=d.base, x=float(d.cx), y=float(d.gy)))
            prev_enemy = now_keys

        # first tower to fall on EITHER side
        for i in range(3):
            if my_alive_prev[i] and not tower.mine_alive[i] and first_tower_t is None:
                first_tower_t = t
            if en_alive_prev[i] and not tower.enemy_alive[i] and first_tower_t is None:
                first_tower_t = t
        my_alive_prev = list(tower.mine_alive[:3])
        en_alive_prev = list(tower.enemy_alive[:3])
        traj.samples.append(s)
        if verbose and len(traj.samples) % 30 == 0:
            print(f"[divergence]   observed {t:6.1f}s  my={[f'{v:.2f}' for v in s.my_hp]} "
                  f"enemy={[f'{v:.2f}' for v in s.enemy_hp]}", flush=True)
    cap.release()

    if t0 is None:
        print("[divergence] no IN_MATCH frames found in this session")
        return None

    # YOUR plays -> the script, with the card identity read off the tray at play time
    deck = list(db.deck_identities())
    for p in my_plays:
        pt = p["t"] - t0 if p["t"] > t0 else p["t"]
        if pt < 0:
            continue
        card = None
        slot = p.get("slot")
        if slot is not None:
            cap2 = cv2.VideoCapture(str(video))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int((pt + t0) * fps))
            ok2, f2 = cap2.read()
            cap2.release()
            if ok2 and f2 is not None:
                ids = vision.recognize_hand(f2)
                if 0 <= slot < len(ids) and 0 <= ids[slot] < len(deck):
                    card = deck[ids[slot]]
        if card:
            traj.plays.append(Play(t=pt, team=0, card=card, x=float(p["nx"]), y=float(p["ny"])))

    traj.plays.sort(key=lambda p: p.t)
    traj.duration = traj.samples[-1].t if traj.samples else 0.0
    traj.first_tower_t = first_tower_t
    if traj.outcome is None:                    # scoreboard never read -> fall back to towers felled
        t_blue, t_red, ek, mk = tower.crown_counts()
        traj.outcome = ("win" if (ek or t_blue > t_red) else
                        "loss" if (mk or t_red > t_blue) else "draw")
    return traj


# --- REPLAY: drive the engine with the same script ---------------------------------------------
def replay_engine(cfg, db, script: List[Play], duration: float, stride_s: float = 1.0,
                  opponent_deck: Optional[List[str]] = None, seed: int = 0,
                  spec_patch=None, engine_patch=None) -> Trajectory:
    """Run the engine through the same play script and record the same observables.

    `spec_patch(spec)` and `engine_patch(engine)` are the MECHANIC ABLATION hooks -- they mutate the
    built card specs / the engine before the replay so one mechanic can be turned off and the replay
    re-scored (see MECHANICS).
    """
    import random

    from .sim.engine import SimEngine, build_spec

    rng = random.Random(seed)
    eng = SimEngine(cfg, db, rng)
    eng.reset()
    if engine_patch is not None:
        engine_patch(eng)

    my_levels = {k: lvl for k, lvl in zip(db.deck_identities(), db.deck_levels())}
    enemy_level = int(cfg.get("divergence", "enemy_card_level",
                              default=cfg.get("sim", "my_tower_level", default=15)))
    specs: Dict[Tuple[int, str], object] = {}

    def spec_for(team: int, card: str):
        key = (team, card)
        if key not in specs:
            lvl = my_levels.get(card, 11) if team == 0 else enemy_level
            s = build_spec(db, card, lvl)
            if spec_patch is not None:
                spec_patch(s)
            specs[key] = s
        return specs[key]

    traj = Trajectory()
    sub_dt = float(cfg.get("sim", "sub_dt", default=0.1))
    pending = sorted(script, key=lambda p: p.t)
    pi = 0
    next_sample = 0.0
    first_tower_t = None
    my_alive_prev = [True, True, True]
    en_alive_prev = [True, True, True]

    while eng.t < duration and not eng.done:
        while pi < len(pending) and pending[pi].t <= eng.t:
            p = pending[pi]
            pi += 1
            s = spec_for(p.team, p.card)
            # The engine is being asked to reproduce a match that HAPPENED, so a play must land even if
            # the engine's own elixir accounting disagrees -- otherwise elixir drift silently eats the
            # script and we would be measuring "the sim refused to play cards", not fidelity.
            eng.elixir[p.team] = max(eng.elixir[p.team], float(s.elixir))
            eng.deploy(p.team, s, p.x, p.y)
        eng.advance(sub_dt)

        if eng.t >= next_sample:
            next_sample += stride_s
            traj.samples.append(Sample(
                t=eng.t,
                my_hp=[(tw.hp / tw.max_hp if tw.max_hp else 0.0) for tw in eng.towers[0]],
                enemy_hp=[(tw.hp / tw.max_hp if tw.max_hp else 0.0) for tw in eng.towers[1]],
                my_units=sum(1 for u in eng.units if u.team == 0 and u.hp > 0),
                enemy_units=sum(1 for u in eng.units if u.team == 1 and u.hp > 0)))
        for i in range(3):
            if my_alive_prev[i] and not eng.towers[0][i].alive and first_tower_t is None:
                first_tower_t = eng.t
            if en_alive_prev[i] and not eng.towers[1][i].alive and first_tower_t is None:
                first_tower_t = eng.t
        my_alive_prev = [tw.alive for tw in eng.towers[0]]
        en_alive_prev = [tw.alive for tw in eng.towers[1]]

    traj.plays = list(script)
    traj.duration = eng.t
    traj.first_tower_t = first_tower_t
    traj.outcome = eng.outcome if eng.done else _score_from_towers(eng)
    return traj


def _score_from_towers(eng) -> str:
    mine = sum(1 for t in eng.towers[0] if not t.alive)
    theirs = sum(1 for t in eng.towers[1] if not t.alive)
    if theirs != mine:
        return "win" if theirs > mine else "loss"
    my_min = min((t.hp / t.max_hp for t in eng.towers[0] if t.max_hp > 0), default=1.0)
    op_min = min((t.hp / t.max_hp for t in eng.towers[1] if t.max_hp > 0), default=1.0)
    if abs(my_min - op_min) < 1e-3:
        return "draw"
    return "win" if op_min < my_min else "loss"


# --- SCORE -------------------------------------------------------------------------------------
def score(real: Trajectory, sim: Trajectory, cfg=None) -> Dict[str, float]:
    """Component divergences (each ~[0,1]) plus the weighted composite."""
    w = {"tower_hp": 0.45, "units": 0.20, "first_tower": 0.15, "outcome": 0.20}
    if cfg is not None:
        w.update(cfg.get("divergence", "weights", default={}) or {})

    # tower HP: mean |fraction error| over the shared time grid, all six towers
    hp_err, n = 0.0, 0
    for s in real.samples:
        m = sim.at(s.t)
        if m is None:
            continue
        for a, b in zip(s.my_hp, m.my_hp):
            hp_err += abs(a - b); n += 1
        for a, b in zip(s.enemy_hp, m.enemy_hp):
            hp_err += abs(a - b); n += 1
    tower_hp = hp_err / n if n else float("nan")

    # unit counts: normalized by a typical push size, so "off by one troop" is small
    u_err, un = 0.0, 0
    for s in real.samples:
        if s.my_units is None:
            continue
        m = sim.at(s.t)
        if m is None or m.my_units is None:
            continue
        u_err += abs(s.my_units - m.my_units) + abs(s.enemy_units - m.enemy_units)
        un += 2
    units = min(1.0, (u_err / un) / 4.0) if un else float("nan")

    # time-to-first-tower, normalized by match length
    span = max(1.0, real.duration)
    if real.first_tower_t is None and sim.first_tower_t is None:
        first_tower = 0.0
    elif real.first_tower_t is None or sim.first_tower_t is None:
        first_tower = 1.0                      # one side lost a tower and the other never did
    else:
        first_tower = min(1.0, abs(real.first_tower_t - sim.first_tower_t) / span)

    outcome = 0.0 if (real.outcome == sim.outcome) else 1.0

    parts = {"tower_hp": tower_hp, "units": units, "first_tower": first_tower, "outcome": outcome}
    total, wsum = 0.0, 0.0
    for k, v in parts.items():
        if v == v:                             # skip NaN (metric unavailable)
            total += w.get(k, 0.0) * v
            wsum += w.get(k, 0.0)
    parts["divergence"] = total / wsum if wsum else float("nan")
    parts["_weight_covered"] = wsum
    return parts


# --- MECHANIC ABLATIONS ------------------------------------------------------------------------
def _patch_straight_pathing(eng):
    """PATHING: ground units cross the river only at a bridge. Ablate = walk straight through it."""
    from .sim import engine as E

    def straight(u, tx, ty, dt, spd_mult=1.0):
        d = E._dist(u.x, u.y, tx, ty)
        if d < 1e-6:
            return
        step = min(u.spec.speed * spd_mult * dt, d)
        u.x += (tx - u.x) / d * step
        u.y += (ty - u.y) / d * step

    # assigned as a plain instance attribute (NOT bound) so it shadows the class method with the
    # same call signature the engine uses internally
    eng._move_toward = straight


MECHANICS: Dict[str, dict] = {
    # name -> how to ablate it, and what the ablation means
    "pathing": {
        "why": "bridge routing -- ground troops must cross at a bridge",
        "engine": _patch_straight_pathing,
    },
    "aggro": {
        "why": "sight range -- troops notice enemy UNITS within it, else march at the tower",
        "engine": lambda e: setattr(e, "sight_range", 0.0),
        "spec": lambda s: setattr(s, "sight", 0.0),
    },
    "building_pull": {
        "why": "building-targeting troops (Hog/Miner) ignore troops entirely",
        "spec": lambda s: setattr(s, "building_only", False),
    },
    "body_block": {
        "why": "soft collision -- a wall of troops physically holds up a tank",
        "engine": lambda e: setattr(e, "collide", False),
    },
    "deploy_delay": {
        "why": "the ~1s spawn delay before a troop can act",
        "spec": lambda s: setattr(s, "deploy_time", 0.0),
    },
    "tower_first_hit": {
        "why": "crown towers wait `first_hit` before their first shot on a new target",
        "engine": lambda e: [setattr(t, "first_hit", 0.0)
                             for team in (0, 1) for t in e.towers[team]],
    },
    "siege_sight": {
        "why": "siege buildings (X-Bow) see much further than normal troops",
        "engine": lambda e: setattr(e, "siege_sight", e.sight_range),
    },
    "spell_delay": {
        "why": "spells land after a travel/cast delay rather than instantly",
        "spec": lambda s: setattr(s, "spell_delay", 0.0),
    },
}


def _charge_gap(script: List[Play], db) -> List[str]:
    """Cards in this match whose real CHARGE mechanic the engine does not model at all.

    Not an ablation -- there is no knob, because `sim/engine.py` states in its own docstring that
    "card-specific quirks (charge / ramp-up) are still out of scope". An unmodelled mechanic cannot be
    turned off, so it is reported as a structural gap instead of a sensitivity number.
    """
    played = {p.card for p in script}
    return sorted(c for c in played if (c[:-4] if c.endswith("_evo") else c) in CHARGE_CARDS)


# --- the command --------------------------------------------------------------------------------
def _latest_session(root: Path) -> Optional[Path]:
    cands = sorted((p for p in root.glob("*") if (p / "meta.json").exists()),
                   key=lambda p: p.name)
    return cands[-1] if cands else None


def divergence(cfg, session_arg: Optional[str] = None, stride: float = 1.0,
               conf: Optional[float] = None, all_mechanics: bool = False,
               out: Optional[str] = None, verbose: bool = False) -> None:
    from .cards import CardDB

    root = Path(cfg.path(cfg.get("record", "out_dir", default="data/sessions")))
    if session_arg and Path(session_arg).exists():
        session = Path(session_arg)
    elif session_arg:
        session = root / session_arg
    else:
        session = _latest_session(root)
    if session is None or not session.exists():
        print(f"[divergence] no session found under {root} -- record one first (run.py record)")
        return

    print(f"[divergence] session {session.name}")
    real = observe_session(cfg, session, stride_s=stride, conf=conf, verbose=verbose)
    if real is None:
        return
    for note in real.notes:
        print(f"[divergence] WARNING: {note}")
    mine = sum(1 for p in real.plays if p.team == 0)
    theirs = sum(1 for p in real.plays if p.team == 1)
    print(f"[divergence] sampling every {stride:.2f}s -- drift shorter than that is invisible to the "
          f"score by construction")
    print(f"[divergence] observed {len(real.samples)} samples over {real.duration:.0f}s; "
          f"script = {mine} your plays + {theirs} inferred opponent plays; outcome {real.outcome}")
    if not real.samples:
        print("[divergence] nothing observed -- cannot score")
        return

    db = CardDB(cfg)
    base_sim = replay_engine(cfg, db, real.plays, real.duration, stride_s=stride)
    base = score(real, base_sim, cfg)
    _print_score("BASELINE", base, real, base_sim)

    # --- mechanic sensitivity ---
    names = list(MECHANICS) if all_mechanics else [
        "pathing", "aggro", "building_pull", "body_block"]
    rows = []
    for name in names:
        spec = MECHANICS[name]
        sim_i = replay_engine(cfg, db, real.plays, real.duration, stride_s=stride,
                              spec_patch=spec.get("spec"), engine_patch=spec.get("engine"))
        sc = score(real, sim_i, cfg)
        # An ablation that changes NOTHING means this match never exercised the mechanic (no
        # building-targeter played, no ground troop that had to path to a bridge...). Its row would
        # otherwise read as "this mechanic is fine", which is the opposite of what it means.
        # Checked at PHYSICS resolution, not the reporting stride: a sub-stride effect (a 0.4s spell
        # delay against 1s sampling) is invisible to the score but is emphatically not "unexercised".
        fine = float(cfg.get("sim", "sub_dt", default=0.1))
        a = replay_engine(cfg, db, real.plays, real.duration, stride_s=fine)
        b = replay_engine(cfg, db, real.plays, real.duration, stride_s=fine,
                          spec_patch=spec.get("spec"), engine_patch=spec.get("engine"))
        unexercised = score(a, b, cfg)["divergence"] < 1e-9
        rows.append((sc["divergence"] - base["divergence"], name, sc, spec["why"], unexercised))
        print(f"[divergence]   ablated {name:16} -> divergence {sc['divergence']:.4f} "
              f"({sc['divergence'] - base['divergence']:+.4f})"
              + ("   [NOT EXERCISED by this match]" if unexercised else ""), flush=True)

    # Ranked by delta ASCENDING -- most NEGATIVE first. Ranking by |delta| looks reasonable and is
    # wrong: a load-bearing mechanic (siege sight in an X-Bow match) always has a huge POSITIVE delta
    # because ablating it wrecks the replay, so magnitude-ranking puts it on top of every list and
    # buries the actual explanation. What "best explains the drift" means is precisely the opposite --
    # ablating it moved the sim TOWARD reality.
    rows.sort(key=lambda r: r[0])
    print()
    print("=" * 96)
    print("WHAT BEST EXPLAINS THE DRIFT -- ranked by change in divergence when the mechanic is ABLATED")
    print("(most NEGATIVE first: ablating it moved the sim TOWARD reality)")
    print("=" * 96)
    print(f"{'mechanic':<18}{'divergence':>12}{'delta':>10}   what it models")
    print("-" * 96)
    for delta, name, sc, why, unexercised in rows:
        tag = "  [NOT EXERCISED]" if unexercised else ""
        print(f"{name:<18}{sc['divergence']:>12.4f}{delta:>+10.4f}   {why}{tag}")
    print("-" * 96)
    print("NEGATIVE delta -> a CANDIDATE EXPLANATION: switching this mechanic off made the sim match")
    print("  reality better, so the way it is currently modelled is doing net harm. These are the rows")
    print("  to investigate, in order.")
    print("LARGE POSITIVE delta -> the opposite: the trajectory leans heavily on this mechanic and")
    print("  removing it wrecks the match. That means it is load-bearing, NOT that it is wrong.")
    print("This ranks SENSITIVITY, not blame -- it says where to look, not what is broken. Confirm a")
    print("candidate on more than one match before acting on it; a single match is one sample.")
    print("A row marked [NOT EXERCISED] means this match never used the mechanic at all -- its delta is")
    print("zero for that reason, not because the mechanic is accurate.")

    gap = _charge_gap(real.plays, db)
    print()
    if gap:
        print(f"[divergence] UNMODELLED -- CHARGE: {', '.join(gap)} were played in this match and the")
        print("[divergence] engine models no charge/ramp-up at all (see sim/engine.py's docstring), so")
        print("[divergence] there is no knob to ablate. Their drift is invisible to the ranking above.")
    else:
        print("[divergence] no charge-mechanic cards were played, so that unmodelled gap is not a")
        print("[divergence] factor in THIS match (it still is in any match with Prince/Hog/Ram/etc).")

    if out:
        path = Path(cfg.path(out))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "session": session.name, "baseline": base, "notes": real.notes,
            "mechanics": [{"name": n, "divergence": s["divergence"], "delta": d, "why": w,
                           "exercised": not ue} for d, n, s, w, ue in rows],
            "unmodelled_charge_cards": gap,
        }, indent=2))
        print(f"[divergence] wrote {path}")


def _print_score(label: str, sc: Dict[str, float], real: Trajectory, sim: Trajectory) -> None:
    def f(v):
        return "  n/a" if v != v else f"{v:.4f}"

    print(f"[divergence] {label}: divergence {f(sc['divergence'])}  "
          f"(tower_hp {f(sc['tower_hp'])}, units {f(sc['units'])}, "
          f"first_tower {f(sc['first_tower'])}, outcome {f(sc['outcome'])})")
    rt = "never" if real.first_tower_t is None else f"{real.first_tower_t:.0f}s"
    st = "never" if sim.first_tower_t is None else f"{sim.first_tower_t:.0f}s"
    print(f"[divergence]   time-to-first-tower  real {rt}  vs sim {st}   |   "
          f"outcome  real {real.outcome}  vs sim {sim.outcome}")
    if sc["_weight_covered"] < 0.99:
        print(f"[divergence]   NOTE: only {sc['_weight_covered']:.0%} of the metric weight was "
              f"available (a component is n/a) -- the composite is renormalized over what was measured")
