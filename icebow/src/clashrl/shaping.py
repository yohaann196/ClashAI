"""POTENTIAL-BASED reward shaping (Ng, Harada & Russell 1999), replacing the hand-tuned term pile.

THE THEOREM. Adding an arbitrary shaping reward F to a task changes what the agent learns -- it can and
does learn to farm F instead of solving the task. Ng et al. proved that F is guaranteed to leave the
OPTIMAL POLICY UNCHANGED if and only if it is potential-based:

    F(s, a, s') = gamma * Phi(s') - Phi(s)

for some Phi over states. The guarantee is worth restating precisely, because it is stronger than it
first looks: the discounted sum of F over ANY trajectory telescopes to `-Phi(s_0)` -- a constant fixed
by the start state, identical for every policy. Shaping therefore cannot change which policy is best,
only how fast the agent finds it. There is nothing to farm, by construction.

WHAT THIS REPLACES. The current terms are NOT potential-based, including the two the codebase already
describes that way:

* `elixir_trade` uses `prev_enemy_value - enemy_value` -- a potential DIFFERENCE, but it (a) omits the
  `gamma` factor, (b) CLIPS the result to +-trade_cap, which destroys the telescoping the invariance
  depends on, and (c) adds `- spent`, an action cost that is not a function of state at all.
* `tower_chip_scale` uses `chip_progress` deltas. This is the closest thing in the codebase to a correct
  potential term -- it is a clean, unclipped state difference -- but it too omits `gamma`.

The remaining terms are action-scored: `threat_response` pays out for the ACT of playing a KB-correct
counter, `wincon_exec` for the ACT of placing the X-Bow well. Those are exactly the farmable kind, which
is why `correctness_cap` had to be invented to stop the policy grinding them. UNDER POTENTIAL SHAPING
THE CAP IS UNNECESSARY AND HARMFUL: unnecessary because the theorem already forbids farming, harmful
because a path-dependent clamp is not a function of state and breaks the invariance it is bolted onto.
`potential_based.enabled` therefore disables it.

THE POTENTIALS. Each is a function of STATE ONLY, weighted by the SAME config keys as the term it
replaces, so ablation numbers carry over:

  trade   (my_elixir - enemy_remaining_value) / value_norm      <- material advantage
  chip    chip_progress(theirs) - chip_progress(mine)           <- the existing potential, gamma-fixed
  threat  -(severity of threats you have no answer to)          <- replaces threat_response/threat_miss
  wincon  +(win condition correctly established for the phase)  <- replaces wincon_exec/wincon_misplace
  cycle   +(you hold an answer to the current threat)           <- replaces cycle_plan/cycle_waste
  leak    -(elixir sitting at cap)                              <- replaces leak_penalty

Note what `trade` does to the old `- spent` action cost: your own elixir is now IN the potential, so
committing 6 elixir drops Phi by 6/value_norm and regenerating raises it back. Spending is no longer
penalised as such -- only spending that fails to buy board advantage is, which is the honest statement
of the thing `- spent` was gesturing at.

TERMS THAT DO NOT CONVERT, and why that is fine:

* `spell_waste` scores an ACTION (a spell cast into emptiness), not a state. It needs no conversion
  because it is SUBSUMED: a whiffed spell drops `trade` (elixir gone) and moves nothing else, so the
  potential already reports it as a loss.
* `nado_clump` / `nado_combo` / `nado_retarget` are measured 2-3.5s after the cast from a watch list --
  functions of HISTORY, not state, so no Phi over states can express them. They are likewise subsumed:
  a pull that clumps enemies into a rocket shows up as enemy value collapsing (`trade`), and a pull that
  drags a wincon off a tower shows up as threat severity dropping (`threat`).
* `nado_king_activate` is the one genuinely NOT subsumed -- waking your own king is not visible in any
  of the potentials above. If the ablation shows it earns its place it must stay a non-potential term,
  and it should be declared as such rather than quietly dropped.

The sparse `win` / `loss` / `take_enemy_tower` / `lose_own_tower` terms are the REAL task reward and are
deliberately left alone -- they are what the shaping is shaping toward.

STATUS: default OFF. Which terms survive is an empirical question that `run.py ablate-rewards` answers
and this module cannot; turn components on for the terms that earn their place.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import card_threat

# component name -> the reward config key whose weight it inherits
COMPONENTS = ("trade", "chip", "threat", "wincon", "cycle", "leak")
DEFAULT_COMPONENTS = COMPONENTS


def enabled(cfg) -> bool:
    pb = cfg.get("rewards", "potential_based", default={}) or {}
    return bool(pb.get("enabled", False))


def components(cfg) -> List[str]:
    pb = cfg.get("rewards", "potential_based", default={}) or {}
    sel = pb.get("components", None)
    if not sel:
        return list(DEFAULT_COMPONENTS)
    bad = [c for c in sel if c not in COMPONENTS]
    if bad:
        raise ValueError(f"rewards.potential_based.components: unknown {bad}; valid: {list(COMPONENTS)}")
    return list(sel)


class PotentialShaper:
    """Computes `F = gamma * Phi(s') - Phi(s)` for the sim env.

    Usage mirrors the invariance requirement exactly:
        shaper.reset(env)                 # latches Phi(s_0); NO reward emitted
        f = shaper.step(env, done)        # F for the transition into the current state

    `done` matters: episodic potential shaping is only invariant when the absorbing state has
    `Phi = 0`, so the terminal transition must emit `-Phi(s)` rather than `gamma*Phi(s') - Phi(s)`.
    Getting this wrong adds a policy-dependent constant to the return -- the exact failure the
    theorem is supposed to rule out.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.gamma = float(cfg.get("train", "gamma", default=0.99))
        self.components = components(cfg)
        r = lambda k, d: float(cfg.get("rewards", k, default=d))  # noqa: E731
        # weights are the SAME config keys the replaced terms used, so an ablation delta measured on
        # the classic form is still meaningful after conversion
        self.w_trade = r("elixir_trade", 1.0)
        self.w_chip = r("tower_chip_scale", 0.3)
        self.w_threat = r("threat_response", 1.0)
        self.w_wincon = r("wincon_exec", 0.8)
        self.w_cycle = r("cycle_plan", 0.4)
        self.w_leak = r("leak_penalty", -0.2)
        self.value_norm = float(cfg.get("env", "value_norm", default=10.0))
        self._prev = 0.0

    # -- the potential ---------------------------------------------------------------------------
    def _phi_trade(self) -> float:
        """MATERIAL ADVANTAGE: your banked elixir minus the enemy's remaining troop value. Killing
        enemy value raises it; committing elixir lowers it and regeneration restores it -- which is why
        no separate `- spent` action cost is needed (or wanted: it was never a state function)."""
        env = self.env
        return (env.eng.elixir[0] - env._enemy_value()) / self.value_norm

    def _phi_chip(self) -> float:
        """Convex tower-chip differential -- the codebase's existing potential, unchanged in shape."""
        env = self.env
        return env._chip_progress(env.eng.towers[1]) - env._chip_progress(env.eng.towers[0])

    def _phi_threat(self) -> float:
        """NEGATIVE severity of enemy pressure you hold no answer to.

        Severity weights each enemy unit on your half by its remaining value and how deep it has come.
        It is scaled by how well-answered the push is: holding (or having fielded) a KB-correct counter
        drives the unanswered fraction toward zero. Playing the right counter therefore RAISES Phi on
        its own -- which is what `threat_response` was paying an action bonus for -- and ignoring an
        answerable push lets Phi fall as the push deepens, which is what `threat_miss` was for."""
        env = self.env
        sev = 0.0
        for u in env.eng.units:
            if u.team != 1 or u.hp <= 0 or u.spec.kind == "spell" or u.y < 0.5:
                continue
            frac = max(0.0, min(1.0, u.hp / u.spec.hp)) if u.spec.hp > 0 else 1.0
            depth = max(0.0, min(1.0, (u.y - 0.5) / 0.5))
            sev += (u.spec.elixir / max(1, u.spec.count)) * frac * (0.5 + 0.5 * depth)
        if sev <= 0.0:
            return 0.0
        tid = getattr(env, "_threat_id", None)
        answered = 0.0
        if tid is not None and len(tid) >= card_threat.IDENTITY_DIM and tid[0] >= 0.5:
            # an answer counts if it is IN HAND and affordable, or already on the board defending
            for cid in env._hand_ids():
                if (card_threat.counters(env._deck_profiles[cid], tid)
                        and env.specs[cid].elixir <= env.eng.elixir[0]):
                    answered = 1.0
                    break
            if answered < 1.0:
                for u in env.eng.units:
                    if u.team == 0 and u.hp > 0 and u.y >= 0.5:
                        base = getattr(u.spec, "base", None)
                        if base and card_threat.counters(card_threat.profile(env.db, base), tid):
                            answered = 1.0
                            break
        return -(sev / self.value_norm) * (1.0 - answered)

    def _phi_wincon(self) -> float:
        """+1 (scaled) while the deck's win condition is CORRECTLY ESTABLISHED for the current phase:
        an alive X-Bow forward-in-range while offensive, or in the centre intercept band while
        defensive. Placing it well raises Phi and it decays back when the building dies -- so a good
        placement is rewarded without paying a bonus for the ACT, and a misplacement simply never
        raises Phi (no separate `wincon_misplace` penalty required)."""
        env = self.env
        best = 0.0
        princesses = [t for t in env.eng.towers[1][:2] if t.alive]
        for u in env.eng.units:
            if u.team != 0 or u.hp <= 0:
                continue
            if getattr(u.spec, "base", None) != "x_bow":
                continue
            frac = max(0.0, min(1.0, u.hp / u.spec.hp)) if u.spec.hp > 0 else 1.0
            central = abs(u.x - 0.48) <= 0.18
            if env._defensive:
                in_band = central and env.xbow_front <= u.y <= env.xbow_back
                v = 1.0 if in_band else (env.xbow_deep_frac if central and u.y > env.xbow_back else 0.0)
            else:
                d = min((np.hypot(u.x - t.x, u.y - t.y) for t in princesses), default=1.0)
                v = 1.0 if d <= env.xbow_range else 0.0
            best = max(best, v * frac)
        return best

    def _phi_cycle(self) -> float:
        """+1 (scaled) while you HOLD an answer to the currently assessed threat. Cycling toward a
        needed counter raises Phi the moment it reaches hand, which is the state `cycle_plan` was
        trying to pay for -- without paying for cheap plays that merely look like cycling."""
        env = self.env
        tid = getattr(env, "_threat_id", None)
        if tid is None or len(tid) < card_threat.IDENTITY_DIM or tid[0] < 0.5:
            return 0.0
        return 1.0 if any(card_threat.counters(env._deck_profiles[c], tid)
                          for c in env._hand_ids()) else 0.0

    def _phi_leak(self) -> float:
        """NEGATIVE while elixir sits at the cap (the bar is overflowing and the surplus is lost)."""
        return -max(0.0, self.env.eng.elixir[0] - 9.5) / 0.5

    def phi(self) -> float:
        total = 0.0
        for c in self.components:
            if c == "trade":
                total += self.w_trade * self._phi_trade()
            elif c == "chip":
                total += self.w_chip * self._phi_chip()
            elif c == "threat":
                total += self.w_threat * self._phi_threat()
            elif c == "wincon":
                total += self.w_wincon * self._phi_wincon()
            elif c == "cycle":
                total += self.w_cycle * self._phi_cycle()
            elif c == "leak":
                # w_leak is configured NEGATIVE and _phi_leak is already negative-going; use the
                # magnitude so a fuller bar is a LOWER potential rather than a higher one.
                total += abs(self.w_leak) * self._phi_leak()
        return float(total)

    # -- the shaping reward ----------------------------------------------------------------------
    def reset(self) -> None:
        """Latch Phi(s_0). Emits nothing: the first F belongs to the first TRANSITION."""
        self._prev = self.phi()

    def step(self, done: bool) -> float:
        """F for the transition that just landed in the env's current state.

        Terminal states get `Phi = 0` (absorbing), so the last transition emits `-Phi(s)`. This is what
        makes the discounted sum of F over a whole episode telescope to exactly `-Phi(s_0)` -- verified
        numerically by tools/potential_invariance.py."""
        cur = 0.0 if done else self.phi()
        f = self.gamma * cur - self._prev
        self._prev = cur
        return float(f)

    def breakdown(self) -> Dict[str, float]:
        """Per-component contribution to the current Phi (diagnostics only)."""
        fns = {"trade": (self.w_trade, self._phi_trade), "chip": (self.w_chip, self._phi_chip),
               "threat": (self.w_threat, self._phi_threat), "wincon": (self.w_wincon, self._phi_wincon),
               "cycle": (self.w_cycle, self._phi_cycle), "leak": (abs(self.w_leak), self._phi_leak)}
        return {c: fns[c][0] * fns[c][1]() for c in self.components}
