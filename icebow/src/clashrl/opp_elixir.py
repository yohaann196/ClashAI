"""OPPONENT ELIXIR + CYCLE estimation -- the two things the policy has never been able to see.

A human plays Clash Royale by counting the opponent's elixir. It is the difference between "commit the
X-Bow now" and "they have 9 elixir and a Rocket, wait". The policy has been blind to it: it observes
its own elixir exactly and the opponent's not at all, so every decision about WHEN to commit has been
made with half the information a player uses.

THE ESTIMATOR. Elixir is an accumulator with known dynamics, so it can be integrated:

    e(t) = clamp(e0 + integral(rate) - sum(costs), 0, 10)

`e0 = 5` and `rate` are exactly known (they are the same for both players, and `elixir_rate()` already
models the 1x/2x/3x tiers). The only unknown term is `sum(costs)` -- which cards they played, and when.
Live that comes from the detector, which MISSES things. So this deliberately does NOT produce a point
estimate:

    hi   an UPPER BOUND under RECALL error: only spends we observed reduce it, and a spend we MISSED
         can only make the truth lower, so dropping detections never breaks it. It is NOT sound under
         PRECISION error -- misreading a 3-cost card as a 6-cost one subtracts too much and pushes `hi`
         below the truth. That asymmetry is real and worth knowing: `hi` is a hard bound against the
         detector's misses and a soft one against its confusions.
    lo   `hi` minus an allowance for the spends we expect to have missed, sized from the detector's
         MEASURED RECALL: observing k deploys at recall r implies ~k/r really happened, so the unobserved
         spend is ~(1/r - 1) x the cost we did observe.
    conf 1 - (hi - lo)/10. A wide interval is the estimator telling the policy not to trust it.

MEASURED REALITY CHECK, because it changes how to read this feature: `detector_cards` covers only a
slice of the meta, and a sampled sim rollout had 1 of the opponent's 8 cards on the whitelist -- the
estimator observed 0 of 33 deploys. So for now the interval is usually WIDE and `confidence` is usually
LOW, and the feature's real value is (a) the cap detection -- knowing they are sitting at 10 is a
punish window and is recoverable with no detections at all -- and (b) telling the policy, honestly,
when it does not know. It gets sharper for free as the detector's whitelist grows.

DEPLOY-DELAY CORRECTION, and why the ORDER matters. A troop becomes visible ~1s after the elixir left
their bar, so a spawn detected at `t` was paid for at `t - deploy_delay`. Simply subtracting the cost on
detection happens to give the right value AT detection time (the regen over the delay applies to truth
and estimate alike) -- but it is wrong at the CAP. An opponent sitting at 10 who spends 5 is at 5+regen
when we see it; if we let the estimate ride at the 10-clamp through the delay and only then subtract,
the clamp has already eaten elixir that was really spent. So `observe_deploy` rewinds the regen over the
delay, applies the cost WITH clamping at that earlier point, then re-applies the regen.

SELF-CORRECTION AT THE CAP. The 10-elixir cap is an absorbing state that ERASES HISTORY: after enough
idle time both players are at 10 regardless of what they spent earlier. So when `hi` has been pinned at
10 with no observed deploy for `cap_hold_s`, the interval collapses to a point and confidence returns to
1 -- the estimator re-synchronises with reality for free, which is what stops missed deploys accumulating
into permanent uncertainty over a 3-minute match.

CYCLE. `cycle.CycleTracker` reconstructs YOUR 8-card queue from hand/next recognition. The opponent's
hand is never visible, so :class:`EnemyCycleTracker` bootstraps from NOTHING -- an all-unknown queue of
`-1` slots (which `cycle_vector` already handles) that fills in as cards are seen played, and rotates
each played card to the back exactly as the real queue does. After ~8 observed plays it knows their deck
and can say how soon their win condition comes back around.

SIM PARITY IS THE POINT. The sim runs THIS SAME ESTIMATOR, fed deploys corrupted to the detector's
measured recall and precision -- never engine ground truth. Handing the policy exact opponent elixir in
training would teach it to lean on precision it does not have live, and the failure would only show up
on the ladder. The corruption draws from a PRIVATE rng (the 5cdf867 DomainRand pattern) so seeded eval
streams stay bit-identical whether this feature is on or off.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import card_threat

OPP_ELIXIR_DIM = 8         # width appended to the threat vector when observation.use_opponent_elixir
_MAX_ELIXIR = 10.0
_START_ELIXIR = 5.0
_DEFAULT_WINCON_COST = 5.0  # meta-typical win-condition cost, used until we have seen theirs
_DECK_SIZE = 8


class OpponentElixir:
    """Interval estimate `[lo, hi]` of the opponent's elixir, driven by OBSERVED enemy deployments.

    Feed it `regen(dt, rate)` every step and `observe_deploy(base_card)` whenever a deploy is detected.
    Never feed it ground truth -- see the module docstring.
    """

    def __init__(self, cfg, db):
        self.db = db
        oe = cfg.get("observation", "opponent_elixir", default={}) or {}
        self.deploy_delay = float(oe.get("deploy_delay_s", 1.0))
        self.cap_hold_s = float(oe.get("cap_hold_s", 3.0))
        # The recall the interval width is sized from. Defaults to the SAME measured figure the sim's
        # detector-realism model uses, so live and sim agree about how uncertain to be.
        # `assumed_recall: null` in config means "follow the measured sim_detector_recall" -- an
        # explicit null is a present key with a None value, so `.get(k, default)` would hand back None.
        _r = oe.get("assumed_recall", None)
        if _r is None:
            _r = cfg.get("observation", "sim_detector_recall", default=1.0)
        self.recall = float(_r)
        self.max_allowance = float(oe.get("max_allowance", _MAX_ELIXIR))
        # BLINDNESS RATE -- and it is the dominant term, not a refinement. The recall term below only
        # covers deploys the detector saw and dropped. It does NOT cover cards outside `detector_cards`,
        # which the detector cannot name AT ALL, and measurement says that is most of them: on a sampled
        # sim rollout the opponent's meta deck had 1 of 8 cards on the whitelist and the estimator
        # observed 0 of 33 deploys. So the interval must also widen with the elixir they have EARNED
        # since the last confident sync -- they cannot have spent more than that, which makes
        # `blind_spend_frac: 1.0` a genuine bound rather than a tuning guess. Lower it only when the
        # detector's whitelist actually covers the meta; the cap re-sync is what closes it again.
        self.blind_frac = float(oe.get("blind_spend_frac", 1.0))
        self.reset()

    def reset(self) -> None:
        self._lo = self._hi = _START_ELIXIR
        self._observed_cost = 0.0        # total elixir we have SEEN them spend
        self._n_deploys = 0
        self._idle_at_cap = 0.0          # seconds `hi` has been pinned at the cap with no observed deploy
        self._seen_costs: Dict[str, float] = {}
        self._wincon_cost: Optional[float] = None
        self._regen_since_sync = 0.0     # elixir they have EARNED since the last confident sync

    # -- dynamics ---------------------------------------------------------------------------------
    def regen(self, dt: float, rate: float, quiet: bool = False) -> None:
        """Advance both bounds by `rate * dt`, clamped at the cap.

        `quiet` = no enemy presence observed on the board right now. It gates the cap re-sync, and the
        distinction matters: "no deploy OBSERVED" is not "no deploy HAPPENED". With a whitelist that
        covers a slice of the meta the estimator can observe nothing for a whole match while the
        opponent plays continuously, and treating that silence as idleness would snap the interval
        confidently to 10 exactly when it is most wrong. Idleness has to be evidenced by an empty
        board, not inferred from our own blindness.
        """
        if dt <= 0:
            return
        gain = rate * dt
        self._regen_since_sync += gain
        self._hi = min(_MAX_ELIXIR, self._hi + gain)
        self._lo = min(_MAX_ELIXIR, self._lo + gain)
        self._apply_miss_allowance()
        if self._hi >= _MAX_ELIXIR - 1e-9 and quiet:
            self._idle_at_cap += dt
            # THE CAP RE-SYNCHRONISES. Sustained idle at the cap means they really are at 10 whatever
            # we missed earlier -- the cap erases history -- so collapse the interval and hand the
            # policy back a confident read.
            if self._idle_at_cap >= self.cap_hold_s:
                self._lo = self._hi = _MAX_ELIXIR
                self._observed_cost = 0.0
                self._n_deploys = 0
                self._regen_since_sync = 0.0            # a confident sync: the past no longer matters
        else:
            self._idle_at_cap = 0.0

    def observe_deploy(self, base_card: Optional[str], rate: float = 0.0) -> None:
        """Account for a DETECTED enemy deployment (deploy-delay corrected -- see the docstring).

        `base_card` None / unknown means we saw a spawn but could not name the card: we still know
        elixir left their bar, so the interval WIDENS by the plausible cost range instead of pretending
        to a number.
        """
        was_capped = self._idle_at_cap >= self.deploy_delay
        self._idle_at_cap = 0.0
        cost = self._cost_of(base_card)
        back = rate * self.deploy_delay          # regen accrued since the elixir actually left the bar

        # Rewind to the spend instant, apply the cost THERE (so the cap lands at the right time), then
        # roll the regen forward again. Rewinding a CLAMPED value would under-estimate the past: if the
        # bar has been pinned at 10 for at least the delay, they were at 10 when they paid -- subtracting
        # `back` from 10 would invent elixir they never lost.
        hi = self._hi if was_capped else max(0.0, self._hi - back)
        lo = self._lo if was_capped else max(0.0, self._lo - back)
        if cost is None:                          # unnamed spawn: bound it by the cheapest/dearest card
            cheap, dear = 1.0, 7.0
            hi = max(0.0, hi - cheap)
            lo = max(0.0, lo - dear)
            self._observed_cost += dear
        else:
            hi = max(0.0, hi - cost)
            lo = max(0.0, lo - cost)
            self._observed_cost += cost
        self._n_deploys += 1
        self._hi = min(_MAX_ELIXIR, hi + back)
        self._lo = min(_MAX_ELIXIR, lo + back)
        self._apply_miss_allowance()

    def _apply_miss_allowance(self) -> None:
        """Widen `lo` by the spend we expect to have MISSED, sized from the detector's recall.

        Observing k deploys at recall r implies ~k/r really happened, so the unobserved spend is about
        `(1/r - 1)` times the cost we did observe. At r = 1 this is zero and the interval is a point.
        """
        r = min(1.0, max(1e-3, self.recall))
        seen_miss = (1.0 / r - 1.0) * self._observed_cost      # deploys the detector dropped
        blind = self.blind_frac * self._regen_since_sync        # ...and cards it cannot name at all
        allowance = min(self.max_allowance, seen_miss + blind)
        self._lo = max(0.0, min(self._lo, self._hi - allowance))

    def _cost_of(self, base_card: Optional[str]) -> Optional[float]:
        if not base_card:
            return None
        base = card_threat.base_key(base_card)
        c = self.db.elixir(base)
        if c is None:
            return None
        c = float(c)
        self._seen_costs[base] = c
        prof = card_threat.profile(self.db, base)
        if prof.win_condition:
            self._wincon_cost = c if self._wincon_cost is None else min(self._wincon_cost, c)
        return c

    # -- read-out ---------------------------------------------------------------------------------
    @property
    def lo(self) -> float:
        return self._lo

    @property
    def hi(self) -> float:
        return self._hi

    @property
    def mid(self) -> float:
        return 0.5 * (self._lo + self._hi)

    @property
    def confidence(self) -> float:
        return max(0.0, 1.0 - (self._hi - self._lo) / _MAX_ELIXIR)

    @property
    def wincon_cost(self) -> float:
        return self._wincon_cost if self._wincon_cost is not None else _DEFAULT_WINCON_COST

    def can_afford(self, cost: float) -> float:
        """Graded 'can they pay for this RIGHT NOW', read CONSERVATIVELY off the bounds.

        1.0 = even the lower bound covers it (they certainly can), 0.0 = even the upper bound does not
        (they certainly cannot), and in between the fraction of the interval that does -- which is the
        honest answer when the estimate is uncertain.
        """
        if self._hi < cost:
            return 0.0
        if self._lo >= cost:
            return 1.0
        span = max(1e-6, self._hi - self._lo)
        return float(np.clip((self._hi - cost) / span, 0.0, 1.0))


class EnemyCycleTracker:
    """The opponent's 8-card queue, bootstrapped from an UNKNOWN deck.

    `cycle.CycleTracker` can read your own hand and Next off the screen. None of that exists for the
    opponent -- the only observable is which card they just played -- so this starts from a queue of
    `-1` unknowns (which `cycle_vector` already tolerates) and fills in as cards are seen. Each observed
    play rotates that card to the BACK, exactly as the real queue rotates, so once ~8 distinct cards
    have been seen the queue order is genuinely known and 'how soon does their win condition come back'
    becomes answerable.
    """

    def __init__(self, db):
        self.db = db
        self.reset()

    def reset(self) -> None:
        self._queue: List[int] = [-1] * _DECK_SIZE   # front(hand) -> back
        self._cards: List[str] = []                  # discovered deck, index = the id used in _queue

    @property
    def known_cards(self) -> List[str]:
        return list(self._cards)

    @property
    def coverage(self) -> float:
        """Fraction of their 8-card deck we have identified."""
        return min(1.0, len(self._cards) / float(_DECK_SIZE))

    def record_play(self, base_card: Optional[str]) -> None:
        """A card was seen played -> it rotates to the back of their queue."""
        if not base_card:
            return
        base = card_threat.base_key(base_card)
        if base not in self._cards:
            if len(self._cards) >= _DECK_SIZE:
                return                                # more than 8 distinct: detector noise, ignore
            self._cards.append(base)
        cid = self._cards.index(base)
        if cid in self._queue:
            self._queue.remove(cid)
        else:
            for i, slot in enumerate(self._queue):    # it came from an as-yet-unknown hand slot
                if slot == -1:
                    self._queue.pop(i)
                    break
            else:
                self._queue.pop(0)
        self._queue.append(cid)
        while len(self._queue) < _DECK_SIZE:
            self._queue.insert(0, -1)

    def soonness(self, base_card: str) -> float:
        """How soon a card comes back around: 1.0 = in hand / imminent, 0 = just played or unknown."""
        base = card_threat.base_key(base_card)
        if base not in self._cards:
            return 0.0
        cid = self._cards.index(base)
        if cid not in self._queue:
            return 0.0
        pos = self._queue.index(cid)
        if pos < 4:
            return 1.0                                 # in their hand right now
        depth = pos - 4                                # 0 = Next ... 3 = deepest
        return float(max(0.0, 1.0 - depth / max(1, _DECK_SIZE - 5)))

    def wincon(self) -> Optional[str]:
        for base in self._cards:
            if card_threat.profile(self.db, base).win_condition:
                return base
        return None

    def wincon_soonness(self) -> float:
        wc = self.wincon()
        return self.soonness(wc) if wc else 0.0


def features(est: OpponentElixir, cyc: EnemyCycleTracker, my_elixir: float) -> np.ndarray:
    """The `OPP_ELIXIR_DIM` block appended to the threat vector.

      0 lo / 10                      their elixir lower bound
      1 hi / 10                      ...and upper bound (a genuine bound, not a guess)
      2 confidence                   1 - interval width / 10; low = do not trust 0 and 1
      3 ELIXIR ADVANTAGE             (mine - their midpoint) / 10, shifted to [0,1] with 0.5 = level
      4 CAN THEY AFFORD THEIR WINCON graded off the interval (conservative)
      5 wincon soonness              how soon their win condition cycles back (0 if never seen)
      6 at the cap                   they have been sitting at 10 -- punish-window signal
      7 deck coverage                how much of their deck we have identified (how much to trust 5)
    """
    v = np.zeros(OPP_ELIXIR_DIM, np.float32)
    v[0] = est.lo / _MAX_ELIXIR
    v[1] = est.hi / _MAX_ELIXIR
    v[2] = est.confidence
    adv = (float(my_elixir) - est.mid) / _MAX_ELIXIR          # [-1, 1]
    v[3] = float(np.clip(0.5 + 0.5 * adv, 0.0, 1.0))
    v[4] = est.can_afford(est.wincon_cost)
    v[5] = cyc.wincon_soonness()
    v[6] = 1.0 if est.lo >= _MAX_ELIXIR - 1e-6 else 0.0
    v[7] = cyc.coverage
    return v


class DeployObserver:
    """Turns a per-frame enemy DETECTION list into discrete 'a card was just deployed' events.

    Live, the detector reports presence, not deploys: a unit is re-detected every frame it survives. A
    deploy is a base card appearing on THEIR half that was not there on the previous read. Coarse -- it
    misses a card replayed while a previous copy is still alive -- but nothing in the game surfaces the
    opponent's taps, so presence-differencing is the only signal there is, and the estimator's interval
    is sized for exactly this kind of miss.
    """

    def __init__(self, own_half_is_bottom: bool = True):
        self.bottom = own_half_is_bottom
        self.reset()

    def reset(self) -> None:
        self._prev: set = set()

    def step(self, enemy_dets: Sequence) -> List[str]:
        """`enemy_dets` = this frame's enemy detections (need `.base` and `.gy`). Returns new deploys."""
        now, fresh = set(), []
        for d in enemy_dets:
            base = card_threat.base_key(getattr(d, "base", "") or "")
            if not base:
                continue
            now.add(base)
            staging = getattr(d, "gy", 0.5) < 0.5 if self.bottom else getattr(d, "gy", 0.5) >= 0.5
            if base not in self._prev and staging:
                fresh.append(base)
        self._prev = now
        return fresh
